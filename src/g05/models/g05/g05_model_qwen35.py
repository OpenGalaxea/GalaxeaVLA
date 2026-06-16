"""
G05ModelQwen35 — Qwen3.5 variant of G05Model.

Inherits G05Model, overrides only Qwen3.5-specific behaviors:
  - No external vision projector (PatchMerger inside VisionModel)
  - Conv3D vision forward with image_grid_thw caching
  - MRoPE position_ids (3D) with reversed call order (embed → mask)
  - SparseKVCache for hybrid attention (full + linear)
  - Recurrent state transfer from VLM to Action Expert
  - Sparse KV prefix slicing in training forward
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import einops

from g05.models.kv_cache import SparseKVCache
from g05.utils.logging.log_box import log_box
from .g05_model import G05Model
from .helpers.mask_helper import MaskHelperQwen35
from .io.input_preprocessor import TOKEN_INDEX
from .model.utils import _load_all_safetensors

logger = logging.getLogger(__name__)


class G05ModelQwen35(G05Model):
    """Qwen3.5 variant: hybrid attention + MRoPE + Conv3D vision.

    Only overrides one class variable and three hooks; __init__ fully reuses
    G05Model, so new functionality is inherited automatically.

    Key differences from G05Model (PaliGemma):
    - mask_helper_cls = MaskHelperQwen35
    - No multi_modal_projector (PatchMerger inside VisionModel)
    - _forward_vision: Conv3D format + image_grid_thw caching
    - vlm_prefill: embed FIRST (for grid_thw), THEN mask (MRoPE)
    - SparseKVCache + recurrent state transfer for GatedDeltaNet
    - forward: sparse KV prefix slicing + 3D position_ids
    """

    # ── Backbone-specific class variable ─────────────────────────────
    mask_helper_cls = MaskHelperQwen35

    def _build_multi_modal_projector(self, cfg):
        return None  # PatchMerger is inside VisionModel, no external projector

    def _vision_ckpt_target(self, cfg) -> tuple:
        """Qwen3.5 vision enables checkpointing per layer via gradient_checkpointing_enable()."""
        return (self.vision_tower, f"vision ({cfg.vision.depth} layers, per-layer)")

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cached_image_grid_thw = None  # image_grid_thw cache required by MRoPE

    # ------------------------------------------------------------------
    # from_pretrained: Qwen3.5 weight loading
    # ------------------------------------------------------------------
    # Embedding resize: Qwen3.5 layout (no padded vocab gap, weight tying)
    # ------------------------------------------------------------------

    def resize_embedding(
        self,
        new_vocab_size: int,
        base_vocab_size: int,
        pad_token_id: int,
        padded_vocab_size: int = None,
        force: bool = False,
        new_token_names: list[str] | None = None,
    ) -> None:
        """Qwen3.5-specific embedding resize.

        Differences from PaliGemma base:
        - Copies ALL old weights (not just [:base_vocab_size]) — no padding gap
        - Ties output_proj.weight to input_proj.weight
        - Updates vlm.config.vocab_size
        """
        vlm = self.vlm
        old_vocab_size = vlm.input_proj.weight.shape[0]

        if old_vocab_size == new_vocab_size and not force:
            return
        if not force and padded_vocab_size and old_vocab_size > padded_vocab_size:
            logger.warning(
                f"[G05ModelQwen35] Vocab already resized ({old_vocab_size} > {padded_vocab_size}), skipping"
            )
            return

        hidden_size = vlm.hidden_size
        is_meta = vlm.input_proj.weight.is_meta
        new_embedding = nn.Embedding(new_vocab_size, hidden_size, padding_idx=pad_token_id)
        if not is_meta:
            with torch.no_grad():
                copy_len = min(old_vocab_size, new_vocab_size)
                new_embedding.weight[:copy_len] = vlm.input_proj.weight[:copy_len]
                if new_vocab_size > old_vocab_size:
                    mean = vlm.input_proj.weight[:copy_len].mean(dim=0)
                    std = vlm.input_proj.weight[:copy_len].std(dim=0)
                    new_embedding.weight[old_vocab_size:] = (
                        torch.randn(new_vocab_size - old_vocab_size, hidden_size) * std + mean
                    )
        # else: meta-device path (FSDP lazy loading) — placeholder weights;
        # real values assigned by load_state_dict(assign=True).
        old_output_weight = vlm.output_proj.weight.data if not is_meta else None
        vlm.input_proj = new_embedding
        vlm.output_proj = nn.Linear(hidden_size, new_vocab_size, bias=False)
        if getattr(vlm.config, "tie_word_embeddings", True):
            vlm.output_proj.weight = vlm.input_proj.weight  # tie (2B/4B)
        else:
            if not is_meta:
                # Untied (9B): copy old lm_head weights, random init for new tokens
                with torch.no_grad():
                    vlm.output_proj.weight[:copy_len] = old_output_weight[:copy_len]
                    if new_vocab_size > copy_len:
                        mean = old_output_weight[:copy_len].mean(dim=0)
                        std = old_output_weight[:copy_len].std(dim=0)
                        vlm.output_proj.weight[copy_len:] = (
                            torch.randn(new_vocab_size - copy_len, hidden_size) * std + mean
                        )
        vlm.config.vocab_size = new_vocab_size

        if new_token_names:
            action_tokens = [t for t in new_token_names if t.startswith("<action")]
            other_tokens = [t for t in new_token_names if not t.startswith("<action")]

            if action_tokens:
                token_summary = (
                    f"{len(action_tokens)} action tokens ({action_tokens[0]}~{action_tokens[-1]})"
                )
                if other_tokens:
                    token_summary += ", " + ", ".join(other_tokens)
            else:
                token_summary = ", ".join(other_tokens)

            logger.info(
                f"[G05ModelQwen35] Resized embedding: {old_vocab_size} → {new_vocab_size} "
                f"(+{len(new_token_names)} tokens: {token_summary})"
            )
        else:
            logger.info(
                f"[G05ModelQwen35] Resized embedding: {old_vocab_size} → {new_vocab_size} "
                f"(+{new_vocab_size - old_vocab_size} tokens)"
            )

    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, cfg) -> "G05ModelQwen35":
        model = cls(cfg)
        pretrained_path = cfg.pretrained_model_path
        tensors = _load_all_safetensors(pretrained_path)

        # Qwen3.5 vision: no HF AutoConfig needed (not in transformers registry)
        model.vision_tower.load_pretrained_weights(None, tensors)
        keys_mapped, keys_missing = model.vlm._load_pretrained_weights(
            pretrained_path, tensors=tensors
        )
        del tensors

        _ckpt_parts = getattr(model, "_grad_ckpt_parts", None)
        _ckpt_label = ", ".join(_ckpt_parts) if _ckpt_parts else "disabled"
        log_box(
            logger,
            "📦  G05Model Qwen3.5 — Pretrained Loading",
            [
                "✅ vision_tower             loaded from pretrained",
                "—  multi_modal_projector    N/A (PatchMerger inside VisionModel)",
                f"✅ vlm                      loaded from pretrained  ({keys_mapped} keys, {keys_missing} missing)",
                "🔀 action_expert            random init",
                None,
                ("grad_checkpointing", _ckpt_label),
            ],
        )
        return model

    # ------------------------------------------------------------------
    # Vision forward: Conv3D + PatchMerger
    # ------------------------------------------------------------------

    def _forward_vision(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Qwen3.5 vision: pixel_values → Conv3D → VisionModel → PatchMerger.

        Supports multi-resolution cameras via Dict input.

        Args:
            pixel_values: Dict[str, Tensor[B, n_k, C, H_k, W_k]]
                Each key = camera name; cameras may have different (H_k, W_k).
                Dict insertion order determines the image token sequence order.

        Sets self._cached_image_grid_thw for MRoPE position_ids.
        The cached grid has shape [n_img, 3] with per-camera [1, grid_h_k, grid_w_k]
        entries matching the token sequence order (same for all batch samples).
        """
        patch_size = self.vision_tower.patch_size
        temporal_patch_size = self.vision_tower.config.temporal_patch_size
        merge_size = self.vision_tower.spatial_merge_size

        first = next(iter(pixel_values.values()))
        bsz = first.shape[0]
        device = first.device

        frame_drop_prob = getattr(self.cfg, "mem_frame_drop_prob", 0.0)
        if self.training and frame_drop_prob > 0:
            drop_flag = torch.rand(1, device=device) < frame_drop_prob
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(drop_flag, src=0)
            drop_history = drop_flag.item()
        else:
            drop_history = False
        self._mem_drop_history = drop_history

        cam_info = []
        for cam_tensor in pixel_values.values():
            n_k = cam_tensor.shape[1]
            if drop_history and n_k > 1:
                cam_tensor = cam_tensor[:, -1:, :, :, :]
                n_k = 1
            C, H_k, W_k = cam_tensor.shape[2:]
            grid_h = H_k // patch_size
            grid_w = W_k // patch_size

            images = einops.rearrange(cam_tensor, "b n c h w -> (b n) c h w")
            total_k = images.shape[0]

            images_temporal = images.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1)
            patches_k = (
                images_temporal.reshape(
                    total_k,
                    temporal_patch_size,
                    C,
                    grid_h // merge_size,
                    merge_size,
                    patch_size,
                    grid_w // merge_size,
                    merge_size,
                    patch_size,
                )
                .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
                .reshape(total_k * grid_h * grid_w, -1)
            )
            grid_thw_k = torch.tensor(
                [[1, grid_h, grid_w]] * total_k,
                dtype=torch.long,
                device=device,
            )
            cam_info.append((patches_k, grid_thw_k, n_k, grid_h, grid_w))

        has_mem = any(info[2] > 1 for info in cam_info) and self.vision_tower.temporal_freq > 0
        batch_all = getattr(self.cfg.vision, "batch_all_cameras", False)

        if not has_mem or batch_all:
            # All cameras in one ViT call.
            # batch_all + MEM: uses varlen T+S (mixed patch counts handled by cu_seqlens)
            all_patches = torch.cat([c[0] for c in cam_info], dim=0)
            all_grid_thw = torch.cat([c[1] for c in cam_info], dim=0)
            n_k = cam_info[0][2]
            total_bsz = bsz * len(cam_info) if batch_all and has_mem else bsz

            _, pooler_output = self.vision_tower(
                all_patches, all_grid_thw, num_frames=n_k, bsz=total_bsz
            )

            all_features = []
            seq_grid_thw = []
            offset = 0
            # n_out: number of output images per camera per batch sample.
            # Same for all cameras (token drop keeps 1 frame if MEM, else n_k frames).
            # Derive from pooler total / (sum of all cameras' tok_per_img * bsz * n_out_per_cam)
            # Simplest: compute total expected tokens per batch and derive n_out.
            total_tok_per_img = sum(
                (gh // merge_size) * (gw // merge_size) for _, _, _, gh, gw in cam_info
            )
            n_out = pooler_output.shape[0] // (bsz * total_tok_per_img)
            for cam_idx, (_, _, n_k_cam, grid_h, grid_w) in enumerate(cam_info):
                tok_per_img = (grid_h // merge_size) * (grid_w // merge_size)
                cam_tok = bsz * n_out * tok_per_img
                feat = pooler_output[offset : offset + cam_tok]
                offset += cam_tok
                feat = feat.reshape(bsz, n_out * tok_per_img, -1)
                all_features.append(feat)
                seq_grid_thw.extend([[1, grid_h, grid_w]] * n_out)
        else:
            from collections import defaultdict

            groups = defaultdict(list)
            for idx, (patches_k, grid_thw_k, n_k, grid_h, grid_w) in enumerate(cam_info):
                key = (grid_h, grid_w, n_k)
                groups[key].append(idx)

            group_results = [None] * len(cam_info)
            for key, indices in groups.items():
                group_patches = torch.cat([cam_info[i][0] for i in indices], dim=0)
                group_grid_thw = torch.cat([cam_info[i][1] for i in indices], dim=0)
                n_k = key[2]

                group_bsz = bsz * len(indices)
                _, pooler_output = self.vision_tower(
                    group_patches,
                    group_grid_thw,
                    num_frames=n_k,
                    bsz=group_bsz,
                )

                tok_per_img = (key[0] // merge_size) * (key[1] // merge_size)
                n_out = pooler_output.shape[0] // (group_bsz * tok_per_img)
                cam_tok = bsz * n_out * tok_per_img

                offset = 0
                for i in indices:
                    feat = pooler_output[offset : offset + cam_tok]
                    offset += cam_tok
                    feat = feat.reshape(bsz, n_out * tok_per_img, -1)
                    group_results[i] = (feat, [[1, key[0], key[1]]] * n_out)

            all_features = []
            seq_grid_thw = []
            for res in group_results:
                feat, grid = res
                all_features.append(feat)
                seq_grid_thw.extend(grid)

        self._cached_image_grid_thw = torch.tensor(
            seq_grid_thw,
            dtype=torch.long,
            device=device,
        )

        image_features = torch.cat(all_features, dim=1)
        return image_features

    # ------------------------------------------------------------------
    # Mask + Position IDs: pass cached grid_thw to MRoPE helper
    # ------------------------------------------------------------------

    def build_causal_mask_and_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        kv_len: int = 0,
        dtype: torch.dtype = torch.float32,
        is_action_block: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        self.mask_helper._is_training = self.training
        self.mask_helper._image_grid_thw = self._cached_image_grid_thw
        self.mask_helper._spatial_merge_size = self.vision_tower.spatial_merge_size
        return self.mask_helper.build_vlm_mask(
            input_ids,
            attention_mask,
            kv_len=kv_len,
            dtype=dtype,
            is_action_block=is_action_block,
        )

    # ------------------------------------------------------------------
    # MEM: mask history state tokens on frame drop
    # ------------------------------------------------------------------

    def _mask_history_state_tokens(self, attention_mask: torch.Tensor) -> None:
        """Mark history state tokens as PADDING during frame drop.

        MEM paper: "if the history frame is dropped out, the corresponding
        state token is masked out as well."

        PADDING positions are skipped by mask_helper: they receive no position_ids
        and do not participate in attention. Only the final state token is kept,
        representing the current frame t=0.
        """
        PROPRIO = TOKEN_INDEX.PROPRIO_TOKEN_INDEX
        PADDING = TOKEN_INDEX.PADDING_TOKEN_INDEX

        for b in range(attention_mask.shape[0]):
            proprio_positions = (attention_mask[b] == PROPRIO).nonzero(as_tuple=True)[0]
            if len(proprio_positions) <= 1:
                continue
            attention_mask[b, proprio_positions[:-1]] = PADDING

    # ------------------------------------------------------------------
    # VLM Prefill: reversed call order (embed → mask) for MRoPE
    # ------------------------------------------------------------------

    def vlm_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: Dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        split_index: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        """Qwen3.5 prefill: embed FIRST (sets grid_thw), THEN mask (uses MRoPE)."""
        # Step 1: embed (sets _cached_image_grid_thw)
        # Vision per-layer checkpointing is handled inside vision_tower.forward().
        inputs_embeds = self._forward_embed(input_ids, attention_mask, pixel_values, proprio)

        # MEM frame drop: mask history state tokens as PADDING so VLM can't see them.
        # Must happen AFTER _forward_embed (scatter is done) but BEFORE build_causal_mask
        # (so mask_helper skips PADDING positions in position_ids and attention).
        if getattr(self, "_mem_drop_history", False):
            self._mask_history_state_tokens(attention_mask)

        # Step 2: mask (uses _cached_image_grid_thw for MRoPE)
        causal_mask, position_ids = self.build_causal_mask_and_position_ids(
            input_ids,
            attention_mask,
            dtype=dtype,
        )

        # Step 3: build unified SparseKVCache (full_attn KV + linear_attn states)
        vlm_cache = self._build_vlm_sparse_cache()

        # Build padding_mask [B, S] for GatedDeltaNet (1=valid, 0=padding)
        padding_mask = (attention_mask != 0).to(inputs_embeds.dtype)

        vlm_hidden, vlm_cache = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            kv_cache=vlm_cache,
            return_kv_cache=True,
            attn_implementation=self.attn_implementation,
            mixture_name="vlm",
            split_idx=split_index,
            padding_mask=padding_mask,
        )
        return vlm_hidden, vlm_cache, position_ids

    # ------------------------------------------------------------------
    # Training forward: sparse KV + 3D position_ids
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: Dict[str, torch.Tensor],
        actions: torch.FloatTensor,
        action_pad_masks: torch.BoolTensor,
        action_dim_is_pad: Optional[torch.BoolTensor],
        split_index: int,
        labels: torch.LongTensor,
        continuous_action: bool = False,
        skip_ce_loss: bool = False,
        proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        dtype = next(iter(pixel_values.values())).dtype

        vlm_hidden, vlm_kv, position_ids = self.vlm_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            dtype=dtype,
            split_index=split_index if continuous_action else None,
            proprio=proprio,
        )

        if not skip_ce_loss:
            ce_loss, overall_accuracy = self.ar_helper.train_step(self, vlm_hidden, labels)
            # Split accuracy from _last_ce_cache (populated by cal_ce_loss)
            cache = self.ar_helper._last_ce_cache
            action_accuracy = cache["action_accuracy"] if cache else 0.0
            cot_accuracy = cache["cot_accuracy"] if cache else 0.0
        else:
            ce_loss, overall_accuracy, action_accuracy, cot_accuracy = None, 0.0, 0.0, 0.0

        # Build prefix-only action cache (KV trimmed to split_index + recurrent states at boundary).
        vlm_kv_prefix = self._build_prefix_action_kv(vlm_kv, split_index)

        # 3D position_ids [3, B, S] → slice to prefix
        pos_prefix = position_ids[..., :split_index]

        fm_loss = self.fm_helper.train_step(
            self,
            vlm_kv_prefix,
            attention_mask[:, :split_index],
            pos_prefix,
            actions,
            action_pad_masks,
            action_dim_is_pad,
            dtype,
            embodiment_types=kwargs.get("embodiment_types"),
        )

        if not continuous_action:
            fm_loss = fm_loss * 0

        if skip_ce_loss:
            return {"fm_loss": fm_loss}
        return {
            "fm_loss": fm_loss,
            "ce_loss": ce_loss,
            "overall_accuracy": overall_accuracy,
            "action_accuracy": action_accuracy,
            "cot_accuracy": cot_accuracy,
        }

    # ------------------------------------------------------------------
    # Action mask: adjust prefix length for ae_vlm_condition_mode
    # ------------------------------------------------------------------

    def build_action_mask_and_position_ids(
        self,
        attention_mask_prefix: torch.Tensor,
        action_len: int,
        position_ids_prefix=None,
        split_index=None,
        dtype: torch.dtype = torch.float32,
        action_causal: bool = False,
    ):
        """Override: when recurrent_only, pass empty prefix mask so action mask
        shape is [B, 1, H, H] instead of [B, 1, H, S_prefix+H].

        position_ids offset still uses the original split_index / position_ids_prefix,
        so action token positions are correctly placed after the VLM prefix.
        """
        mode = getattr(self.cfg, "ae_vlm_condition_mode", "both")
        if mode == "recurrent_only":
            # AE full-attn layers have no VLM prefix KV → prefix part of mask is empty
            eff_prefix_mask = attention_mask_prefix[:, :0]  # [B, 0]
        else:
            eff_prefix_mask = attention_mask_prefix
        return self.mask_helper.build_action_mask(
            eff_prefix_mask,
            action_len,
            position_ids_prefix=position_ids_prefix,
            split_index=split_index,
            dtype=dtype,
            action_causal=action_causal,
        )

    # ------------------------------------------------------------------
    # SparseKVCache helpers
    # ------------------------------------------------------------------

    def _build_vlm_sparse_cache(self) -> SparseKVCache:
        layer_types = self.vlm.layer_types
        last_linear = len(layer_types) - 1 - layer_types[::-1].index("linear_attention")
        return SparseKVCache(last_linear_layer=last_linear)

    def _build_prefix_action_kv(self, vlm_kv: SparseKVCache, split_index: int) -> SparseKVCache:
        """Build prefix-only SparseKVCache for action_expert.

        Controlled by cfg.ae_vlm_condition_mode:
          "recurrent_only"  — GDN init from VLM recurrent states, full_attn = action self-attn only
          "cross_attn_only" — full_attn cross-attends VLM prefix KV, GDN starts from scratch
          "both"            — both prefix KV and recurrent states (full conditioning)

        Training: split_index = prefix len, may be < vlm_kv seq len due to AR suffix tokens.
        Inference: split_index = vlm_kv.num_items() (no trimming needed).
        """
        mode = getattr(self.cfg, "ae_vlm_condition_mode", "both")
        prefix_kv = SparseKVCache(last_linear_layer=vlm_kv.last_linear_layer)

        if mode != "recurrent_only":
            for layer_idx, k in vlm_kv.key_cache.items():
                prefix_kv.key_cache[layer_idx] = k[..., :split_index, :]
                prefix_kv.value_cache[layer_idx] = vlm_kv.value_cache[layer_idx][
                    ..., :split_index, :
                ]

        if mode != "cross_attn_only":
            src = (
                vlm_kv.split_recurrent_states
                if vlm_kv.split_recurrent_states
                else vlm_kv.recurrent_states
            )
            for layer_idx, rs in src.items():
                prefix_kv.recurrent_states[layer_idx] = rs

        return prefix_kv
