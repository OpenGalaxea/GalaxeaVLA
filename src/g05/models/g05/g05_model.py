# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
G05Model: assembles Vision + 2x Mixture.
Vision (SiGLIP + projector) lives at the G05Model layer, while Mixtures remain independent.
Mask/position construction is delegated to MaskHelper (see mask_helper.py).
Based on PiAR + GalaxeaJoint refactoring.
"""

import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from g05.utils.common.import_utils import get_obj_from_str
from g05.utils.hf import resolve_hf_model_path
from g05.utils.logging.log_box import log_box
from .helpers.ar_helper import ARHelper
from .helpers.fm_helper import FMHelper
from .helpers.mask_helper import MaskHelper, make_attn_mask  # noqa: F401 (make_attn_mask re-exported)
from .model.utils import _load_all_safetensors
from .helpers.proprio_helper import ProprioEmbedder
from .io.input_preprocessor import TOKEN_INDEX, normalize_proprio_encoder_mode

logger = logging.getLogger(__name__)


class G05Model(nn.Module):
    """Vision + VLM Mixture + Action Expert Mixture.

    Does not own embed_tokens / lm_head; those live inside the VLM Mixture.
    Mask/position construction happens at this layer.

    Attributes:
        vision_tower: SiglipVisionModel
        multi_modal_projector: PaliGemmaMultiModalProjector
        vlm: Mixture (VLM, with Embedding input + lm_head output)
        action_expert: Mixture (AE, with Linear input + Linear output + adaLN)
    """

    mask_helper_cls: type = MaskHelper

    def __init__(self, cfg):
        """Build the full model from config, including all submodules.

        cfg must include these sections:
            - vlm: complete VLM Mixture config
            - action_expert: complete Action Expert Mixture config
            - fm: Flow Matching algorithm config
            - ar: Autoregressive algorithm config
            - vision: SiglipVisionModel config
            - vision_projector: PaliGemmaMultiModalProjector config
        and top-level fields:
            - image_token_index, pad_token_id
            - position_ids_type, action_position_offset
            - attn_implementation, checkpoint_vision, inference_offset
        """
        super().__init__()
        self.cfg = cfg
        # --- Model-level config ---
        self.image_token_index = cfg.image_token_index
        self.pad_token_id = cfg.pad_token_id
        self.attn_implementation = cfg.attn_implementation
        self.checkpoint_vision = cfg.checkpoint_vision
        self.proprio_encoder = normalize_proprio_encoder_mode(cfg.proprio_encoder)
        # --- Helpers (weight-free, initialized from independent config sections) ---
        self.mask_helper = self.mask_helper_cls(cfg)
        self.fm_helper = FMHelper(cfg.fm)
        self.ar_helper = ARHelper(cfg.ar)
        # --- Submodules (created in __init__; from_pretrained later overwrites weights) ---
        # Dynamic instantiation: config.name gives the class path, aligned with
        # GalaxeaZero vision/projector.
        self.vision_tower = get_obj_from_str(cfg.vision.name)(cfg.vision)
        self.multi_modal_projector = self._build_multi_modal_projector(cfg)
        self.vlm = get_obj_from_str(cfg.vlm.name)(cfg.vlm)
        self.action_expert = get_obj_from_str(cfg.action_expert.name)(cfg.action_expert)
        self.proprio_embedder: Optional[ProprioEmbedder] = None
        if self.proprio_encoder in {"mlp", "mlp_dropout", "zeros"}:
            self.proprio_embedder = ProprioEmbedder(
                proprio_dim=int(cfg.proprio_dim),
                hidden_size=int(cfg.vlm.hidden_size),
            )
            logger.info(
                f"[G05Model] proprio_embedder enabled: "
                f"proprio_dim={int(cfg.proprio_dim)} → d_vlm={int(cfg.vlm.hidden_size)}"
            )
        # --- Gradient Checkpointing ---
        # Vision is special: forward wraps it with checkpoint.checkpoint() instead of
        # calling enable(). VLM / action_expert use HF gradient_checkpointing_enable().
        vis_module, vis_label = self._vision_ckpt_target(cfg)
        ckpt_targets = [
            (cfg.checkpoint_vision, vis_module, vis_label),
            (cfg.checkpoint_vlm, self.vlm, f"vlm ({cfg.vlm.num_hidden_layers} layers)"),
            (
                cfg.checkpoint_action_expert,
                self.action_expert,
                f"action_expert ({cfg.action_expert.num_hidden_layers} layers)",
            ),
        ]
        ckpt_parts = []
        for enabled, module, label in ckpt_targets:
            if enabled:
                if module is not None:
                    module.gradient_checkpointing_enable()
                ckpt_parts.append(label)
        self._grad_ckpt_parts = ckpt_parts  # read by the from_pretrained log box

    # ------------------------------------------------------------------
    # from_pretrained: first-training entry point
    # ------------------------------------------------------------------

    def _build_multi_modal_projector(self, cfg):
        """Build multi_modal_projector; subclasses may override and return None."""
        return get_obj_from_str(cfg.vision_projector.name)(cfg.vision_projector)

    def _vision_ckpt_target(self, cfg) -> tuple:
        """Return (module_or_None, label) for gradient checkpointing setup.

        Base PaliGemma vision is manually wrapped with checkpoint.checkpoint() and
        does not use enable().
        """
        return (None, "vision (27 layers)")

    @classmethod
    def from_pretrained(cls, cfg) -> "G05Model":
        """Load HF pretrained weights into an already constructed model.

        __init__ creates the full structure with random weights; this method
        overwrites VLM + Vision weights. Action Expert remains randomly initialized.

        Args:
            cfg: top-level model_arch config containing pretrained_model_path
        """
        from transformers import AutoConfig

        model = cls(cfg)
        pretrained_path = resolve_hf_model_path(cfg.pretrained_model_path)
        hf_config = AutoConfig.from_pretrained(pretrained_path)
        # Load all safetensors once, share across components
        tensors = _load_all_safetensors(pretrained_path)
        # Load pretrained weights into existing modules
        model.vision_tower.load_pretrained_weights(hf_config, tensors)
        model.multi_modal_projector.load_pretrained_weights(hf_config, tensors)
        keys_mapped, keys_missing = model.vlm._load_pretrained_weights(
            pretrained_path, tensors=tensors
        )
        del tensors
        # ── Box 2: Pretrained Loading Summary ────────────────────────
        _ckpt_parts = getattr(model, "_grad_ckpt_parts", None)
        _ckpt_label = ", ".join(_ckpt_parts) if _ckpt_parts else "disabled"
        log_box(
            logger,
            "📦  G05 Model — Pretrained Loading",
            [
                "✅ vision_tower             loaded from pretrained",
                "✅ multi_modal_projector    loaded from pretrained",
                f"✅ vlm                      loaded from pretrained  ({keys_mapped} keys, {keys_missing} missing)",
                "🔀 action_expert            random init",
                None,
                ("grad_checkpointing", _ckpt_label),
            ],
        )
        return model

    # ------------------------------------------------------------------
    # Embedding resize
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
        """Resize VLM embedding to accommodate extra action tokens.
        Args:
            new_vocab_size: Target vocab size (base + new tokens).
            base_vocab_size: Original pretrained vocab size (used to slice
                the existing embedding before expansion).
            pad_token_id: Padding token id for the new embedding.
            padded_vocab_size: If set, used for sanity check — raise if
                current size already exceeds it (unless *force*).
            force: Skip early-return / safety checks when True.
        """
        from g05.tokenizer.utils.misc import resize_embeddings_with_distribution_init

        vlm = self.vlm
        old_vocab_size = vlm.input_proj.weight.shape[0]
        if old_vocab_size == new_vocab_size and not force:
            return
        if not force and padded_vocab_size and old_vocab_size > padded_vocab_size:
            raise ValueError(
                f"Current vocab size ({old_vocab_size}) already exceeds "
                f"padded_vocab_size ({padded_vocab_size}). "
                f"Pass force=True to override."
            )
        hidden_size = vlm.hidden_size
        new_embedding = nn.Embedding(new_vocab_size, hidden_size, padding_idx=pad_token_id)
        if not vlm.input_proj.weight.is_meta:
            num_truly_new = new_vocab_size - old_vocab_size
            with torch.no_grad():
                # Preserve ALL existing rows exactly (base vocab + any already-initialised
                # action/control token embeddings). Only the truly new rows are sampled
                # from the pretrained base distribution.
                new_embedding.weight[:old_vocab_size].copy_(vlm.input_proj.weight)
                if num_truly_new > 0:
                    new_rows = resize_embeddings_with_distribution_init(
                        embedding=vlm.input_proj.weight[:base_vocab_size],
                        num_new_tokens=num_truly_new,
                        padding_idx=pad_token_id,
                    )[base_vocab_size:]  # shape: [num_truly_new, hidden_size]
                    new_embedding.weight[old_vocab_size:].copy_(new_rows)
        # else: meta-device path (FSDP lazy loading) — weight tensor is a placeholder;
        # real values will be assigned by load_state_dict(assign=True).
        vlm.input_proj = new_embedding
        vlm.output_proj = nn.Linear(hidden_size, new_vocab_size, bias=False)
        vlm.output_proj.weight = vlm.input_proj.weight
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
                f"Resized VLM embedding: {old_vocab_size} → {new_vocab_size} "
                f"(+{len(new_token_names)} tokens: {token_summary})"
            )
        else:
            num_new_tokens = new_vocab_size - base_vocab_size
            logger.info(
                f"Resized VLM embedding: {old_vocab_size} → {new_vocab_size} "
                f"(+{num_new_tokens} tokens)"
            )

    # ------------------------------------------------------------------
    # Vision forward
    # ------------------------------------------------------------------
    def _forward_vision(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SiGLIP (video) -> projector -> [B, n_cam*P, d_vlm].

        K=1 (single-frame inference): concatenate all cameras into [B*n_cam, C, H, W]
            and run vision_tower once. This exactly matches the g05-0414 checkpoint
            training path and preserves bfloat16 numeric alignment. CUDA tile
            partitioning for bfloat16 matmul depends on batch size; calling
            [1,C,H,W] n_cam times vs one [n_cam,C,H,W] call creates different
            rounding errors. Transformer cascading can amplify this and fork AR
            decode token sequences.

        K>1 (video memory): run each camera [B, K, C, H, W] through vision_tower
            separately, concatenate outputs, then feed projector.

        Args:
            pixel_values: Dict[camera_key, Tensor[B, C, H, W]] for K=1 after squeeze,
                       or Dict[camera_key, Tensor[B, K, C, H, W]] for K>1 video memory
        Returns:
            image_features: [B, n_cam*P, d_vlm]
        """
        pv_list = list(pixel_values.values())
        first = pv_list[0]
        n_cam = len(pv_list)

        if first.dim() == 4:
            # K=1: concatenate into [B*n_cam, C, H, W] and run one vision_tower forward.
            B = first.shape[0]
            pv_batch = torch.cat(pv_list, dim=0)  # [B*n_cam, C, H, W]
            feats = self.vision_tower(pv_batch)  # [B*n_cam, P, d]
            P, d = feats.shape[1], feats.shape[2]
            # [B*n_cam, P, d] -> [B, n_cam*P, d], preserving pv_list camera order.
            feats = feats.view(n_cam, B, P, d).permute(1, 0, 2, 3).reshape(B, n_cam * P, d)
            return self.multi_modal_projector(feats)
        else:
            # K>1: each camera [B, K, C, H, W] -> flatten -> vision_tower -> unflatten.
            features = []
            for v in pv_list:
                B, K = v.shape[:2]
                feat = self.vision_tower(v.flatten(0, 1))  # [B*K, P, d]
                feat = feat.reshape(B, K * feat.shape[1], -1)  # [B, K*P, d]
                features.append(feat)
            image_features = torch.cat(features, dim=1)  # [B, n_cam*K*P, d]
            return self.multi_modal_projector(image_features)

    # ------------------------------------------------------------------
    # VLM Embedding (text + vision merge)
    # ------------------------------------------------------------------
    def _forward_embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """VLM embedding: text embed + vision merge (+ optional state scatter) -> [B, S, d_vlm].

        Args:
            input_ids: [B, S]
            pixel_values: Dict[str, Tensor[B, n_k, C, H_k, W_k]]
            proprio: optional [B, T, proprio_dim], used only when
                proprio_encoder=="mlp"/"mlp_dropout"/"zeros". Caller is responsible
                for zero-masking padded dims. T is usually num_obs_steps = 1.
            attention_mask: [B, S] TOKEN_INDEX enum values used to identify token types.
        Returns:
            final_embedding: [B, S, d_vlm]
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        # Image features
        image_features = self._forward_vision(pixel_values)
        _, _, embed_dim = image_features.shape
        dtype = image_features.dtype  # use vision output dtype (may be bf16 under autocast)

        # Text embeddings (with √d scaling via vlm.embed)
        inputs_embeds = self.vlm.embed(input_ids).to(dtype)

        # Merge: put image features where input_ids == image_token_index
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim),
            self.pad_token_id,
            dtype=dtype,
            device=device,
        )
        # Base text mask: exclude image, and exclude state when enabled so text
        # embeddings do not overwrite MLP output.
        # In mlp mode, G05Policy.__init__ must inject state_token_id; otherwise state
        # scatter would be silently skipped, making state_mlp ineffective without a
        # loss error.
        if self.proprio_embedder is not None:
            assert self.proprio_embedder.state_token_id is not None, (
                "proprio_embedder.state_token_id was not synchronized. "
                "Verify that G05Policy.__init__ backfilled state_token_id from the processor."
            )
            state_token_id = self.proprio_embedder.state_token_id
        else:
            state_token_id = None
        image_mask = attention_mask == TOKEN_INDEX.IMAGE_TOKEN_INDEX
        text_mask = (attention_mask != TOKEN_INDEX.PADDING_TOKEN_INDEX) & ~image_mask
        if state_token_id is not None:
            text_mask = text_mask & (attention_mask != TOKEN_INDEX.PROPRIO_TOKEN_INDEX)

        final_embedding[text_mask] = inputs_embeds[text_mask]

        # Vectorized image feature assignment
        image_feature_indices = image_mask.long().cumsum(dim=1) - 1
        image_feature_indices = image_feature_indices.clamp(min=0)

        # Guard: detect image token count vs vision output mismatch before CUDA assert
        n_vis = image_features.shape[1]
        max_img_idx = image_feature_indices[image_mask].max().item() if image_mask.any() else 0
        if max_img_idx >= n_vis:
            import os

            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
            n_img_per_sample = image_mask.sum(dim=1).tolist()
            pv_info = (
                {k: list(v.shape) for k, v in pixel_values.items()}
                if isinstance(pixel_values, dict)
                else f"Tensor{list(pixel_values.shape)}"
            )
            raise RuntimeError(
                f"\n{'=' * 70}\n"
                f"[RANK {rank}] IMAGE TOKEN MISMATCH in _forward_embed\n"
                f"  max_gather_idx={max_img_idx} >= image_features.shape[1]={n_vis}\n"
                f"  batch_size={bsz}, seq_len={seq_len}\n"
                f"  per_sample_img_tokens={n_img_per_sample}\n"
                f"  pixel_values={pv_info}\n"
                f"{'=' * 70}"
            )

        gathered_features = torch.gather(
            image_features,
            1,
            image_feature_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
        )
        final_embedding[image_mask] = gathered_features[image_mask]

        # State scatter (proprio -> proprio_embedder -> replace <state> token positions).
        # Runs when proprio_embedder.state_token_id exists (mlp mode) and proprio is
        # provided. Mirrors the vectorized cumsum-gather implementation used by image
        # scatter. Convention: policy has already ensured proprio is float32 and 3D
        # [B, T, D].
        if state_token_id is not None and proprio is not None:
            # ProprioEmbedder.forward disables autocast and computes in fp32; cast
            # output back to final_embedding dtype.
            state_features = self.proprio_embedder(proprio).to(dtype)  # [B, T, d_vlm]
            state_mask = attention_mask == TOKEN_INDEX.PROPRIO_TOKEN_INDEX
            state_feature_indices = state_mask.long().cumsum(dim=1) - 1
            state_feature_indices = state_feature_indices.clamp(min=0)
            gathered_state = torch.gather(
                state_features,
                1,
                state_feature_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
            )
            final_embedding[state_mask] = gathered_state[state_mask]

        return final_embedding

    # ------------------------------------------------------------------
    # Mask + Position IDs construction
    # ------------------------------------------------------------------
    def build_causal_mask_and_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        kv_len: int = 0,
        dtype: torch.dtype = torch.float32,
        is_action_block: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """VLM mask + position_ids delegated to mask_helper.

        is_action_block only affects the Qwen35 (MRoPE) subclass. The PaliGemma path
        does not use it; it is passed through and accepted by the base signature.
        """
        self.mask_helper._is_training = self.training
        return self.mask_helper.build_vlm_mask(
            input_ids,
            attention_mask,
            kv_len=kv_len,
            dtype=dtype,
        )

    def build_action_mask_and_position_ids(
        self,
        attention_mask_prefix: torch.Tensor,
        action_len: int,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        split_index: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        action_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Action Expert mask + position_ids delegated to mask_helper."""
        return self.mask_helper.build_action_mask(
            attention_mask_prefix,
            action_len,
            position_ids_prefix=position_ids_prefix,
            split_index=split_index,
            dtype=dtype,
            action_causal=action_causal,
        )

    # ------------------------------------------------------------------
    # VLM Prefill (shared by training + FM/AR inference)
    # ------------------------------------------------------------------
    def vlm_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: Dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        """VLM prefill: embed -> mask -> forward -> (hidden, kv, position_ids).

        Shared by training and inference. During training, checkpoint_vision=True
        applies activation checkpointing to vision.

        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            pixel_values: [B, n_img, C, H, W]
            dtype: mask dtype, usually pixel_values.dtype
            proprio: optional [B, T, proprio_dim], used only when
                proprio_encoder=="mlp"/"mlp_dropout"/"zeros"
        Returns:
            vlm_hidden: [B, S, d_vlm]
            vlm_kv: List[(K, V)] per layer
            position_ids: [B, S] (0-indexed)
        """
        causal_mask, position_ids = self.build_causal_mask_and_position_ids(
            input_ids,
            attention_mask,
            dtype=dtype,
        )
        if self.checkpoint_vision and self.training:
            pv_for_ckpt = {k: v.detach().requires_grad_(True) for k, v in pixel_values.items()}
            checkpoint_kwargs = {"use_reentrant": False, "preserve_rng_state": False}
            inputs_embeds = checkpoint.checkpoint(
                self._forward_embed,
                input_ids,
                attention_mask,
                pv_for_ckpt,
                proprio,
                **checkpoint_kwargs,
            )
        else:
            inputs_embeds = self._forward_embed(input_ids, attention_mask, pixel_values, proprio)

        vlm_hidden, vlm_kv = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            return_kv_cache=True,
            attn_implementation=self.attn_implementation,
            mixture_name="vlm",
        )
        return vlm_hidden, vlm_kv, position_ids

    # ------------------------------------------------------------------
    # Forward (training)
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
        """Training entry: VLM forward -> AR loss -> FM loss.

        Time sampling is done internally by fm_helper; callers do not need to handle it.

        Returns:
            loss_dict with keys: fm_loss, ce_loss (optional), action_accuracy (optional)
        """
        dtype = next(iter(pixel_values.values())).dtype
        device = input_ids.device
        batch_size = actions.size(0)
        # VLM prefill (shared method)
        vlm_hidden, vlm_kv, position_ids = self.vlm_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            dtype=dtype,
            proprio=proprio,
        )
        # 4. AR loss (CE) delegated to ar_helper.
        if not skip_ce_loss:
            ce_loss, overall_accuracy = self.ar_helper.train_step(
                self,
                vlm_hidden,
                labels,
            )
            # Split accuracy from _last_ce_cache (populated by cal_ce_loss)
            cache = self.ar_helper._last_ce_cache
            action_accuracy = cache["action_accuracy"] if cache else 0.0
            cot_accuracy = cache["cot_accuracy"] if cache else 0.0
        else:
            ce_loss, overall_accuracy, action_accuracy, cot_accuracy = None, 0.0, 0.0, 0.0
        # 5. FM loss delegated to fm_helper; time sampling happens inside the helper.
        # Pre-slice the prefix so FM helper does not need to know split_index.
        vlm_kv_prefix = [(k[:, :, :split_index], v[:, :, :split_index]) for k, v in vlm_kv]
        fm_loss = self.fm_helper.train_step(
            self,
            vlm_kv_prefix,
            attention_mask[:, :split_index],
            position_ids[:, :split_index],
            actions,
            action_pad_masks,
            action_dim_is_pad,
            dtype,
            embodiment_types=kwargs.get("embodiment_types"),
        )
        # Mask FM loss for VLM-only batches
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
    # Inference facades delegated to helpers
    # ------------------------------------------------------------------
    def inference_fm(
        self,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        past_key_values: list,
        **kwargs,
    ) -> torch.Tensor:
        """FM inference facade delegated to fm_helper.infer. Caller owns prefill."""
        return self.fm_helper.infer(
            self,
            attention_mask,
            pixel_values,
            past_key_values,
            **kwargs,
        )

    def inference_ar(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        past_key_values=None,
        **kwargs,
    ) -> dict:
        """AR inference facade delegated to ar_helper.infer.

        Caller must provide past_key_values from vlm_prefill and last_hidden from
        prefill or the previous AR last hidden. The first token is sampled directly
        from last_hidden; the prefix last token is not forwarded again, avoiding the
        duplicated KV slot bug.
        """
        return self.ar_helper.infer(
            self,
            last_hidden,
            attention_mask,
            pixel_values,
            past_key_values=past_key_values,
            **kwargs,
        )
