"""
G05PolicyQwen35 — Qwen3.5 variant of G05Policy.

Inherits G05Policy, overrides only backbone-specific class variables:
  - model_cls = G05ModelQwen35 (hybrid attention + MRoPE)
  - Qwen3.5 vocab layout (no padding gap)
  - Qwen3.5 processor / model_type
  - _post_processor_init: action token offset fix instead of proprio_embedder sync
  - Additional fp32_param_patterns for QK norms and GatedDeltaNet
"""

from __future__ import annotations

from typing import List

from g05.utils.logging.logging_config import get_logger

from .g05_policy import G05Policy
from .g05_model_qwen35 import G05ModelQwen35

logger = get_logger(__name__)


class G05PolicyQwen35(G05Policy):
    """Qwen3.5 variant of G05Policy.

    Only overrides five class variables and the _post_processor_init hook.
    __init__ fully reuses G05Policy, so new functionality is inherited automatically.
    """

    # ── Backbone-specific class variables ────────────────────────────
    model_cls = G05ModelQwen35
    default_base_vocab_size: int = 248044
    default_padded_vocab_size: int = 248320
    default_hf_processor_class: str = "g05.models.g05.qwen35.processing.Qwen35ProcessorWrapper"
    default_model_type: str = "qwen35"

    @property
    def fp32_param_patterns(self) -> List[str]:
        return [
            # vision embeddings (Qwen3.5 Conv3D + position embedding)
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.pos_embed.weight",
            # vision PatchMerger full projector (norm + linear) in fp32
            "vision_tower.merger.norm",
            "vision_tower.merger.linear_fc1",
            "vision_tower.merger.linear_fc2",
            "norm1",
            "norm2",
            # per-layer norms
            "input_layernorm",
            "post_attention_layernorm",
            # QK norms (Qwen3.5 specific)
            "q_norm",
            "k_norm",
            # GatedDeltaNet norms + decay parameters (used in fp32 gate computation)
            "linear_attn.norm",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            # final norms
            "vlm.norm",
            "action_expert.norm",
            # action expert I/O projections + time conditioning
            "action_expert.input_proj",
            "action_expert.output_proj",
            "action_expert.time_embedding",
            "action_expert.time_mlp_in",
            "action_expert.time_mlp_out",
            # proprio encoder has autocast(enabled=False) internally, so weights must be fp32
            "proprio_embedder",
        ]

    def _post_processor_init(self) -> None:
        """Fix action token offset first, then let the base class sync proprio_embedder.state_token_id."""
        self._maybe_extend_loc_token_range()
        self._fix_action_token_offset()
        super()._post_processor_init()

    def _maybe_extend_loc_token_range(self):
        """When add_loc_tokens=True, extend ar_helper.loc_token_end past <loc1023>.

        Qwen3.5 vocab layout after enabling the flag:
          [0, 248044)             — Qwen BPE
          [248044, 248077)        — Qwen built-in added special tokens (33)
          [248077, 249101)        — newly added 1024 <loc0000>~<loc1023> tokens
          [249101, ...)           — action tokens set by _fix_action_token_offset

        Extending loc_token_end to 249101 makes <loc> tokens fall into the
        COT_TOKEN_INDEX bucket.
        """
        if not bool(self.config.get("add_loc_tokens", False)):
            return
        tokenizer = self.processor.tokenizer
        last_loc_id = tokenizer.convert_tokens_to_ids("<loc1023>")
        if last_loc_id is None:
            logger.warning("add_loc_tokens=True but <loc1023> not in tokenizer; skipping")
            return
        ar = self.model.ar_helper
        new_loc_end = last_loc_id + 1  # exclusive upper bound
        if ar.loc_token_end < new_loc_end:
            old = ar.loc_token_end
            ar.loc_token_end = new_loc_end
            logger.warning(
                f"add_loc_tokens=True: extended ar_helper.loc_token_end {old} -> {new_loc_end}"
            )

    def _fix_action_token_offset(self):
        """Fix action_token_begin_idx for Qwen3.5 special token layout."""
        if not self.action_tokenizer.use_extra_tokens:
            return
        if not hasattr(self.action_tokenizer, "action_tokens"):
            return

        first_action_token = self.action_tokenizer.action_tokens[0]
        correct_begin = self.processor.tokenizer.convert_tokens_to_ids(first_action_token)
        if correct_begin is None:
            return

        wrong_begin = self.action_tokenizer.action_token_begin_idx
        if correct_begin == wrong_begin:
            return

        offset = correct_begin - wrong_begin
        self.action_tokenizer.action_token_begin_idx = correct_begin
        self.action_tokenizer.action_token_end_idx += offset

        self.model.ar_helper.set_token_index_ranges(
            self.action_tokenizer.action_token_begin_idx,
            self.action_tokenizer.action_token_end_idx,
        )
        logger.warning(
            f"Fixed Qwen3.5 action_token_begin_idx: {wrong_begin} -> {correct_begin} (offset={offset})"
        )

    def _per_image_tokens(self, cfg) -> List[int]:
        """Return post-merge token count for each of the num_input_images images.

        FLOPs estimation assumes: 1 exterior (head) camera + remaining all wrist cameras.
        Both "exterior" and "wrist" must be present in camera_size_config.
        """
        from omegaconf import OmegaConf

        patch_size = cfg.vision.patch_size
        merge_size = cfg.vision.spatial_merge_size
        camera_size_config = OmegaConf.to_object(cfg.camera_size_config)

        def _t(H, W):
            return (H // patch_size // merge_size) * (W // patch_size // merge_size)

        n_cams = cfg.num_input_images // cfg.num_obs_steps
        per_camera = (
            [_t(*camera_size_config["exterior"])]
            + [_t(*camera_size_config["wrist_left"])]
            + [_t(*camera_size_config["wrist_right"])]
        )
        return per_camera * cfg.num_obs_steps

    def estimate_training_flops_per_sample(self) -> int:
        """Qwen3.5 FLOPs: ViT uses pre-merge patches, no external projector, hybrid attention."""
        cfg = self.model.cfg
        vision_tower = self.model.vision_tower
        num_merger_params = sum(p.numel() for p in vision_tower.merger.parameters())
        num_vit_params = sum(p.numel() for p in vision_tower.parameters()) - num_merger_params

        merge_size = cfg.vision.spatial_merge_size
        vis_L = cfg.vision.depth
        vis_H = cfg.vision.num_heads
        vis_d_head = cfg.vision.hidden_size // vis_H

        # ViT FLOPs: iterate per-image to support mixed camera resolutions.
        vision_flops = 0
        for vis_tokens in self._per_image_tokens(cfg):
            vis_patches = vis_tokens * (merge_size**2)  # pre-merge (ViT attention)
            vision_flops += 6 * num_vit_params * vis_patches + 6 * num_merger_params * vis_tokens
            vision_flops += 12 * vis_L * vis_H * vis_d_head * vis_patches**2

        per_img_tokens = self._per_image_tokens(cfg)
        S_vlm = sum(per_img_tokens) + cfg.max_text_tokens
        S_action = cfg.horizon_steps
        vlm_flops = self.model.vlm.estimate_flops(query_len=S_vlm)
        ae_flops = self.model.action_expert.estimate_flops(
            query_len=S_action, kv_len=S_vlm + S_action
        )

        return vision_flops + vlm_flops + ae_flops
