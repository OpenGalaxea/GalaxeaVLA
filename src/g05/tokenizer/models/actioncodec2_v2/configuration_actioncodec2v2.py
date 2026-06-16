# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""HuggingFace-style configuration for ActionCodecV2."""

import math
from typing import List, Optional

from transformers import PretrainedConfig


class ActionCodecV2Config(PretrainedConfig):
    """
    Configuration for ActionCodecV2Model.

    ActionCodecV2 is a universal action tokenizer that accepts an arbitrary dict of action
    components {key: (B, T, D_i)} — each with a potentially different last dimension —
    pads/clips them all to ``max_component_dim``, and encodes through a shared
    2D Conv-Transformer + EMA-RVQ pipeline inspired by FasterV2.

    Key design choices vs FasterV2:
    - Generic dict input (any keys, any D_i) instead of hardcoded WBC body parts
    - ``conv_in`` uses kernel=(1, conv_in_action_kernel) to map A=max_component_dim → A_code
      (e.g. A=9 → A=8 when conv_in_action_kernel=2, giving a power-of-2 token count)
    - GEGLU FFN (full 4× expansion) instead of SwiGLU — better capacity density for small models
    - No WeightNorm on convolutions, plain nn.Conv2d with kaiming init
    - Thread-safe BlockDCT (frame_length passed explicitly, not stored as instance state)
    - Single "independent" pretransform mode only (no bimanual/whole_body dispatch)

    Shape flow (defaults):
        {key: (B, 32, D_i)}
            → pad/clip to max_component_dim=9
            → (optional) Block DCT (block_size=8)
            → pretransform: (B, 32, 9) → (B, lh=8, h=4, a=9)
            → conv_in Conv2d(8→128, kernel=(1,2)): a 9→8
            → encoder strides [[1,1],[2,1],[2,1]]: h 4→1
            → flatten: (B, 128, 8)  [code_h=1 × code_a=8]
            → EMA-RVQ n_codebooks=4: codes (B, 4, 8)
            → decoder (mirror of encoder)
            → conv_out ConvTranspose2d(128→8, kernel=(1,2)): a 8→9
            → reverse pretransform: (B, 32, 9)
            → (optional) inverse DCT
            → slice to original D_i

        tokens per key = code_h × code_a × n_codebooks = 1 × 8 × 4 = 32
    """

    model_type = "actioncodec2_v2"

    def __init__(
        self,
        # ── Input ──────────────────────────────────────────────────────────────
        max_component_dim: int = 9,
        horizon: int = 32,
        # ── Pretransform ───────────────────────────────────────────────────────
        horizon_patch_size: int = 8,
        # ── conv_in / conv_out ─────────────────────────────────────────────────
        # Conv2d(horizon_patch_size → encoder_channels, kernel=(1, conv_in_action_kernel))
        # maps A: max_component_dim → (max_component_dim - conv_in_action_kernel + 1)
        # With max_component_dim=9, conv_in_action_kernel=2 → code_a = 8 (power of 2)
        conv_in_action_kernel: int = 2,
        encoder_channels: int = 128,
        # ── Encoder / Decoder ──────────────────────────────────────────────────
        latent_dim: int = 128,
        c_mults: Optional[List[int]] = None,
        strides: Optional[List[List[int]]] = None,
        transformer_depths: Optional[List[int]] = None,
        num_heads: int = 4,
        dim_heads: int = 64,
        # FFN: GEGLU with full 4× expansion (replaces SwiGLU).
        # SwiGLU was designed for large models (2/3 compression to save compute).
        # For small models, maximising FFN capacity is more important.
        # ffn_mult=4: same total params as a standard 4× FFN.
        # ffn_mult=6~8: explicitly larger FFN — more capacity for the ~8 M model budget.
        ffn_mult: float = 4.0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 0.01,
        use_qk_norm: bool = True,
        rope_base: int = 10000,
        # ── Block DCT ──────────────────────────────────────────────────────────
        use_block_dct: bool = True,
        block_dct_block_size: int = 8,
        # ── EMA-RVQ ────────────────────────────────────────────────────────────
        n_codebooks: int = 4,
        codebook_size: int = 4096,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        ema_decay: float = 0.95,
        threshold_ema_dead: float = 2.0,
        # Rotation-trick STE gives better gradients than vanilla STE for VQ.
        use_rotation_trick: bool = True,
        # ── Loss weights ───────────────────────────────────────────────────────
        commitment_loss_weight: float = 0.25,
        reconstruction_loss_weight: float = 1.0,
        # ── Consistency regularization ─────────────────────────────────────────
        # Set > 0 to enable token-space consistency loss (λ in the paper).
        # layer_weights linearly interpolate from begin→end over [0, end_fraction*max_steps].
        # None → use spec defaults for n_codebooks.
        consistency_loss_weight: float = 0.0,
        # Time-shift augmentation: δ ~ Uniform_int([-delta_max, delta_max] \ {0})
        # delta_max linearly interpolates begin→end over the schedule.
        consistency_delta_max_begin: int = 1,
        consistency_delta_max_end: int = 3,
        # Amplitude-scale augmentation: α ~ Uniform(1-eps, 1+eps) per sample
        # eps linearly interpolates begin→end over the schedule.
        consistency_eps_begin: float = 0.02,
        consistency_eps_end: float = 0.10,
        consistency_layer_weights_begin: Optional[List[float]] = None,
        consistency_layer_weights_end: Optional[List[float]] = None,
        # Fraction of total training steps at which begin values start ramping up.
        # Before begin_fraction: stay at begin values. After end_fraction: stay at end values.
        consistency_schedule_begin_fraction: float = 0.0,
        # Fraction of total training steps at which end values are fully reached.
        consistency_schedule_end_fraction: float = 0.9,
        # ── Residual FSQ ───────────────────────────────────────────────────────
        # When use_fsq=True, the EMA-VQ at each RVQ level is replaced by FSQ.
        # codebook_size and codebook_dim are then fully determined by fsq_levels:
        #   codebook_size = product(fsq_levels), codebook_dim = len(fsq_levels)
        # Set codebook_size / codebook_dim to null in config when use_fsq=True.
        use_fsq: bool = False,
        fsq_levels: Optional[List[int]] = None,  # default [4,4,4,4,4,4] → size=4096, dim=6
        # ── Lipschitz Encoder ──────────────────────────────────────────────────
        # Apply spectral_norm to all Linear/Conv2d in conv_in + ActionEncoder,
        # skipping Attention submodules (to_qkv, to_out).
        use_lipschitz_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_component_dim = max_component_dim
        self.horizon = horizon

        self.horizon_patch_size = horizon_patch_size
        assert horizon % horizon_patch_size == 0, (
            f"horizon ({horizon}) must be divisible by horizon_patch_size ({horizon_patch_size})"
        )

        self.conv_in_action_kernel = conv_in_action_kernel
        self.encoder_channels = encoder_channels

        self.latent_dim = latent_dim
        self.c_mults = c_mults if c_mults is not None else [1, 2, 2]
        self.strides = strides if strides is not None else [[1, 1], [2, 1], [2, 1]]
        self.transformer_depths = (
            transformer_depths if transformer_depths is not None else [2, 2, 2]
        )
        assert len(self.c_mults) == len(self.strides) == len(self.transformer_depths), (
            "c_mults, strides, and transformer_depths must have the same length"
        )

        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.ffn_mult = ffn_mult
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base

        self.use_block_dct = use_block_dct
        self.block_dct_block_size = block_dct_block_size

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.ema_decay = ema_decay
        self.threshold_ema_dead = threshold_ema_dead
        self.use_rotation_trick = use_rotation_trick

        self.commitment_loss_weight = commitment_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_delta_max_begin = consistency_delta_max_begin
        self.consistency_delta_max_end = consistency_delta_max_end
        self.consistency_eps_begin = consistency_eps_begin
        self.consistency_eps_end = consistency_eps_end
        self.consistency_layer_weights_begin = consistency_layer_weights_begin
        self.consistency_layer_weights_end = consistency_layer_weights_end
        self.consistency_schedule_begin_fraction = consistency_schedule_begin_fraction
        self.consistency_schedule_end_fraction = consistency_schedule_end_fraction
        self.use_fsq = use_fsq
        self.fsq_levels = fsq_levels if fsq_levels is not None else [4, 4, 4, 4, 4, 4]
        self.use_lipschitz_encoder = use_lipschitz_encoder

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def code_a(self) -> int:
        """Number of action-axis codes after conv_in kernel reduction."""
        return self.max_component_dim - self.conv_in_action_kernel + 1

    @property
    def code_h(self) -> int:
        """Number of time-axis codes after all encoder stride reductions."""
        h = self.horizon // self.horizon_patch_size
        for stride_h, _ in self.strides:
            h = h // stride_h
        return h

    @property
    def tokens_per_key(self) -> int:
        """Total tokens per input key = code_h × code_a × n_codebooks."""
        return self.code_h * self.code_a * self.n_codebooks

    @property
    def vocab_size_with_specials(self) -> int:
        """Effective codebook size. For FSQ, derived from fsq_levels; otherwise codebook_size."""
        if self.use_fsq:
            return math.prod(self.fsq_levels)
        return self.codebook_size
