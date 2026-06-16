# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
Core building blocks for ActionCodecV2.

All modules are stateless helpers or lightweight nn.Modules. They are imported
by modeling_actioncodec2v2.py to assemble the full model.

Improvements over FasterV2:
- No WeightNorm: plain nn.Conv2d with kaiming_normal_ init
- No SnakeBeta: snake_beta function was never defined (latent bug); use nn.SiLU
- GEGLU FFN with full 4× expansion instead of SwiGLU (better capacity for small models)
- Thread-safe BlockDCT: frame_length passed as argument, not stored as instance state
- Simplified RoPE: xpos / natten removed, only the essentials
- Sliding window: raises RuntimeError if Flash Attention 2 is not available
- LayerNorm eps=1e-6 (standard) instead of FasterV2's non-standard 1e-2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_func

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False
    flash_attn_func = None


# ─────────────────────────────────────────────────────────────────────────────
# Block DCT (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────


class BlockDCT(nn.Module):
    """
    Block DCT-II with orthonormal normalization.

    Applies DCT independently on non-overlapping blocks of ``block_size`` frames
    along the time axis.  The basis matrix is stored as a non-persistent buffer
    so it moves to the right device automatically.

    Thread-safe: ``frame_length`` (original sequence length before possible
    padding) is passed explicitly to ``idct``; it is never stored as instance
    state.
    """

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("basis", self._build_basis(block_size), persistent=False)

    @staticmethod
    def _build_basis(n: int) -> torch.Tensor:
        """Orthonormal DCT-II basis matrix C ∈ R^{n×n}."""
        k = torch.arange(n).float()
        t = torch.arange(n).float()
        C = torch.cos(math.pi / n * (t + 0.5).unsqueeze(0) * k.unsqueeze(1))  # (n, n)
        C[0] *= math.sqrt(1.0 / n)
        C[1:] *= math.sqrt(2.0 / n)
        return C  # (n, n)

    def dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward block DCT.

        Args:
            x: (B, T, D)  — T need not be a multiple of block_size (zero-padded)
        Returns:
            y: (B, T_padded, D)
        """
        B, T, D = x.shape
        bs = self.block_size
        pad = (bs - T % bs) % bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        # reshape to (B * num_blocks, bs, D)
        num_blocks = x.shape[1] // bs
        x = x.reshape(B * num_blocks, bs, D)
        # C: (bs, bs)  →  y[k,d] = Σ_n C[k,n] x[n,d]
        y = torch.einsum("kn, bnd -> bkd", self.basis.to(x), x)
        return y.reshape(B, num_blocks * bs, D)

    def idct(self, y: torch.Tensor, frame_length: int) -> torch.Tensor:
        """
        Inverse block DCT.

        Args:
            y:            (B, T_padded, D)  — output of ``dct``
            frame_length: original T before padding (used to trim output)
        Returns:
            x: (B, frame_length, D)
        """
        B, T_pad, D = y.shape
        bs = self.block_size
        num_blocks = T_pad // bs
        y = y.reshape(B * num_blocks, bs, D)
        # Inverse: C is orthonormal → C^{-1} = C^T
        x = torch.einsum("nk, bkd -> bnd", self.basis.to(y), y)
        x = x.reshape(B, num_blocks * bs, D)
        return x[:, :frame_length, :]


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embedding (simplified)
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoPE).  Only the essentials: no xpos, no natten.

    Partial rotary: applies rotation to the first ``dim`` channels of each head;
    the rest pass through unchanged.  ``dim`` defaults to ``dim_heads // 2``
    (capped at 32) which matches FasterV2's convention.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
            self._cache[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self._cache[key]

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        return self.get_cos_sin(seq_len, device, dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to the first ``cos.shape[-1]`` dims of q and k."""
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    """
    Multi-head self-attention with:
    - RoPE positional encoding (partial)
    - Per-head QK-LayerNorm for numerical stability
    - Flash Attention 2 / PyTorch SDPA / manual fallback (priority order)
    - output projection zero-initialised for training stability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_heads: int = 64,
        use_qk_norm: bool = True,
        rope_base: int = 10000,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        inner_dim = num_heads * dim_heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)  # zero-init output proj

        # QK-LayerNorm (per head, applied after q/k split)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(dim_heads, eps=1e-6)
            self.k_norm = nn.LayerNorm(dim_heads, eps=1e-6)

        # RoPE: rotate the first ``rope_dim`` channels of each head
        rope_dim = max(dim_heads // 2, 32)
        self.rope = RotaryEmbedding(rope_dim, base=rope_base)

        self.sliding_window = sliding_window
        if sliding_window is not None and not _FLASH_AVAILABLE:
            raise RuntimeError(
                "sliding_window requires Flash Attention 2 but it is not installed. "
                "Install flash-attn or disable sliding_window."
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D)
            mask: (B, T, T) bool mask; True = positions to IGNORE (will be −∞)
        Returns:
            (B, T, D)
        """
        B, T, _ = x.shape
        qkv = self.to_qkv(x)  # (B, T, 3 * H * dim_heads)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, H, T, dim_heads)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        cos, sin = self.rope(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, rope_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)

        # Attention kernel selection. flash_attn only supports fp16/bf16; the action
        # tokenizer runs fp32 at serve time, so pin fp32 to SDPA (bit-identical to the
        # no-FA2 path) — otherwise flash_attn_func raises "only support fp16 and bf16".
        if _FLASH_AVAILABLE and mask is None and q.dtype in (torch.float16, torch.bfloat16):
            # flash_attn expects (B, T, H, D)
            q_fa = rearrange(q, "b h t d -> b t h d")
            k_fa = rearrange(k, "b h t d -> b t h d")
            v_fa = rearrange(v, "b h t d -> b t h d")
            window = self.sliding_window if self.sliding_window is not None else (-1, -1)
            out = flash_attn_func(q_fa, k_fa, v_fa, window_size=window)  # (B, T, H, D)
            out = rearrange(out, "b t h d -> b t (h d)")
        else:
            # PyTorch SDPA (fused; efficient on modern GPUs without FA2)
            if mask is not None:
                attn_mask = mask.unsqueeze(1)  # (B, 1, T, T)
                attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))
            else:
                attn_mask = None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = rearrange(out, "b h t d -> b t (h d)")

        return self.to_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# GEGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────


class GEGLUFFFN(nn.Module):
    """
    Feed-forward network using GEGLU activation with full 4× expansion.

    GEGLU: ``GEGLU(x) = x1 * GELU(x2)`` where ``(x1, x2) = split(W_up(x))``.

    Why GEGLU instead of SwiGLU:
        SwiGLU is typically used with a 2/3× inner_dim compression to keep
        parameter count equal to a standard 4× FFN — this trades capacity for
        compute savings, which is the right trade-off for *large* models.
        For small models (~8 M params) we want to *maximise* capacity; use
        full 4× expansion (``ffn_mult=4``) or even larger (``ffn_mult=6~8``).

    Parameters:
        dim:       input / output dimension
        ffn_mult:  expansion multiplier; inner_dim = int(dim * ffn_mult)
    """

    def __init__(self, dim: int, ffn_mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * ffn_mult)
        # Fused up-projection: produces (x1, x2) of shape (B, T, inner_dim) each
        self.w_up = nn.Linear(dim, inner_dim * 2, bias=False)
        self.w_down = nn.Linear(inner_dim, dim, bias=False)
        nn.init.zeros_(self.w_down.weight)  # zero-init output proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(x1 * F.gelu(x2))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:  LN → Attention → residual (+LayerScale)
                                  LN → GEGLU-FFN → residual (+LayerScale)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_heads: int = 64,
        ffn_mult: float = 4.0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 0.01,
        use_qk_norm: bool = True,
        rope_base: int = 10000,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            dim_heads=dim_heads,
            use_qk_norm=use_qk_norm,
            rope_base=rope_base,
            sliding_window=sliding_window,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = GEGLUFFFN(dim, ffn_mult=ffn_mult)

        self.ls1 = nn.Parameter(torch.full([dim], layer_scale_init)) if use_layer_scale else None
        self.ls2 = nn.Parameter(torch.full([dim], layer_scale_init)) if use_layer_scale else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask=mask)
        if self.ls1 is not None:
            attn_out = attn_out * self.ls1
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        if self.ls2 is not None:
            ffn_out = ffn_out * self.ls2
        x = x + ffn_out
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Conv + Transformer blocks (encoder / decoder building blocks)
# ─────────────────────────────────────────────────────────────────────────────


def _make_transformer_stack(
    n_layers: int,
    dim: int,
    num_heads: int,
    dim_heads: int,
    ffn_mult: float,
    use_layer_scale: bool,
    layer_scale_init: float,
    use_qk_norm: bool,
    rope_base: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            TransformerBlock(
                dim,
                num_heads=num_heads,
                dim_heads=dim_heads,
                ffn_mult=ffn_mult,
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
                use_qk_norm=use_qk_norm,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ]
    )


class DownBlock2D(nn.Module):
    """
    Encoder building block:
        Conv2d(stride_h, stride_a) → N × TransformerBlock

    The Conv2d downsamples the H (time-patch) axis; stride_a=1 by default
    so the action-dimension axis is never strided in the encoder.

    Kernel size along H is ``2 * stride_h`` (standard for strided conv encoders
    in audio codecs like DAC/EnCodec); kernel along A is 1 (no mixing).

    No WeightNorm: plain Conv2d with kaiming_normal_ init.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_h: int,
        stride_a: int,
        n_transformer_layers: int,
        num_heads: int,
        dim_heads: int,
        ffn_mult: float,
        use_layer_scale: bool,
        layer_scale_init: float,
        use_qk_norm: bool,
        rope_base: int,
    ):
        super().__init__()
        if stride_h > 1 or in_channels != out_channels:
            kernel_h = 2 * stride_h if stride_h > 1 else 1
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_h, 1),
                stride=(stride_h, stride_a),
                padding=(kernel_h // 2 - (1 if stride_h > 1 else 0), 0),
            )
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            self.conv = nn.Identity()

        # The transformer operates on the (H × A) sequence flattened to 1-D.
        self.transformer_layers = _make_transformer_stack(
            n_transformer_layers,
            out_channels,
            num_heads,
            dim_heads,
            ffn_mult,
            use_layer_scale,
            layer_scale_init,
            use_qk_norm,
            rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, A)"""
        x = self.conv(x)  # (B, C', H', A')
        B, C, H, A = x.shape
        x = rearrange(x, "b c h a -> b (h a) c")  # (B, H*A, C)
        for layer in self.transformer_layers:
            x = layer(x)
        x = rearrange(x, "b (h a) c -> b c h a", h=H, a=A)
        return x


class UpBlock2D(nn.Module):
    """
    Decoder building block:
        N × TransformerBlock → ConvTranspose2d(stride_h, stride_a)

    Mirror of DownBlock2D.  Up-sampling happens *after* the transformer
    (decoder runs in reverse block order compared to the encoder, but within
    each block the transformer precedes the upsampling conv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_h: int,
        stride_a: int,
        n_transformer_layers: int,
        num_heads: int,
        dim_heads: int,
        ffn_mult: float,
        use_layer_scale: bool,
        layer_scale_init: float,
        use_qk_norm: bool,
        rope_base: int,
    ):
        super().__init__()
        self.transformer_layers = _make_transformer_stack(
            n_transformer_layers,
            in_channels,
            num_heads,
            dim_heads,
            ffn_mult,
            use_layer_scale,
            layer_scale_init,
            use_qk_norm,
            rope_base,
        )

        if stride_h > 1 or in_channels != out_channels:
            kernel_h = 2 * stride_h if stride_h > 1 else 1
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_h, 1),
                stride=(stride_h, stride_a),
                padding=(kernel_h // 2 - (1 if stride_h > 1 else 0), 0),
            )
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, A)"""
        B, C, H, A = x.shape
        x = rearrange(x, "b c h a -> b (h a) c")
        for layer in self.transformer_layers:
            x = layer(x)
        x = rearrange(x, "b (h a) c -> b c h a", h=H, a=A)
        x = self.conv(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder stacks
# ─────────────────────────────────────────────────────────────────────────────


class ActionEncoder(nn.Module):
    """
    Stack of DownBlock2D blocks followed by a 1×1 conv projection to latent_dim.

    Channel dims: encoder_channels → encoder_channels*c_mults[0]
                  → encoder_channels*c_mults[1] → … → latent_dim
    """

    def __init__(
        self,
        in_channels: int,  # == encoder_channels (after conv_in)
        c_mults: List[int],
        strides: List[List[int]],
        transformer_depths: List[int],
        latent_dim: int,
        num_heads: int,
        dim_heads: int,
        ffn_mult: float,
        use_layer_scale: bool,
        layer_scale_init: float,
        use_qk_norm: bool,
        rope_base: int,
    ):
        super().__init__()
        channel_dims = [in_channels * m for m in c_mults]
        # prepend in_channels so we get in→out pairs
        dims = [in_channels] + channel_dims

        self.blocks = nn.ModuleList()
        for i, (stride_h_a, n_layers, d_in, d_out) in enumerate(
            zip(strides, transformer_depths, dims[:-1], dims[1:])
        ):
            stride_h, stride_a = stride_h_a
            self.blocks.append(
                DownBlock2D(
                    d_in,
                    d_out,
                    stride_h=stride_h,
                    stride_a=stride_a,
                    n_transformer_layers=n_layers,
                    num_heads=num_heads,
                    dim_heads=dim_heads,
                    ffn_mult=ffn_mult,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    use_qk_norm=use_qk_norm,
                    rope_base=rope_base,
                )
            )

        # 1×1 projection to latent_dim
        self.out_proj = nn.Conv2d(dims[-1], latent_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, A) → (B, latent_dim, H', A')"""
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


class ActionDecoder(nn.Module):
    """
    Mirror of ActionEncoder: 1×1 proj from latent_dim, then UpBlock2D blocks
    (in reverse stride order).
    """

    def __init__(
        self,
        out_channels: int,  # == encoder_channels (before conv_out)
        c_mults: List[int],
        strides: List[List[int]],
        transformer_depths: List[int],
        latent_dim: int,
        num_heads: int,
        dim_heads: int,
        ffn_mult: float,
        use_layer_scale: bool,
        layer_scale_init: float,
        use_qk_norm: bool,
        rope_base: int,
    ):
        super().__init__()
        channel_dims = [out_channels * m for m in c_mults]
        dims = [out_channels] + channel_dims  # same as encoder

        # Input projection from latent_dim to the deepest channel dimension
        self.in_proj = nn.Conv2d(latent_dim, dims[-1], kernel_size=1)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        # Build blocks in reverse order (deepest → shallowest)
        self.blocks = nn.ModuleList()
        rev_strides = list(reversed(strides))
        rev_depths = list(reversed(transformer_depths))
        rev_dims_in = list(reversed(dims[1:]))  # deepest to shallowest
        rev_dims_out = list(reversed(dims[:-1]))

        for stride_h_a, n_layers, d_in, d_out in zip(
            rev_strides, rev_depths, rev_dims_in, rev_dims_out
        ):
            stride_h, stride_a = stride_h_a
            self.blocks.append(
                UpBlock2D(
                    d_in,
                    d_out,
                    stride_h=stride_h,
                    stride_a=stride_a,
                    n_transformer_layers=n_layers,
                    num_heads=num_heads,
                    dim_heads=dim_heads,
                    ffn_mult=ffn_mult,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    use_qk_norm=use_qk_norm,
                    rope_base=rope_base,
                )
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim, H', A') → (B, out_channels, H, A)"""
        x = self.in_proj(z)
        for block in self.blocks:
            x = block(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# FSQ Quantizer (Mentzer et al. 2023)
# ─────────────────────────────────────────────────────────────────────────────


class FSQQuantize(nn.Module):
    """
    Finite Scalar Quantization (FSQ) drop-in replacement for EMAVectorQuantize.

    Instead of learning a codebook via EMA, FSQ quantizes each dimension
    independently to a finite set of levels defined by ``levels``.

    Interface is identical to EMAVectorQuantize:
        forward(z: B,D,T) → (z_q_out: B,D,T,  commit_loss: B,  codes: B,T)
        decode_codes(codes: B,T) → (B,D,T)

    ``commit_loss`` is always zeros (FSQ has no commitment loss).
    ``cluster_size`` is a zero buffer kept for interface compatibility with
    the codebook-stats logging loop in ActionCodecV2Model.forward().

    Quantization pipeline:
        z (B,D,T)
          → in_proj Linear(D → codebook_dim)
          → bound:  z_bounded_i = half_width_i * tanh(z_i)
          → STE round
          → out_proj Linear(codebook_dim → D)

    Mixed-radix integer codes:
        idx_i  = round(z_bounded_i) + half_width_i  ∈ {0, …, L_i-1}
        code   = Σ_i  idx_i * base_i,  base_i = product(levels[:i])
    """

    def __init__(
        self,
        input_dim: int,
        levels: List[int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.codebook_dim = len(levels)
        self.codebook_size = math.prod(levels)
        # dummy for interface compat with EMAVectorQuantize
        self.threshold_ema_dead = 0.0

        # Projections (always non-identity because codebook_dim != input_dim in general)
        self.in_proj = nn.Linear(input_dim, self.codebook_dim, bias=False)
        self.out_proj = nn.Linear(self.codebook_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        # Per-dimension half-widths: (L_i - 1) / 2
        half_widths = torch.tensor([(L - 1) / 2.0 for L in levels], dtype=torch.float32)
        self.register_buffer("half_widths", half_widths)  # (codebook_dim,)

        # Mixed-radix bases: base_i = product(levels[:i])
        bases = [1]
        for L in levels[:-1]:
            bases.append(bases[-1] * L)
        self.register_buffer("bases", torch.tensor(bases, dtype=torch.long))  # (codebook_dim,)

        # Dummy buffers for interface compatibility (never updated)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("inited", torch.tensor(True))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bound(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Bound each dimension to [-half_width_i, half_width_i] via tanh.

        Args:
            z_e: (B, codebook_dim, T)
        Returns:
            z_bounded: (B, codebook_dim, T)
        """
        hw = self.half_widths.view(1, -1, 1)
        return hw * torch.tanh(z_e)

    def _ste_round(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """
        Round to nearest grid point via shift-round-shift. STE backward.

        For L levels with half_width hw = (L-1)/2, the valid grid points are
        {-hw, -hw+1, ..., hw-1, hw}.  Shifting by +hw maps them to {0,...,L-1}
        (integers), enabling standard rounding; shifting back by -hw restores
        the original grid.  This handles both odd L (hw integer) and even L
        (hw half-integer) correctly without index collisions.
        """
        hw = self.half_widths.view(1, -1, 1)
        z_shifted = z_bounded + hw
        z_q_shifted = z_shifted + (z_shifted.round() - z_shifted).detach()
        return z_q_shifted - hw

    def _to_codes(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized real values to mixed-radix integer codes.

        Args:
            z_q: (B, codebook_dim, T)  values in [-half_width_i, half_width_i]
        Returns:
            codes: (B, T)  integers in [0, codebook_size)
        """
        hw = self.half_widths.view(1, -1, 1)
        idx = (z_q + hw).round().long()
        bases = self.bases.view(1, -1, 1)
        return (idx * bases).sum(dim=1)

    def _from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode mixed-radix integer codes back to real values.

        Args:
            codes: (B, T)
        Returns:
            z_q: (B, codebook_dim, T)  real values in [-half_width_i, half_width_i]
        """
        levels = self.bases.new_tensor(self.levels)  # (codebook_dim,)
        idx = (codes.unsqueeze(1) // self.bases.view(1, -1, 1)) % levels.view(1, -1, 1)
        hw = self.half_widths.view(1, -1, 1)
        return idx.float() - hw

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, T)
        Returns:
            z_q_out:     (B, D, T)   quantized representation projected back to D
            commit_loss: (B,)        always zeros (FSQ has no commitment loss)
            codes:       (B, T)      mixed-radix integer codes
        """
        orig_dtype = z.dtype
        z = z.float()

        # Project to codebook space: (B, D, T) → (B, codebook_dim, T)
        z_e = rearrange(z, "b d t -> b t d")
        z_e = self.in_proj(z_e)
        z_e = rearrange(z_e, "b t d -> b d t")

        # Bound + STE round
        z_bounded = self._bound(z_e)
        z_q = self._ste_round(z_bounded)

        # Mixed-radix codes
        codes = self._to_codes(z_q)  # (B, T)

        # Project back to input_dim
        z_q_out = rearrange(z_q, "b d t -> b t d")
        z_q_out = self.out_proj(z_q_out)
        z_q_out = rearrange(z_q_out, "b t d -> b d t")  # (B, D, T)

        commit_loss = torch.zeros(z.shape[0], device=z.device)
        return z_q_out.to(orig_dtype), commit_loss, codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, T)
        Returns:
            z_q: (B, D, T)
        """
        z_q = self._from_codes(codes)  # (B, codebook_dim, T)
        z_q_out = rearrange(z_q.float(), "b d t -> b t d")
        z_q_out = self.out_proj(z_q_out)
        return rearrange(z_q_out, "b t d -> b d t")  # (B, D, T)


# ─────────────────────────────────────────────────────────────────────────────
# EMA Vector Quantizer
# ─────────────────────────────────────────────────────────────────────────────


def _sample_vectors(samples: torch.Tensor, n: int) -> torch.Tensor:
    """Randomly sample n rows from samples (with replacement if needed)."""
    N = samples.shape[0]
    if N >= n:
        idx = torch.randperm(N, device=samples.device)[:n]
    else:
        idx = torch.randint(0, N, (n,), device=samples.device)
    return samples[idx].float()


def _kmeans(samples: torch.Tensor, n_clusters: int, n_iters: int = 10):
    """Lloyd's k-means in fp32. Returns (means, cluster_sizes)."""
    dim = samples.shape[-1]
    means = _sample_vectors(samples, n_clusters)  # (K, D)

    for _ in range(n_iters):
        # (N, K) pairwise squared distances via expansion
        dists = (
            samples.float().pow(2).sum(1, keepdim=True)
            - 2 * samples.float() @ means.t()
            + means.t().float().pow(2).sum(0, keepdim=True)
        )
        buckets = dists.argmin(dim=-1)  # (N,)
        bins = torch.bincount(buckets, minlength=n_clusters)
        bins_safe = bins.masked_fill(bins == 0, 1)

        new_means = torch.zeros(n_clusters, dim, dtype=torch.float32, device=samples.device)
        new_means.scatter_add_(0, buckets.unsqueeze(1).expand(-1, dim), samples.float())
        new_means = new_means / bins_safe.float().unsqueeze(1)
        means = torch.where((bins == 0).unsqueeze(1), means, new_means)

    # Final cluster sizes
    dists = (
        samples.float().pow(2).sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.t().float().pow(2).sum(0, keepdim=True)
    )
    bins = torch.bincount(dists.argmin(dim=-1), minlength=n_clusters).float()
    return means, bins


def _ema_inplace(moving_avg: torch.Tensor, new_val: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new_val.float(), alpha=1.0 - decay)


class EMAVectorQuantize(nn.Module):
    """
    Single-level EMA vector quantizer with:
    - K-means initialisation (first forward pass)
    - EMA codebook updates (no gradient through codebook)
    - Laplace smoothing to avoid division by zero
    - Dead-code replacement (random batch sample replaces cold codewords)
    - Optional rotation-trick STE for better encoder gradients
    - DDP-safe: all_reduce for EMA counts/sums, broadcast for k-means / dead codes

    Input/output convention: (B, D, T) — channels-first, like a 1-D Conv.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float = 0.25,
        decay: float = 0.95,
        epsilon: float = 1e-5,
        threshold_ema_dead: float = 2.0,
        use_rotation_trick: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.use_rotation_trick = use_rotation_trick

        # Low-dim projection (matches FasterV2's in_project / out_project)
        if input_dim != codebook_dim:
            self.in_proj = nn.Linear(input_dim, codebook_dim, bias=False)
            self.out_proj = nn.Linear(codebook_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.zeros_(self.out_proj.weight)  # zero-init for stability
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # EMA buffers (fp32 always)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("inited", torch.tensor(False))

    # ── EMA update ────────────────────────────────────────────────────────────

    def _ema_update(self, encodings: torch.Tensor, onehot: torch.Tensor):
        """encodings: (N, D'); onehot: (N, K)"""
        new_cluster_size = onehot.sum(0)  # (K,)
        new_embed_sum = encodings.t() @ onehot  # (D', K)

        if dist.is_initialized():
            dist.all_reduce(new_cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_embed_sum, op=dist.ReduceOp.SUM)

        _ema_inplace(self.cluster_size, new_cluster_size, self.decay)
        _ema_inplace(self.embed_avg, new_embed_sum.t(), self.decay)

        # Laplace-smoothed normalisation
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        self.codebook.copy_((self.embed_avg / smoothed.unsqueeze(1)).float())

    # ── Dead-code replacement ─────────────────────────────────────────────────

    def _replace_dead_codes(self, encodings: torch.Tensor):
        if self.threshold_ema_dead <= 0:
            return
        dead = self.cluster_size < self.threshold_ema_dead  # (K,)
        if not dead.any():
            return
        n_dead = int(dead.sum().item())
        if dist.is_initialized() and dist.get_rank() == 0:
            samples = _sample_vectors(encodings.float(), n_dead)
        else:
            samples = torch.zeros(n_dead, self.codebook_dim, device=encodings.device)
        if dist.is_initialized():
            dist.broadcast(samples, src=0)
        self.codebook[dead] = samples.to(self.codebook.dtype)

    # ── K-means init ─────────────────────────────────────────────────────────

    def _init_codebook(self, encodings: torch.Tensor):
        if self.inited.item():
            return
        if dist.is_initialized() and dist.get_rank() == 0:
            means, sizes = _kmeans(encodings.float(), self.codebook_size)
        else:
            means = torch.zeros(self.codebook_size, self.codebook_dim, device=encodings.device)
            sizes = torch.zeros(self.codebook_size, device=encodings.device)
        if dist.is_initialized():
            dist.broadcast(means, src=0)
            dist.broadcast(sizes, src=0)
        self.codebook.copy_(means)
        self.embed_avg.copy_(means)
        self.cluster_size.copy_(sizes)
        self.inited.fill_(True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, T)  input latent (channels-first)
        Returns:
            z_q:          (B, D, T)  quantised + projected back to D
            commit_loss:  (B,)       per-sample commitment loss
            codes:        (B, T)     codebook indices
        """
        orig_dtype = z.dtype
        z = z.float()

        # Project to codebook dimension: (B, D, T) → (B, D', T)
        # in_proj operates on channels; use it as a 1×1 linear
        if isinstance(self.in_proj, nn.Identity):
            z_e = z
        else:
            z_e = rearrange(z, "b d t -> b t d")
            z_e = self.in_proj(z_e)
            z_e = rearrange(z_e, "b t d -> b d t")

        # Flatten to (N, D') for distance computation
        N_flat = z_e.shape[0] * z_e.shape[2]
        enc = rearrange(z_e, "b d t -> (b t) d").float()

        # K-means init on first batch (use detached enc — codebook init is gradient-free)
        if not self.inited.item():
            self._init_codebook(enc.detach())

        # L2 distances to codebook: (N, K)
        dist_sq = (
            enc.pow(2).sum(1, keepdim=True)
            - 2 * enc @ self.codebook.float().t()
            + self.codebook.float().pow(2).sum(1, keepdim=True).t()
        )
        codes_flat = dist_sq.argmin(dim=-1)  # (N,)
        codes = rearrange(codes_flat, "(b t) -> b t", b=z.shape[0])  # (B, T)

        # Lookup quantised vectors: (N, D')
        z_q_proj = F.embedding(codes_flat, self.codebook.float())  # (N, D')
        z_q_proj = rearrange(z_q_proj, "(b t) d -> b d t", b=z.shape[0])  # (B, D', T)

        # Commitment loss (per sample)
        commit_loss = F.mse_loss(z_e, z_q_proj.detach(), reduction="none").mean([1, 2])
        commit_loss = commit_loss * self.commitment  # (B,)

        # EMA updates (training only)
        if self.training and torch.is_grad_enabled():
            onehot = F.one_hot(codes_flat, self.codebook_size).float()
            self._ema_update(enc.detach(), onehot)
            self._replace_dead_codes(enc.detach())

        # Straight-through estimator
        if self.use_rotation_trick:
            z_q = _rotation_trick_ste(z_e, z_q_proj)
        else:
            z_q = (z_q_proj - z_e).detach() + z_e  # (B, D', T)

        # Project back to input_dim
        if isinstance(self.out_proj, nn.Identity):
            z_q_out = z_q
        else:
            z_q_out = rearrange(z_q, "b d t -> b t d")
            z_q_out = self.out_proj(z_q_out)
            z_q_out = rearrange(z_q_out, "b t d -> b d t")

        return z_q_out.to(orig_dtype), commit_loss, codes  # (B,D,T), (B,), (B,T)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, T) integer indices
        Returns:
            z_q: (B, D, T) — projected back to input_dim
        """
        z_q_proj = F.embedding(codes, self.codebook.float())  # (B, T, D')
        z_q_proj = rearrange(z_q_proj, "b t d -> b d t")  # (B, D', T)
        if isinstance(self.out_proj, nn.Identity):
            return z_q_proj
        z_q = rearrange(z_q_proj, "b d t -> b t d")
        z_q = self.out_proj(z_q)
        return rearrange(z_q, "b t d -> b d t")


def _rotation_trick_ste(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """
    Rotation-trick STE (Fifty et al., arXiv:2410.06424).

    Instead of the vanilla copy-gradient trick ``(z_q - z_e).detach() + z_e``,
    this rotates ``z_e`` in the direction of ``z_q`` without changing norms,
    giving a better-conditioned gradient signal.

    Falls back to vanilla STE when norms are near zero.
    """
    # Work in fp32
    ze = z_e.float()
    zq = z_q.float()

    # Per-sample, per-position norms: (B, 1, T)
    norm_e = ze.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_q = zq.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Normalise
    ze_hat = ze / norm_e
    zq_hat = zq / norm_q

    # Slerp-like rotation: scale z_e to match z_q's norm, then STE
    z_rot = ze_hat * norm_q  # same direction as z_e, same norm as z_q
    return (zq - z_rot).detach() + z_rot


# ─────────────────────────────────────────────────────────────────────────────
# Residual Vector Quantizer
# ─────────────────────────────────────────────────────────────────────────────


class ResidualVectorQuantize(nn.Module):
    """
    Residual Vector Quantization (RVQ) with quantizer dropout.

    At each of the ``n_codebooks`` levels, the *residual* from the previous
    level is quantized.  During training, a fraction ``quantizer_dropout`` of
    samples randomly uses only 1..n_codebooks levels (forcing each level to
    be independently useful).

    Input/output: (B, D, T) channels-first.
    """

    def __init__(
        self,
        input_dim: int,
        n_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float = 0.25,
        decay: float = 0.95,
        epsilon: float = 1e-5,
        threshold_ema_dead: float = 2.0,
        quantizer_dropout: float = 0.5,
        use_rotation_trick: bool = True,
        use_fsq: bool = False,
        fsq_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.quantizer_dropout = quantizer_dropout

        if use_fsq:
            _levels = fsq_levels if fsq_levels is not None else [4, 4, 4, 4, 4, 4]
            self.quantizers = nn.ModuleList(
                [FSQQuantize(input_dim=input_dim, levels=_levels) for _ in range(n_codebooks)]
            )
        else:
            self.quantizers = nn.ModuleList(
                [
                    EMAVectorQuantize(
                        input_dim=input_dim,
                        codebook_size=codebook_size,
                        codebook_dim=codebook_dim,
                        commitment=commitment,
                        decay=decay,
                        epsilon=epsilon,
                        threshold_ema_dead=threshold_ema_dead,
                        use_rotation_trick=use_rotation_trick,
                    )
                    for _ in range(n_codebooks)
                ]
            )

    def forward(
        self, z: torch.Tensor, n_quantizers: Optional[int] = None, return_level_data: bool = False
    ):
        """
        Args:
            z:                (B, D, T)
            n_quantizers:     number of codebooks to use at inference (None = all)
            return_level_data: if True, also return per-level consistency residuals
                              and codes for computing the token consistency loss.
        Returns (default):
            z_q:           (B, D, T) sum of all quantised residuals
            codes:         (B, n_codebooks, T)
            commit_loss:   scalar
        Returns (return_level_data=True), additionally:
            consist_residuals: List[K] of (B, codebook_dim, T) — pre-quantization
                residual at each level, projected to codebook space via in_proj.
                Loss computed in codebook space so gradients align with the actual
                VQ decision boundary; gradient flows through in_proj back to encoder.
            level_codes:       List[K] of (B, T) — integer codes per level.
        """
        B = z.shape[0]

        if self.training:
            # Quantizer dropout: each sample independently uses a random number
            # of quantizers from 1 to n_codebooks.
            n_q_per_sample = torch.ones(B, device=z.device) * (self.n_codebooks + 1)
            dropout_mask = torch.rand(B, device=z.device) < self.quantizer_dropout
            rand_n = torch.randint(1, self.n_codebooks + 1, (B,), device=z.device)
            n_q_per_sample[dropout_mask] = rand_n[dropout_mask].float()
        else:
            n_q = n_quantizers if n_quantizers is not None else self.n_codebooks
            n_q_per_sample = torch.full((B,), n_q + 0.5, device=z.device)

        z_q = torch.zeros_like(z)
        residual = z
        all_codes = []
        total_commit_loss = torch.tensor(0.0, device=z.device)

        # Separate residual chain for consistency loss:
        # Updated with detached quantized vectors so that the consistency loss
        # gradient flows to the encoder via this chain, not through quantizer STEs.
        if return_level_data:
            consist_residual = z
            consist_residuals: List[torch.Tensor] = []
            level_codes_list: List[torch.Tensor] = []

        for i, quantizer in enumerate(self.quantizers):
            # Mask: which samples should use level i?
            active = (i < n_q_per_sample).float()  # (B,)

            if return_level_data:
                # Project to codebook space (D → codebook_dim) before storing.
                # VQ decisions happen in codebook_dim space; computing the
                # consistency loss there ensures gradients are aligned with
                # what actually determines code assignments.
                # Gradient flows: loss → in_proj → consist_residual → encoder.
                if isinstance(quantizer.in_proj, nn.Identity):
                    consist_residuals.append(consist_residual)
                else:
                    cr = rearrange(consist_residual, "b d t -> b t d")
                    cr = quantizer.in_proj(cr)
                    consist_residuals.append(rearrange(cr, "b t d -> b d t"))

            z_q_i, commit_i, codes_i = quantizer(residual)

            if return_level_data:
                level_codes_list.append(codes_i)
                # Detached update: gradient of consistency loss at level k flows
                # through consist_residual[k] directly to z, not through z_q_i's STE.
                consist_residual = consist_residual - z_q_i.detach()

            # Apply mask per sample
            z_q = z_q + z_q_i * active[:, None, None]
            residual = residual - z_q_i

            total_commit_loss = total_commit_loss + (commit_i * active).mean()
            all_codes.append(codes_i)  # (B, T)

        codes = torch.stack(all_codes, dim=1)  # (B, n_codebooks, T)

        if return_level_data:
            return z_q, codes, total_commit_loss, consist_residuals, level_codes_list
        return z_q, codes, total_commit_loss

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct quantised representation from integer codes.

        Args:
            codes: (B, n_codebooks, T)
        Returns:
            z_q:   (B, D, T)
        """
        z_q = torch.zeros(
            codes.shape[0],
            self.quantizers[0].input_dim,
            codes.shape[2],
            device=codes.device,
            dtype=torch.float32,
        )
        for i, quantizer in enumerate(self.quantizers[: codes.shape[1]]):
            z_q = z_q + quantizer.decode_codes(codes[:, i, :])
        return z_q
