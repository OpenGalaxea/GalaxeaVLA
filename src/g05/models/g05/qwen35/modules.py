"""
Qwen3.5 basic modules: MRoPE, RMSNorm, RMSNormGated.

Ported from HF transformers/models/qwen3_5/modeling_qwen3_5.py.
Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Supports partial_rotary_factor: when cos/sin have fewer dims than q/k,
    only the first `rotary_dim` dimensions are rotated, and the rest pass through.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Qwen3.5 Text RoPE — Multimodal RoPE (MRoPE)
# ---------------------------------------------------------------------------
class Qwen3_5TextRotaryEmbedding(nn.Module):
    """MRoPE: 3D interleaved rotary embedding for text/vision tokens.

    position_ids shape: (3, batch, seq_len) for temporal/height/width.
    Supports partial_rotary_factor (default 0.25 for Qwen3.5).
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]

        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: tensor for dtype/device reference
            position_ids: (3, batch, seq_len) or (batch, seq_len)
        Returns:
            cos, sin: (batch, seq_len, rotary_dim)
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, seq_len)

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE: [TTT...HHH...WWW] -> [THWTHWTHW...TT]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


# ---------------------------------------------------------------------------
# Qwen3.5 RMSNorm — (1 + weight) style, zeros init
# ---------------------------------------------------------------------------


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling. Weight initialized to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Qwen3.5 RMSNormGated — used inside GatedDeltaNet
# ---------------------------------------------------------------------------


class Qwen3_5RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating, used in GatedDeltaNet."""

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.autocast("cuda", enabled=False)
    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
