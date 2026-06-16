"""
Qwen3.5 GatedDeltaNet — linear attention with gated delta rule.

Ported from HF transformers/models/qwen3_5/modeling_qwen3_5.py.
Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
Pure PyTorch fallback implementation (no causal_conv1d / FLA dependency).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Qwen3_5RMSNormGated

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Zero out hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Pure PyTorch causal Conv1d single-step update (decode phase)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Chunk-wise Gated Delta Rule (training / prefill)
# ---------------------------------------------------------------------------


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# Recurrent Gated Delta Rule (decode / single-step)
# ---------------------------------------------------------------------------


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# Try to import fast implementations
# ---------------------------------------------------------------------------

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    _has_causal_conv1d = True
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    _has_causal_conv1d = False

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as fla_recurrent_gated_delta_rule,
    )

    _has_fla = True
except ImportError:
    FusedRMSNormGated = None
    fla_chunk_gated_delta_rule, fla_recurrent_gated_delta_rule = None, None
    _has_fla = False

try:
    from flash_qla import chunk_gated_delta_rule as _flashqla_chunk_gdr

    def flashqla_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        chunk_size=None,
        **kwargs,
    ):
        # FlashQLA backward always returns dh0 gradient; provide a zero initial_state
        # as a requires_grad tensor to avoid "not a Variable" error
        if initial_state is None:
            B, _, H_v, V = v.shape
            K = q.shape[-1]
            initial_state = torch.zeros(B, H_v, K, V, dtype=torch.float32, device=q.device)
        # FlashQLA TileLang kernels are incompatible with torch.amp.autocast
        with torch.amp.autocast("cuda", enabled=False):
            return _flashqla_chunk_gdr(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                beta.contiguous(),
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

    _has_flashqla = True
except ImportError:
    flashqla_chunk_gated_delta_rule = None
    _has_flashqla = False

_is_fast_path_available = _has_fla


# ---------------------------------------------------------------------------
# GatedDeltaNet
# ---------------------------------------------------------------------------


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated Delta Net linear attention layer.

    Uses fast path (causal_conv1d + FLA kernels) when available,
    falls back to pure PyTorch otherwise.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Use FusedRMSNormGated when available, else fallback
        if FusedRMSNormGated is not None:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=config.rms_norm_eps,
                activation=config.hidden_act,
                device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
                dtype=torch.get_default_dtype(),
            )
        else:
            self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Select fast or fallback implementations, FIXME: Ubuntu22 cannot use causal_conv1d, need to investigate further
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update

        # linear_attn_backend: "flashqla" | "fla" | "torch" (default: "fla")
        backend = getattr(config, "linear_attn_backend", "fla")
        if backend == "flashqla":
            if not _has_flashqla:
                raise ImportError("linear_attn_backend='flashqla' but flash_qla is not installed")
            self.chunk_gated_delta_rule = flashqla_chunk_gated_delta_rule
            # FlashQLA only supports chunk prefill; decode falls back to FLA or torch
            self.recurrent_gated_delta_rule = (
                fla_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
            )
            logging.info(f"Layer {layer_idx}: using FlashQLA for chunk_gated_delta_rule")
        elif backend == "fla":
            self.chunk_gated_delta_rule = fla_chunk_gated_delta_rule or torch_chunk_gated_delta_rule
            self.recurrent_gated_delta_rule = (
                fla_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
            )
        else:
            self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

        if not _is_fast_path_available and backend != "torch":
            logging.warning(
                f"causal_conv1d available: {_has_causal_conv1d}, FLA available: {_has_fla}"
            )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        split_idx=None,
        return_kv_cache: bool = True,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and hasattr(cache_params, "has_previous_state")
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None and return_kv_cache:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        with torch.autocast(device_type="cuda", enabled=False):
            A = -torch.exp(self.A_log.float())
            dt = F.softplus(a.float() + self.dt_bias.float())
            g = A * dt
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            init_state = recurrent_state if cache_params is not None else None
            # Full-sequence chunk: VLM output is UNCHANGED regardless of split_idx
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=init_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                chunk_size=32,  # fastest chunk size among tested values (16, 32, 64, 128)
            )
            # Extra prefix-only chunk: extract clean recurrent_state at split_idx
            # (without AR action token information leakage)
            if (
                split_idx is not None
                and 0 < split_idx < seq_len
                and cache_params is not None
                and return_kv_cache
            ):
                _, split_recurrent_state = self.chunk_gated_delta_rule(
                    query[:, :split_idx],
                    key[:, :split_idx],
                    value[:, :split_idx],
                    g=g[:, :split_idx],
                    beta=beta[:, :split_idx],
                    initial_state=init_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    chunk_size=32,
                )
                cache_params.split_recurrent_states[self.layer_idx] = split_recurrent_state
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None and return_kv_cache:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        # FusedRMSNormGated (FLA fast path) outputs dtype == input dtype, so pass fp32
        # explicitly and disable autocast to guarantee fp32 norm regardless of which
        # implementation (FLA Triton kernel vs. fallback Qwen3_5RMSNormGated) is active.
        with torch.autocast("cuda", enabled=False):
            core_attn_out = self.norm(core_attn_out.float(), z.float())
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output
