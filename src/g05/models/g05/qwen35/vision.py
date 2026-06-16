"""
Qwen3.5 Vision Model — complete ViT with Conv3D patch embedding, 2D RoPE,
learnable absolute position embedding, and PatchMerger projector.

Ported from HF transformers/models/qwen3_5/modeling_qwen3_5.py.
Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
Aligned with g05 interface: provides load_pretrained_weights(hf_config, tensors).
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils

from .modules import rotate_half

_flash_attn_varlen = None
_flash_attn_backend = None

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen

    _flash_attn_varlen = _fa4_varlen
    _flash_attn_backend = "fa4"
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen

        _flash_attn_varlen = _fa2_varlen
        _flash_attn_backend = "fa2"
    except ImportError:
        pass

_VISION_FLASH_ATTN_WARNED = False

logger = logging.getLogger(__name__)


def _sinusoidal_temporal_pe(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal temporal PE with sin/cos pairing and e(0) = 0.

    e(t)_{2i}   = sin(t · ω_i)
    e(t)_{2i+1} = cos(t · ω_i) − 1
    where ω_i = 1/10000^(i / (dim//2))

    The −1 shift on cos ensures e(0) = 0 for initialization compatibility
    with pretrained image weights. Returns [K, D].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [K, half]
    pe = torch.stack([torch.sin(args), torch.cos(args) - 1], dim=2)  # [K, half, 2]
    return pe.reshape(timesteps.shape[0], dim)  # [K, D]


# ---------------------------------------------------------------------------
# Vision-specific helpers
# ---------------------------------------------------------------------------


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple:
    """Apply RoPE to vision Q/K. cos/sin are (seq_len, dim)."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


# ---------------------------------------------------------------------------
# Vision RoPE
# ---------------------------------------------------------------------------


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """2D rotary embedding for vision tokens."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


# ---------------------------------------------------------------------------
# Vision Patch Embedding (Conv3D)
# ---------------------------------------------------------------------------


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Patch Merger (spatial merge + MLP projector)
# ---------------------------------------------------------------------------


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(
            -1, self.hidden_size
        )
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden_states → Q, K, V. Returns each [seq, H, hd]."""
        seq_len = hidden_states.shape[0]
        return (
            self.qkv(hidden_states)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

    def _attend_spatial(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        global _VISION_FLASH_ATTN_WARNED

        seq_length = query_states.shape[0]

        if _flash_attn_varlen is not None:
            cu_seqlens_cuda = cu_seqlens.cuda() if not cu_seqlens.is_cuda else cu_seqlens
            max_seqlen = (cu_seqlens_cuda[1:] - cu_seqlens_cuda[:-1]).max().item()
            if _flash_attn_backend == "fa4":
                attn_output, _ = _flash_attn_varlen(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_cuda,
                    cu_seqlens_k=cu_seqlens_cuda,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    softmax_scale=self.scaling,
                    causal=False,
                )
            else:
                attn_output = _flash_attn_varlen(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_cuda,
                    cu_seqlens_cuda,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=0.0,
                    causal=False,
                )
            return attn_output.reshape(seq_length, -1)
        else:
            if not _VISION_FLASH_ATTN_WARNED:
                logger.warning(
                    "flash_attn not installed — vision attention uses SDPA fallback. "
                    "For Hopper/Blackwell GPUs (H100/B200), install FA4: "
                    "pip install flash-attn-4. For Ampere/Ada (A100/RTX4090), "
                    "install FA2: pip install flash-attn --no-build-isolation"
                )
                _VISION_FLASH_ATTN_WARNED = True

            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            query_states_b = query_states.transpose(0, 1).unsqueeze(0)
            key_states_b = key_states.transpose(0, 1).unsqueeze(0)
            value_states_b = value_states.transpose(0, 1).unsqueeze(0)
            attn_output = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=self.scaling,
                    )
                    for q, k, v in zip(
                        torch.split(query_states_b, lengths, dim=2),
                        torch.split(key_states_b, lengths, dim=2),
                        torch.split(value_states_b, lengths, dim=2),
                    )
                ],
                dim=2,
            )
            return attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple,
        **kwargs,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(hidden_states)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        return self.proj(self._attend_spatial(q, k, v, cu_seqlens))


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple,
        **kwargs,
    ) -> torch.Tensor:
        """Spatial-only forward (standard ViT block)."""
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    # ------------------------------------------------------------------
    # MEM: Space-Time separable attention
    #
    # "factorized": QKV once, temporal mixes V→V', spatial uses original
    #   Q,K + V', out_proj once. α^space × α^time are independent.
    #
    # "cascaded" (TimeSformer-style): temporal produces V', spatial
    #   re-projects QKV from V'. More expressive but not factorized.
    # ------------------------------------------------------------------

    def forward_spacetime(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple,
        num_frames: int,
        bsz: int,
        causal_mask: torch.Tensor,
        temporal_pe: torch.Tensor,
        mode: str = "factorized",
        varlen_drop: bool = False,
    ) -> torch.Tensor:
        if mode == "factorized":
            return self._spacetime_factorized(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                num_frames,
                bsz,
                causal_mask,
                temporal_pe,
                varlen_drop,
            )
        elif mode == "cascaded":
            return self._spacetime_cascaded(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                num_frames,
                bsz,
                causal_mask,
                temporal_pe,
            )
        raise ValueError(f"Unknown spacetime_mode: {mode!r}")

    def _spacetime_factorized(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple,
        num_frames: int,
        bsz: int,
        causal_mask: torch.Tensor,
        temporal_pe: torch.Tensor,
        varlen_drop: bool = False,
    ) -> torch.Tensor:
        """Factorized T+S: one QKV, temporal mixes V→V', spatial uses original Q,K + V'."""
        total, D = hidden_states.shape
        H, hd = self.attn.num_heads, self.attn.head_dim
        residual = hidden_states

        if not varlen_drop:
            N_p = total // (bsz * num_frames)
            hs = (
                hidden_states.view(bsz, num_frames, N_p, D) + temporal_pe.unsqueeze(0).unsqueeze(2)
            ).reshape(total, D)
        else:
            n_images = cu_seqlens.shape[0] - 1
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            frame_indices = torch.arange(n_images, device=hidden_states.device) % num_frames
            pe_per_token = torch.repeat_interleave(temporal_pe[frame_indices], lengths, dim=0)
            hs = hidden_states + pe_per_token

        normed = self.norm1(hs)
        q, k, v = self.attn._project_qkv(normed)  # each [total, H, hd]

        # --- Temporal: per-patch across K frames, causal, NO out_proj ---
        if not varlen_drop:
            N_p = total // (bsz * num_frames)
            q_t = (
                q.view(bsz, num_frames, N_p, H, hd)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * N_p, H, num_frames, hd)
            )
            k_t = (
                k.view(bsz, num_frames, N_p, H, hd)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * N_p, H, num_frames, hd)
            )
            v_t = (
                v.view(bsz, num_frames, N_p, H, hd)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * N_p, H, num_frames, hd)
            )

            attn_w = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.attn.scaling
            attn_w = attn_w + causal_mask.unsqueeze(0).unsqueeze(0)
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q_t.dtype)
            v_prime = torch.matmul(attn_w, v_t)

            v_prime = (
                v_prime.view(bsz, N_p, H, num_frames, hd)
                .permute(0, 3, 1, 2, 4)
                .reshape(total, H, hd)
            )
        else:
            # Varlen temporal: group by patch count, vectorized gather/scatter
            n_images = cu_seqlens.shape[0] - 1
            n_videos = n_images // num_frames
            lengths_cpu = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

            v_prime = torch.zeros_like(q)

            from collections import defaultdict

            groups = defaultdict(list)
            for vid_idx in range(n_videos):
                np_vid = lengths_cpu[vid_idx * num_frames]
                groups[np_vid].append(vid_idx)

            for np_val, vid_indices in groups.items():
                g_bsz = len(vid_indices)
                img_indices = torch.tensor(
                    [
                        vid_idx * num_frames + f
                        for vid_idx in vid_indices
                        for f in range(num_frames)
                    ],
                    device=cu_seqlens.device,
                    dtype=torch.long,
                )
                starts = cu_seqlens[img_indices]
                offsets = torch.arange(np_val, device=q.device, dtype=torch.long)
                flat_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)

                g_q = q[flat_idx].view(g_bsz, num_frames, np_val, H, hd)
                g_k = k[flat_idx].view(g_bsz, num_frames, np_val, H, hd)
                g_v = v[flat_idx].view(g_bsz, num_frames, np_val, H, hd)

                g_q_t = g_q.permute(0, 2, 3, 1, 4).reshape(g_bsz * np_val, H, num_frames, hd)
                g_k_t = g_k.permute(0, 2, 3, 1, 4).reshape(g_bsz * np_val, H, num_frames, hd)
                g_v_t = g_v.permute(0, 2, 3, 1, 4).reshape(g_bsz * np_val, H, num_frames, hd)

                aw = torch.matmul(g_q_t, g_k_t.transpose(-2, -1)) * self.attn.scaling
                aw = aw + causal_mask.unsqueeze(0).unsqueeze(0)
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(g_q_t.dtype)
                g_vp = torch.matmul(aw, g_v_t)
                g_vp = g_vp.view(g_bsz, np_val, H, num_frames, hd).permute(0, 3, 1, 2, 4)

                v_prime[flat_idx] = g_vp.reshape(-1, H, hd)

        # --- Spatial: original Q,K + 2D RoPE, V' as values, out_proj once ---
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb_vision(q, k, cos, sin)
        spatial_out = self.attn.proj(self.attn._attend_spatial(q_rope, k_rope, v_prime, cu_seqlens))

        hidden_states = residual + spatial_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def _spacetime_cascaded(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple,
        num_frames: int,
        bsz: int,
        causal_mask: torch.Tensor,
        temporal_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Cascaded T+S: temporal → V', spatial re-projects QKV from V'."""
        total, D = hidden_states.shape
        N_p = total // (bsz * num_frames)
        H, hd = self.attn.num_heads, self.attn.head_dim
        residual = hidden_states

        hs = (
            hidden_states.view(bsz, num_frames, N_p, D) + temporal_pe.unsqueeze(0).unsqueeze(2)
        ).reshape(total, D)
        normed = self.norm1(hs)
        q, k, v = self.attn._project_qkv(normed)

        # --- Temporal: causal attention, NO out_proj ---
        q_t = (
            q.view(bsz, num_frames, N_p, H, hd)
            .permute(0, 2, 3, 1, 4)
            .reshape(bsz * N_p, H, num_frames, hd)
        )
        k_t = (
            k.view(bsz, num_frames, N_p, H, hd)
            .permute(0, 2, 3, 1, 4)
            .reshape(bsz * N_p, H, num_frames, hd)
        )
        v_t = (
            v.view(bsz, num_frames, N_p, H, hd)
            .permute(0, 2, 3, 1, 4)
            .reshape(bsz * N_p, H, num_frames, hd)
        )

        attn_w = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.attn.scaling
        attn_w = attn_w + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q_t.dtype)
        temporal_out = torch.matmul(attn_w, v_t)  # [bsz*N_p, H, K, hd]

        temporal_out = (
            temporal_out.view(bsz, N_p, H, num_frames, hd).permute(0, 3, 1, 2, 4).reshape(total, D)
        )

        # --- Spatial: full attn re-projects QKV from temporal_out ---
        spatial_out = self.attn(
            temporal_out,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + spatial_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------


class Qwen3_5VisionModel(nn.Module):
    """Complete Qwen3.5 Vision encoder: PatchEmbed -> Blocks -> Merger.

    Returns (last_hidden_state, pooler_output) where pooler_output is the
    merged representation that feeds into the language model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.temporal_freq = getattr(config, "temporal_freq", 0)
        self.spacetime_mode = getattr(config, "spacetime_mode", "factorized")
        self.token_drop_layer = getattr(config, "token_drop_layer", None) or config.depth
        self.temporal_attn_interval = self.temporal_freq or 4
        self._varlen_drop = getattr(config, "batch_all_cameras", False)
        self._temporal_pe_pretrain_frames = getattr(config, "temporal_pe_pretrain_frames", None)

        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])

        if self.temporal_freq > 0:
            drop_idx = self.token_drop_layer - 1
            ts_layers = [
                i
                for i in range(config.depth)
                if i <= drop_idx and (drop_idx - i) % self.temporal_attn_interval == 0
            ]
            assert self.spacetime_mode in ("factorized", "cascaded"), (
                f"spacetime_mode={self.spacetime_mode!r}"
            )
            assert 1 <= self.token_drop_layer <= config.depth, (
                f"token_drop_layer={self.token_drop_layer} out of [1, {config.depth}]"
            )
            logger.info(
                "MEM video encoder: mode=%s, T+S every %d layers (drop at %d → T+S at %s)",
                self.spacetime_mode,
                self.temporal_attn_interval,
                self.token_drop_layer,
                ts_layers,
            )
        self.merger = Qwen3_5VisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        num_frames: int = 1,
        bsz: int = 1,
    ):
        """
        Args:
            hidden_states: pixel values tensor
            grid_thw: (num_images, 3) — temporal, height, width grid sizes
            num_frames: K frames per camera; >1 enables MEM temporal attention
            bsz: batch size
        Returns:
            (last_hidden_state, pooler_output)
        """
        with torch.autocast(hidden_states.device.type, enabled=False):
            hidden_states = self.patch_embed(hidden_states)
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        if num_frames <= 1 or self.temporal_freq <= 0:
            # Single-frame: original path, zero overhead
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    hidden_states = checkpoint_utils.checkpoint(
                        blk,
                        hidden_states,
                        use_reentrant=False,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                    )
                else:
                    hidden_states = blk(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                    )
        else:
            # === Multi-frame MEM path ===
            D = hidden_states.shape[1]
            N_p = hidden_states.shape[0] // (bsz * num_frames) if not self._varlen_drop else 0

            if not hasattr(self, "_mem_forward_logged"):
                interp_info = ""
                if (
                    self._temporal_pe_pretrain_frames is not None
                    and num_frames > self._temporal_pe_pretrain_frames
                ):
                    scale = (self._temporal_pe_pretrain_frames - 1) / (num_frames - 1)
                    interp_info = (
                        f", PE interpolation: [{-(num_frames - 1)},0] → "
                        f"[{-(self._temporal_pe_pretrain_frames - 1):.1f},0] (scale={scale:.3f})"
                    )
                logger.info(
                    f"MEM forward: num_frames={num_frames}, bsz={bsz}, "
                    f"pretrain_frames={self._temporal_pe_pretrain_frames}{interp_info}"
                )
                self._mem_forward_logged = True

            timesteps = torch.arange(-(num_frames - 1), 1, device=hidden_states.device).float()
            if self._temporal_pe_pretrain_frames is not None:
                pretrain_range = self._temporal_pe_pretrain_frames - 1
                current_range = num_frames - 1
                if current_range > pretrain_range:
                    timesteps = timesteps * (pretrain_range / current_range)
            temporal_pe = _sinusoidal_temporal_pe(timesteps, D).to(hidden_states.dtype)  # [K, D]
            causal_mask = torch.triu(
                torch.full(
                    (num_frames, num_frames),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )

            drop_idx = self.token_drop_layer - 1
            for i, blk in enumerate(self.blocks):
                use_temporal = (
                    num_frames > 1
                    and i <= drop_idx
                    and (drop_idx - i) % self.temporal_attn_interval == 0
                )

                if use_temporal:
                    if self.gradient_checkpointing and self.training:
                        hidden_states = checkpoint_utils.checkpoint(
                            blk.forward_spacetime,
                            hidden_states,
                            cu_seqlens,
                            position_embeddings,
                            num_frames,
                            bsz,
                            causal_mask,
                            temporal_pe,
                            self.spacetime_mode,
                            self._varlen_drop,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states = blk.forward_spacetime(
                            hidden_states,
                            cu_seqlens,
                            position_embeddings,
                            num_frames,
                            bsz,
                            causal_mask,
                            temporal_pe,
                            self.spacetime_mode,
                            self._varlen_drop,
                        )
                else:
                    if self.gradient_checkpointing and self.training:
                        hidden_states = checkpoint_utils.checkpoint(
                            blk,
                            hidden_states,
                            use_reentrant=False,
                            cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        hidden_states = blk(
                            hidden_states,
                            cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings,
                        )

                # Token drop: keep only last frame (t=0)
                if i == drop_idx and num_frames > 1:
                    if self._varlen_drop:
                        n_images = cu_seqlens.shape[0] - 1
                        keep_indices = torch.arange(
                            num_frames - 1, n_images, num_frames, device=hidden_states.device
                        )
                        cos, sin = position_embeddings
                        kept_h_parts, kept_cos_parts, kept_sin_parts, new_lengths = [], [], [], []
                        for idx in keep_indices:
                            start = cu_seqlens[idx].item()
                            end = cu_seqlens[idx + 1].item()
                            kept_h_parts.append(hidden_states[start:end])
                            kept_cos_parts.append(cos[start:end])
                            kept_sin_parts.append(sin[start:end])
                            new_lengths.append(end - start)
                        hidden_states = torch.cat(kept_h_parts, dim=0)
                        position_embeddings = (
                            torch.cat(kept_cos_parts, dim=0),
                            torch.cat(kept_sin_parts, dim=0),
                        )
                        num_frames = 1
                        cu_seqlens = torch.zeros(
                            len(new_lengths) + 1, dtype=torch.int32, device=hidden_states.device
                        )
                        cu_seqlens[1:] = torch.tensor(
                            new_lengths, dtype=torch.int32, device=hidden_states.device
                        ).cumsum(0)
                    else:
                        hidden_states = hidden_states.view(bsz, num_frames, N_p, D)[:, -1].reshape(
                            bsz * N_p, D
                        )
                        num_frames = 1
                        cu_seqlens = torch.arange(
                            0,
                            (bsz + 1) * N_p,
                            N_p,
                            dtype=torch.int32,
                            device=hidden_states.device,
                        )
                        cos, sin = position_embeddings
                        position_embeddings = (cos[: bsz * N_p], sin[: bsz * N_p])

        with torch.autocast(hidden_states.device.type, enabled=False):
            merged_hidden_states = self.merger(hidden_states)

        return hidden_states, merged_hidden_states

    # ---------------------------------------------------------------------------
    # g05 interface: load pretrained weights
    # ---------------------------------------------------------------------------

    def load_pretrained_weights(self, hf_config, tensors: dict):
        """Load Qwen3.5 vision weights from HF safetensors.

        Key mapping: model.visual.* → *
        """
        state_dict = self.state_dict()
        for k, v in tensors.items():
            if "model.visual." in k:
                new_key = k.replace("model.visual.", "")
                if new_key in state_dict:
                    state_dict[new_key] = v
        self.load_state_dict(state_dict, strict=True)
        logger.info("Loaded Qwen3.5 vision weights from pretrained")
