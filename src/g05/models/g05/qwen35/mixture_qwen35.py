"""
MixtureQwen35 — Complete Qwen3.5 LLM module for g05 architecture.

VLM and Action Expert are two MixtureQwen35 instances, config-driven:
  VLM:            input_proj=Embedding, output_proj=lm_head(tied), no time_cond
  Action Expert:  input_proj=Linear,    output_proj=Linear,        time_cond=adaLN

Key differences from the PaliGemma Mixture:
  - Hybrid attention: full_attention (every 4th layer) + linear_attention (GatedDeltaNet)
  - Gated attention output: q_proj → (query, gate), attn * sigmoid(gate)
  - QK norm (Qwen3_5RMSNorm on Q, K heads)
  - MRoPE with partial_rotary_factor=0.25
  - SwiGLU MLP (SiLU activation)
  - No sqrt(hidden_size) embedding scaling
  - SparseKVCache for hybrid attention layers
"""

import glob
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from g05.utils.training.train_utils import get_global_monitor

from .modules import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
    rotate_half,
)
from .gated_deltanet import Qwen3_5GatedDeltaNet
from ..model.utils import repeat_kv, AttentionModuleProxy, eager_attention_forward
from ..model.modules import AdaptiveRMSNorm, SinusoidalPosEmbPi0

logger = logging.getLogger(__name__)


def _load_all_safetensors(path: str) -> dict:
    """Load all safetensor files from a directory into a single dict."""
    from safetensors import safe_open

    safetensors_files = glob.glob(os.path.join(path, "*.safetensors"))
    assert len(safetensors_files) > 0, f"No safetensors found in {path}"
    tensors = {}
    for f_path in safetensors_files:
        with safe_open(f_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


# ---------------------------------------------------------------------------
# Qwen3.5 SwiGLU MLP
# ---------------------------------------------------------------------------


class Qwen3_5MLP(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MixtureAttentionQwen35 — full_attention layers with gated output
# ---------------------------------------------------------------------------


class MixtureAttentionQwen35(nn.Module):
    """Qwen3.5 full attention with:
    - q_proj outputs 2x dim (query + gate)
    - QK norm (Qwen3_5RMSNorm on head_dim)
    - partial_rotary_factor=0.25 via MRoPE
    - Gated output: attn_output * sigmoid(gate)
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert config.hidden_size % self.num_heads == 0

        # q_proj outputs 2x for query + gate
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # QK Norm
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # MRoPE
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)


# ---------------------------------------------------------------------------
# MixtureDecoderLayerQwen35
# ---------------------------------------------------------------------------


class MixtureDecoderLayerQwen35(nn.Module):
    """Qwen3.5 decoder layer with 2 norms (standard pre-norm).

    For full_attention layers: uses MixtureAttentionQwen35
    For linear_attention layers: uses Qwen3_5GatedDeltaNet
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.num_hidden_layers = config.num_hidden_layers

        self.adaptive_mode = getattr(config, "adaptive_mode", None)

        if self.layer_type == "full_attention":
            self.self_attn = MixtureAttentionQwen35(config, layer_idx)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)

        self.mlp = Qwen3_5MLP(config)

        # 2-norm pre-norm architecture
        if self.adaptive_mode == "adaLN":
            self.input_layernorm = AdaptiveRMSNorm(
                config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = AdaptiveRMSNorm(
                config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3_5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values=None,
        time_cond: Optional[torch.FloatTensor] = None,
        return_kv: bool = False,
        layer_idx: int = -1,
        mixture_name: Optional[str] = None,
        attn_implementation: str = "eager",
        kv_cache=None,
        split_idx=None,
        return_kv_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Single-mixture layer forward.

        For full_attention: standard attention with gated output
        For linear_attention: GatedDeltaNet independent forward
        """
        attn = getattr(self, "self_attn", None)
        bsz, seq_len = hidden_states.shape[:2]

        # Pre-attention norm
        residual = hidden_states
        if self.adaptive_mode == "adaLN":
            hidden_states, gate = self.input_layernorm(hidden_states, time_cond)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        new_kv = None

        if self.layer_type == "full_attention":
            head_dim = attn.head_dim
            num_heads = attn.num_heads
            num_kv_heads = attn.num_key_value_heads
            num_kv_groups = attn.num_key_value_groups
            scaling = head_dim**-0.5

            # Q (+ gate), K, V projections with QK Norm
            qg = attn.q_proj(hidden_states)
            query_states, attn_gate = torch.chunk(
                qg.view(bsz, seq_len, -1, head_dim * 2), 2, dim=-1
            )
            attn_gate = attn_gate.reshape(bsz, seq_len, -1)  # [B, S, num_heads * head_dim]
            query_states = attn.q_norm(query_states).transpose(1, 2)  # [B, H, S, D]

            key_states = attn.k_proj(hidden_states)
            key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim)
            key_states = attn.k_norm(key_states).transpose(1, 2)

            value_states = attn.v_proj(hidden_states)
            value_states = value_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply partial RoPE in fp32 to match the reference joint-model precision.
            with torch.autocast("cuda", enabled=False):
                orig_qk_dtype = query_states.dtype
                q_fp32 = query_states.float()
                k_fp32 = key_states.float()

                _cos = cos.float()
                _sin = sin.float()
                q_seq_len = q_fp32.shape[-2]
                if _cos.shape[-2] != q_seq_len:
                    _cos = _cos[:, :, :q_seq_len, :]
                    _sin = _sin[:, :, :q_seq_len, :]

                rotary_dim = _cos.shape[-1]
                q_rot, q_pass = q_fp32[..., :rotary_dim], q_fp32[..., rotary_dim:]
                k_rot, k_pass = k_fp32[..., :rotary_dim], k_fp32[..., rotary_dim:]
                q_rotated = q_rot * _cos + rotate_half(q_rot) * _sin
                k_rotated = k_rot * _cos + rotate_half(k_rot) * _sin
                query_states = torch.cat([q_rotated, q_pass], dim=-1).to(orig_qk_dtype)
                key_states = torch.cat([k_rotated, k_pass], dim=-1).to(orig_qk_dtype)

            # KV cache: use SparseKVCache.update() for in-place append.
            # This avoids bf16 precision drift vs the list-based full-concat approach.
            key_states_new = key_states
            value_states_new = value_states

            if kv_cache is not None:
                # Read cached KV (if any) from SparseKVCache
                if kv_cache.has_item(layer_idx):
                    key_states_cached, value_states_cached = kv_cache.get(layer_idx)
                else:
                    key_states_cached = None

                # Store new KV into SparseKVCache (in-place concat)
                kv_cache.update(key_states_new, value_states_new, layer_idx)

                # Assemble old cached KV plus new KV.
                if key_states_cached is not None:
                    key_states = torch.cat((key_states_cached, key_states_new), dim=-2)
                    value_states = torch.cat((value_states_cached, value_states_new), dim=-2)
                # else: first time (prefill), key_states = key_states_new already

                # Return the accumulated KV from SparseKVCache for kv_cache list
                # (used by FM training for prefix slicing via vlm_kv[layer_idx])
                new_kv = kv_cache.get(layer_idx)
            elif past_key_values is not None:
                # Fallback: list-based KV cache (PaliGemma compatibility)
                past_k, past_v = past_key_values
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
                new_kv = (key_states, value_states)
            else:
                new_kv = (key_states, value_states)

            # GQA repeat
            key_states = repeat_kv(key_states, num_kv_groups)
            value_states = repeat_kv(value_states, num_kv_groups)

            # AE full_attn layers without VLM prefix KV (cross_attn_only + layer_idx not
            # in VLM full_attn set) have key_len = action_len, but the shared attention_mask
            # has shape [B, 1, H, S_prefix + H] with layout [_prefix_mask | action_self_mask].
            # eager_attention_forward slices the FRONT of the mask, which would pick the
            # prefix-mask segment for these layers. Slice the trailing action_self segment
            # explicitly here so each layer sees the correct mask region.
            if attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
                layer_attention_mask = attention_mask[..., -key_states.shape[-2] :]
                _branch = "sliced"
            else:
                layer_attention_mask = attention_mask
                _branch = "full"

            # Attention
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                attn_implementation, eager_attention_forward
            )
            attn_proxy = AttentionModuleProxy(num_key_value_groups=1, training=self.training)
            attn_output, _ = attention_interface(
                attn_proxy,
                query_states,
                key_states,
                value_states,
                layer_attention_mask,
                scaling=scaling,
                dropout=0.0,
            )

            attn_output = attn_output.view(bsz, seq_len, -1)
            # Gated output
            attn_output = attn_output * torch.sigmoid(attn_gate)
            hidden_states = attn.o_proj(attn_output)

        elif self.layer_type == "linear_attention":
            # GatedDeltaNet: independent forward, no cross-mixture
            cache_position = 0 if kv_cache is not None else None
            hidden_states = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                cache_params=kv_cache,
                cache_position=cache_position,
                split_idx=split_idx,
                return_kv_cache=return_kv_cache,
            )

        # Residual 1
        if self.adaptive_mode == "adaLN":
            hidden_states = residual + hidden_states * gate
        else:
            hidden_states = residual + hidden_states

        # Post-attention norm + MLP
        residual = hidden_states
        if self.adaptive_mode == "adaLN":
            hidden_states, gate = self.post_attention_layernorm(hidden_states, time_cond)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.adaptive_mode == "adaLN":
            hidden_states = residual + hidden_states * gate
        else:
            hidden_states = residual + hidden_states

        return hidden_states, new_kv


# ---------------------------------------------------------------------------
# MixtureQwen35 — Complete LLM module for g05
# ---------------------------------------------------------------------------


class MixtureQwen35(nn.Module):
    """Qwen3.5 mixture: input_proj + N × DecoderLayer + norm + output_proj.

    API-compatible with g05 Mixture for use in G05Model.

    Config fields required (same as Mixture + Qwen3.5-specific):
        hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
        num_key_value_heads, head_dim, rms_norm_eps, max_position_embeddings,
        attention_bias, input_type, output_type, use_final_norm,
        layer_types, rope_parameters,
        linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
        linear_num_key_heads, linear_num_value_heads, hidden_act
    """

    DEFAULT_PREFIX_MAP = [
        ("model.language_model.embed_tokens.", "input_proj."),
        ("model.language_model.norm.", "norm."),
        ("model.language_model.layers.", "layers."),
    ]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_types = list(config.layer_types)

        # --- Input Projection ---
        input_type = config.input_type
        if input_type == "embedding":
            self.input_proj = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
            )
        elif input_type == "linear":
            self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        # --- Output Projection ---
        output_type = config.output_type
        if output_type == "lm_head":
            self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if getattr(config, "tie_word_embeddings", True):
                self.output_proj.weight = self.input_proj.weight  # tie weights (2B/4B)
        elif output_type == "linear":
            self.output_proj = nn.Linear(config.hidden_size, config.output_dim)
            # DiT-style zero init: AE outputs zero velocity at step 0, keeping early gradients clean.
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        # --- Time Conditioning (Action Expert only) ---
        self.adaptive_mode = getattr(config, "adaptive_mode", None)
        time_hidden_size = getattr(config, "time_hidden_size", 0) or 0
        if time_hidden_size > 0:
            self.time_embedding = SinusoidalPosEmbPi0(time_hidden_size)
            self.time_mlp_in = nn.Linear(config.hidden_size, config.hidden_size)
            # kaiming_normal_ + zero bias: align with the mixture.py weight-forensics fix.
            nn.init.kaiming_normal_(self.time_mlp_in.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.time_mlp_in.bias)
            self.time_mlp_out = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.kaiming_normal_(self.time_mlp_out.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.time_mlp_out.bias)

        # --- Transformer Layers ---
        self.layers = nn.ModuleList(
            [
                MixtureDecoderLayerQwen35(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # --- Final Norm ---
        if config.use_final_norm:
            if self.adaptive_mode == "adaLN":
                self.norm = AdaptiveRMSNorm(
                    config.hidden_size,
                    config.time_hidden_size,
                    eps=config.rms_norm_eps,
                )
            else:
                self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- Gradient Checkpointing ---
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    # --- Properties ---

    @property
    def _is_linear_proj(self) -> bool:
        return isinstance(self.input_proj, nn.Linear)

    @property
    def head_dim(self) -> int:
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                return layer.self_attn.head_dim
        raise RuntimeError("No full_attention layer found")

    # --- FLOPs estimation ---

    def estimate_flops(self, query_len: int, kv_len: int = None) -> int:
        """Estimate FLOPs for one sample through this Mixture (forward + backward).

        Uses the standard 6*N*S formula for linear projections (fwd+bwd), plus
        explicit O(n^2) term for full-attention layers.

        Note: 6*N*S already covers ALL parameter-dependent ops including GDN
        projections, so no separate linear_attn term is needed (avoids double-count).
        """
        if kv_len is None:
            kv_len = query_len

        num_params = sum(p.numel() for p in self.parameters())
        H = self.config.num_attention_heads
        d = self.config.head_dim
        num_full_attn = sum(1 for lt in self.layer_types if lt == "full_attention")

        # 6*N*S: covers all linear projections (MLP + QKV + GDN in/out), fwd+bwd
        linear_flops = 6 * num_params * query_len

        # Full attention softmax QKV: 12*L*H*d*Q*KV (fwd+bwd), O(n^2) term
        full_attn_flops = 12 * num_full_attn * H * d * query_len * kv_len

        return linear_flops + full_attn_flops

    # --- High-level interfaces (precision managed here) ---

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Input projection.

        VLM:  input_ids [B,S] → [B,S,d] (Embedding, NO sqrt scaling for Qwen3.5)
        AE:   psi_t [B,H,D] → [B,H,d] (Linear, float32, autocast disabled)
        """
        if self._is_linear_proj:
            with torch.autocast(x.device.type, enabled=False):
                return self.input_proj(x.float())
        else:
            # Qwen3.5: plain embedding, no sqrt(hidden_size) scaling
            return self.input_proj(x)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Output projection.

        VLM:  hidden [B,S,d] → logits [B,S,V]
        AE:   hidden [B,H,d] → velocity [B,H,D] (float32, autocast disabled)
        """
        if self._is_linear_proj:
            with torch.autocast(hidden.device.type, enabled=False):
                return self.output_proj(hidden.float())
        else:
            return self.output_proj(hidden)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Time conditioning: t [B] → time_cond [B, hidden_size]. Always float32."""
        time_cond = self.time_embedding(t)
        time_cond = self.time_mlp_in(time_cond)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        return F.silu(time_cond)

    # --- Forward ---

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        kv_cache=None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        time_cond: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
        attn_implementation: str = "eager",
        mixture_name: Optional[str] = None,
        split_idx=None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transformer forward with hybrid attention.

        A single SparseKVCache (kv_cache) manages all layer types:
        - full_attention layers: kv_cache.key_cache / value_cache (dict keyed by layer_idx)
        - linear_attention layers: kv_cache.recurrent_states / conv_states

        Args:
            inputs_embeds: [B, S, hidden_size]
            attention_mask: [B, 1, S, S_total] (0 or -inf), for full_attention layers
            position_ids: [3, B, S] for MRoPE or [B, S]
            kv_cache: SparseKVCache — unified cache for full_attention KV and GDN states.
                      When return_kv_cache=True: reads and writes (VLM prefill path).
                      When return_kv_cache=False: reads prefix KV as past_key_values,
                      GDN reads initial recurrent_state — no writes (AE FM path).
            past_key_values: List[(K,V)] per layer — read-only prefix KV fallback.
            time_cond: [B, hidden_size] for AdaLN
            return_kv_cache: controls whether cache is written to and returned.
                             True = VLM path (collect KV). False = AE FM path (read-only).
            split_idx: prefix boundary for GDN split_recurrent_state extraction
            padding_mask: [B, S] float (1=valid, 0=padding) for GatedDeltaNet

        Returns:
            hidden_states, or (hidden_states, kv_cache) if return_kv_cache
        """

        hidden_states = inputs_embeds

        # Find first full_attention layer for RoPE
        full_attn_layer_idx = next(
            i for i, lt in enumerate(self.layer_types) if lt == "full_attention"
        )

        # Precompute MRoPE
        # NB: torch.zeros(()) (a fill kernel), not torch.tensor(0.0) (a sync H2D copy
        # that is illegal inside CUDA-graph capture — FlashRT commit/FM graphs).
        _dtype_ref = torch.zeros((), device=hidden_states.device)
        with torch.autocast("cuda", enabled=False):
            cos, sin = self.layers[full_attn_layer_idx].self_attn.rotary_emb(
                _dtype_ref, position_ids
            )
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        for layer_idx, layer in enumerate(self.layers):
            if self.layer_types[layer_idx] == "full_attention":
                if self._gradient_checkpointing and self.training:
                    # Extract prefix KV (if exists) to pass as past_key_values INSIDE checkpoint.
                    # Reason: kv_cache is a mutable object — if mutated inside checkpoint (via
                    # update()), backward recomputation would double-concat: [prefix+action+action].
                    # Passing as immutable past_key_values keeps the checkpointed function pure.
                    past_kv = (
                        kv_cache.get(layer_idx)
                        if (kv_cache is not None and kv_cache.has_item(layer_idx))
                        else None
                    )
                    hidden_states, _new_kv = checkpoint_utils.checkpoint(
                        layer,
                        hidden_states,
                        attention_mask,
                        cos,
                        sin,
                        past_kv,
                        time_cond,
                        False,
                        layer_idx,
                        mixture_name,
                        attn_implementation,
                        None,
                        None,
                        use_reentrant=False,
                    )
                    if return_kv_cache and kv_cache is not None and _new_kv is not None:
                        if past_kv is not None:
                            # AE case: _new_kv = [prefix + action] KV, replace directly
                            kv_cache.key_cache[layer_idx] = _new_kv[0]
                            kv_cache.value_cache[layer_idx] = _new_kv[1]
                        else:
                            # VLM case: no prior cache, append to empty
                            kv_cache.update(_new_kv[0], _new_kv[1], layer_idx)
                else:
                    if return_kv_cache:
                        # VLM path: read existing KV and write new KV back (in-place concat)
                        hidden_states, _new_kv = layer(
                            hidden_states,
                            attention_mask,
                            cos,
                            sin,
                            past_key_values=None,
                            time_cond=time_cond,
                            layer_idx=layer_idx,
                            mixture_name=mixture_name,
                            attn_implementation=attn_implementation,
                            kv_cache=kv_cache,
                        )
                    else:
                        # AE FM path: read prefix KV as read-only past_key_values, no write-back
                        past_kv = (
                            kv_cache.get(layer_idx)
                            if (kv_cache is not None and kv_cache.has_item(layer_idx))
                            else None
                        )
                        hidden_states, _new_kv = layer(
                            hidden_states,
                            attention_mask,
                            cos,
                            sin,
                            past_key_values=past_kv,
                            time_cond=time_cond,
                            layer_idx=layer_idx,
                            mixture_name=mixture_name,
                            attn_implementation=attn_implementation,
                            kv_cache=None,
                        )
            else:
                # linear_attention (GatedDeltaNet): never gradient-checkpointed.
                # FLA's custom CUDA autograd function has its own ctx.save_for_backward.
                # return_kv_cache=False (AE FM): reads initial recurrent_state but skips all writes.
                hidden_states, _new_kv = layer(
                    hidden_states,
                    padding_mask,
                    cos,
                    sin,
                    past_key_values=None,
                    time_cond=time_cond,
                    layer_idx=layer_idx,
                    mixture_name=mixture_name,
                    attn_implementation=attn_implementation,
                    kv_cache=kv_cache,
                    split_idx=split_idx,
                    return_kv_cache=return_kv_cache,
                )

            # Monitor
            if self.training and layer_idx == len(self.layers) - 2 and mixture_name is not None:
                monitor = get_global_monitor()
                if monitor is not None and getattr(monitor, "_should_log_x_out", False):
                    monitor.log(
                        {
                            f"monitor/{mixture_name}/x_out_absmax_layer{layer_idx}": hidden_states.abs()
                            .max()
                            .detach(),
                            f"monitor/{mixture_name}/x_out_absmean_layer{layer_idx}": hidden_states.abs()
                            .mean()
                            .detach(),
                            f"monitor/{mixture_name}/x_out_std_layer{layer_idx}": hidden_states.std().detach(),
                        }
                    )

        # Final norm
        if hasattr(self, "norm"):
            if isinstance(self.norm, AdaptiveRMSNorm):
                hidden_states = self.norm(hidden_states, time_cond)[0]
            else:
                hidden_states = self.norm(hidden_states)

        if return_kv_cache:
            return hidden_states, kv_cache
        return hidden_states

    # --- from_pretrained (VLM) ---

    def _load_pretrained_weights(
        self,
        pretrained_model_path: Optional[str] = None,
        tensors: Optional[dict] = None,
    ):
        """Load HF weights with Qwen3.5 key mapping:
        language_model.model.embed_tokens.*  → input_proj.*
        language_model.model.layers.*        → layers.*
        language_model.model.norm.*          → norm.*
        """
        if tensors is None:
            tensors = _load_all_safetensors(pretrained_model_path)

        prefix_map = self.DEFAULT_PREFIX_MAP
        mapped = {}
        for key, tensor in tensors.items():
            for src, dst in prefix_map:
                if key.startswith(src):
                    mapped[key.replace(src, dst, 1)] = tensor
                    break

        # Handle vocab size mismatch: HF vocab < our vocab → partial load
        our_vocab_size = self.input_proj.weight.shape[0]
        hf_key = "input_proj.weight"
        if hf_key in mapped and mapped[hf_key].shape[0] < our_vocab_size:
            hf_vocab_size = mapped[hf_key].shape[0]
            logger.info(
                f"HF vocab ({hf_vocab_size}) < model vocab ({our_vocab_size}), "
                f"partial loading embed_tokens"
            )
            with torch.no_grad():
                self.input_proj.weight[:hf_vocab_size] = mapped[hf_key]
            del mapped[hf_key]

        # For untied models (e.g., 9B): load lm_head weights separately
        # Note: lm_head.weight is a top-level key in safetensors (not under model.language_model)
        if not getattr(self.config, "tie_word_embeddings", True):
            if "lm_head.weight" in tensors:
                mapped["output_proj.weight"] = tensors["lm_head.weight"]

        # strict=False: output_proj (lm_head, tied) doesn't need separate loading
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys in pretrained weights: {unexpected}")
        logger.info(
            f"Loaded pretrained VLM weights: {len(mapped)} keys mapped, "
            f"{len(missing)} missing (expected for tied lm_head)"
        )
        return len(mapped), len(missing)
