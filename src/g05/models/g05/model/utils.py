# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import glob
import os

import torch
import torch.nn as nn
from g05.models.kv_cache import SparseKVCache
from g05.utils.hf import resolve_hf_model_path


def _load_all_safetensors(path: str) -> dict:
    """Load all safetensor files from a directory into a single dict."""
    from safetensors import safe_open

    path = resolve_hf_model_path(
        path,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    safetensors_files = glob.glob(os.path.join(path, "*.safetensors"))
    assert len(safetensors_files) > 0, f"No safetensors found in {path}"
    tensors = {}
    for f_path in safetensors_files:
        with safe_open(f_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def kv_cache_seq_len(past_key_values) -> int:
    if isinstance(past_key_values, SparseKVCache):
        return past_key_values.num_items()
    return past_key_values[0][0].size(2)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )


class AttentionModuleProxy:
    """Proxy to satisfy ALL_ATTENTION_FUNCTIONS interface signature."""

    def __init__(self, num_key_value_groups: int = 1, training: bool = False):
        self.num_key_value_groups = num_key_value_groups
        self.training = training


def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, softcap=None, **kwargs
):
    """Eager attention forward compatible with ALL_ATTENTION_FUNCTIONS interface."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    with torch.autocast("cuda", enabled=False):
        # Gemma 2 style soft capping: tanh(logits / cap) * cap
        if softcap is not None:
            attn_weights = (torch.tanh(attn_weights.float() / softcap) * softcap).to(query.dtype)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    with torch.autocast("cuda", enabled=False):
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
