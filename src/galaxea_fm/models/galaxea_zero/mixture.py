"""
This file is based on work from open-pi-zero (https://github.com/allenzren/open-pi-zero),
licensed under the MIT License.

Modifications:
   Copyright (c) 2025 Galaxea AI.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Individual mixture in PaliGemma format

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .paligemma.modules import (
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
)
from .utils import apply_rotary_pos_emb, repeat_kv
from .modules import AdaptiveLayerscale, AdaptiveRMSNorm


class Mixture(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([MixtureDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.adaptive_mode = None
        if config.use_final_norm:
            self.norm = GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    def layer_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx], method_name)(*args)

    def attn_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx].self_attn, method_name)(*args)

    def forward_norm(
        self,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor | None:
        if hasattr(self, "norm"):
            args = [x] if self.adaptive_mode is None else [x, cond]
            return self.norm(*args)
        else:
            return None


class MixtureDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MixtureAttention(config)

        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward_norm(
        self,
        norm_name: str,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor | None:
        return getattr(self, norm_name)(x)


class MixtureAttention(nn.Module):
    """assume head_dim same for all blocks"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert config.hidden_size % self.num_heads == 0

        layer = nn.Linear
        self.q_proj = layer(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = layer(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = layer(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = layer(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward_q_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(x)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        return query_states

    def forward_k_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return key_states

    def forward_v_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return value_states

    def forward_o_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.o_proj(x)

    @torch.autocast("cuda", enabled=False)  # Critical!!!
    def forward_rotary_emb(self, x: torch.FloatTensor, position_ids: torch.LongTensor) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(x.float(), position_ids, seq_len=None)
        return cos, sin

    @torch.autocast("cuda", enabled=False)  # Critical!!!
    def forward_apply_rotary_emb(
        self,
        states: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Seq_Len, Head_Dim]
        orig_dtype = states.dtype
        states = apply_rotary_pos_emb(states, cos, sin)
        return states.to(orig_dtype)

    def repeat_kv(self, key_states: torch.FloatTensor, value_states: torch.FloatTensor) -> Tuple[torch.FloatTensor]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states
