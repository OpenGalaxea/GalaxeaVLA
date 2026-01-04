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
Wrapper around the mixtures (without encoder / decoder)

Agnostic to the mixture setup

KV caches --- There are a few different modes depending on the setting:
    - text generation, only vlm active, use vlm cache --- append active (mode="append")
    - action naive inference, all active, use vlm and proprio cache --- no new tokens for the active mixture (mode="no_append")
    - action inference, no cache during vlm and proprio forward, then use vlm and proprio cache --- append, non-active (mode="append_non_active")
    - action flow matching training, all active, no cache (mode does not matter)
"""

import copy
import math

from typing import List, Optional

import torch
import torch.nn as nn

from accelerate.logging import get_logger
from omegaconf import OmegaConf

from .mixture import Mixture
from galaxea_fm.models.kv_cache import KVCache

logger = get_logger(__name__)


def forward_mixture_layers(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    embeds_all: dict[torch.FloatTensor],
    layer_idx: int,
    kv_caches: dict[KVCache] = {},
    cache_mode: str = "append_non_active",
    time_cond: Optional[torch.FloatTensor] = None,
) -> dict[torch.FloatTensor]:
    """the usual norm + attn + res + norm + mlp + res"""
    active_mixture_names = list(embeds_all.keys())

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_attn = embeds_all
    hidden_states_input_norm = {}
    for name in active_mixture_names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "input_layernorm",
            embeds_all[name],
        )
    hidden_states_pre_attn = hidden_states_input_norm

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_attn = forward_mixture_attn(
        mixtures,
        hidden_states_all=hidden_states_pre_attn,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        layer_idx=layer_idx,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in active_mixture_names:
        hidden_states_post_res[name] = (
            residuals_pre_attn[name] + hidden_states_pre_res[name]
        )
    hidden_states_pre_post_attn = hidden_states_post_res

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_post_attn = hidden_states_pre_post_attn
    hidden_states_post_post_attn = {}
    for name in active_mixture_names:
        hidden_states_post_post_attn[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "post_attention_layernorm",
            hidden_states_pre_post_attn[name],
        )
    hidden_states_pre_mlp = hidden_states_post_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_pos_mlp = {}
    for name in active_mixture_names:
        hidden_states_pos_mlp[name] = mixtures[name].layer_func(
            "mlp",
            layer_idx,
            hidden_states_pre_mlp[name],
        )
    hidden_states_pre_final_res = hidden_states_pos_mlp

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_final = {}
    for name in active_mixture_names:
        hidden_states_final[name] = residuals_pre_post_attn[name] + hidden_states_pre_final_res[name]
    return hidden_states_final


def forward_mixture_attn(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    hidden_states_all: dict[torch.FloatTensor],
    layer_idx: int,
    kv_caches: dict[KVCache] = {},
    cache_mode: str = "append_non_active",
    attn_softclamp: float = 50.0,  # default in gemma
    attention_dropout: float = 0.0,
) -> dict[torch.FloatTensor]:
    """Assume all mixtures have the same head dim"""
    assert cache_mode in [
        "no_append",
        "append",
        "append_non_active",
    ], f"Invalid cache mode: {cache_mode}"
    bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    active_mixture_names = list(hidden_states_all.keys())

    # always re-compute queries
    query_states_all = {}
    for name in active_mixture_names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func("forward_q_proj", layer_idx, hidden_states_all[name])
        query_states_all[name] = query_states

    # use kv caches from non-active mixtures
    key_states_all = {}
    value_states_all = {}
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                key_states_all[name], value_states_all[name] = kv_cache.get(layer_idx)

    # the caching logic below can be much simplified if we ignore the "no_append" mode, which is only used in the naive action inference mode
    for name in active_mixture_names:
        # prepare rope
        query_states = query_states_all[name]
        rope_cos, rope_sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, query_states, position_ids_all[name]
        )

        # always use kv cache if it has the current layer
        flag_cached_mixture = name in kv_caches and kv_caches[name].has_item(layer_idx)
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                layer_idx
            )  # note: rope already applied before they were cached

        # always add to cache in append mode, or kv cache does not have the layer yet (in no_append mode)
        flag_to_cache_mixture = (name in kv_caches and not kv_caches[name].has_item(layer_idx)) or cache_mode == "append"

        # calculate kv for new tokens if in append mode or this layer is not cached
        key_states_new, value_states_new = None, None
        flag_calc_new_kv = not flag_cached_mixture or cache_mode == "append"
        assert flag_cached_mixture or flag_calc_new_kv, "Cannot skip new kv calculation while also not using cache!"
        if flag_calc_new_kv:
            hidden_states = hidden_states_all[name]
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func("forward_k_proj", layer_idx, hidden_states)
            value_states_new = mixtures[name].attn_func("forward_v_proj", layer_idx, hidden_states)

            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_apply_rotary_emb",
                layer_idx,
                key_states_new,
                rope_cos,
                rope_sin,
            )

            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    layer_idx,
                )

        # always apply rope to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func("forward_apply_rotary_emb", layer_idx, query_states, rope_cos, rope_sin)
        query_states_all[name] = query_states

        # assign K and V carefully for this active mixture
        if flag_cached_mixture:
            key_states = key_states_cached
            value_states = value_states_cached
            if key_states_new is not None:
                key_states = torch.cat((key_states, key_states_new), dim=-2)
            if value_states_new is not None:
                value_states = torch.cat((value_states, value_states_new), dim=-2)
        else:
            key_states = key_states_new
            value_states = value_states_new
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # Repeat the key and values to match the number of heads of the query
    for name in key_states_all:
        key_states, value_states = mixtures[name].attn_func(
            "repeat_kv",
            layer_idx,
            key_states_all[name],
            value_states_all[name],
        )
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    # Perform the calculation as usual, Q * K^T / sqrt(head_dim)
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        mixtures[active_mixture_names[0]].head_dim
    )

    # Soft capping
    attn_weights = attn_weights / attn_softclamp
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_softclamp

    # Apply the softmax / dropout
    attn_weights = attn_weights + attention_mask
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    with torch.autocast("cuda", enabled=False):
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=attention_dropout,
        training=mixtures[active_mixture_names[0]].training,
    )
    # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    attn_output = torch.matmul(attn_weights, value_states)

    # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Concatenate all the heads together. [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q * Head_Dim]
    attn_output = attn_output.view(bsz, sum(q_lens), -1)

    # Split into the different mixtures
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {key: value for key, value in zip(active_mixture_names, attn_outputs)}

    # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
    attn_outputs_final = {}
    for name in active_mixture_names:
        attn_outputs_final[name] = mixtures[name].attn_func("forward_o_proj", layer_idx, attn_outputs[name])
    return attn_outputs_final


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.num_mixture = len(config.mixture)
        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]  # name of the mixtures that use cache during generation; no cache during training

        # Mixtures
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            # mixture_config = OmegaConf.merge(config, mixture_config)
            mixture_config = copy.deepcopy(mixture_config)
            mixture_config = OmegaConf.merge(config, mixture_config)
            self.mixtures[mixture_name] = Mixture(mixture_config)

            num_params = sum(p.numel() for p in self.mixtures[mixture_name].parameters())
            if num_params >= 1e9:
                unit = "B"
                num_params_display = num_params / 1e9
            else:
                unit = "M"
                num_params_display = num_params / 1e6

            logger.info("[bold]{}[/] number of parameters: [bold]{:.2f}{}[/]".format(mixture_name, num_params_display, unit))

        self.mixture_names = list(config.mixture.keys())

    def init_mixture_caches(self):
        return {name: KVCache() for name in self.cache_names}

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: dict[torch.LongTensor],
        embeds_all: dict[torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,
        kv_caches: dict[KVCache] = {},
        cache_mode: str = "append_non_active",
        return_caches: bool = False,
        return_intermediate_layers: bool = False,
    ) -> dict[torch.FloatTensor]:
        """
        Assume attention_mask is in the right block attention form

        embeds_all and position_ids_all need to be in the correct order, e.g., {"vlm": ..., "proprio": ..., "action": ...}
        """
        active_mixture_names = list(embeds_all.keys())

        # normalization
        # [Batch_Size, Seq_Len, Hidden_Size]
        for name in active_mixture_names:
            if self.config.mixture[name].get("scale_input", True):
                hidden_size = embeds_all[name].shape[-1]
                normalizer = torch.tensor(
                    hidden_size**0.5,
                    dtype=embeds_all[name].dtype,
                    device=embeds_all[name].device,
                )
                embeds_all[name] *= normalizer

        intermediates = {k: [embeds_all[k]] for k in embeds_all.keys()}
        # layers
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = forward_mixture_layers(
                self.mixtures,
                attention_mask,
                position_ids_all,
                embeds_all,
                layer_idx=layer_idx,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode=cache_mode,
            )
            for k in embeds_all.keys():
                intermediates[k].append(embeds_all[k])

        # each mixture has N+1 layers, where N is the number of attention blocks
        intermediates = {k: torch.stack(v, dim=1) for k, v in intermediates.items()}

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_all = {}
        for name in active_mixture_names:
            hidden_states_all[name] = self.mixtures[name].forward_norm(embeds_all[name], time_cond)  # can be None
        if return_caches:
            if return_intermediate_layers:
                return hidden_states_all, kv_caches, intermediates
            return hidden_states_all, kv_caches
        if return_intermediate_layers:
            return hidden_states_all, intermediates
        return hidden_states_all
