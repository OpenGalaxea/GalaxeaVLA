# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""ActionPartitioner — key routing and dimension chunking for ActionCodecV2.

Owns the full partition → merge → chunk pipeline (encode direction) and its
inverse unchunk → unmerge → unpartition (decode direction).

Key routing is driven by rule_based_key_patterns: a list of substrings.
If any pattern appears in a key name, that key goes to the rule-based path;
otherwise it goes to the NN model path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PartitionState:
    """State produced by encode_partition, consumed by decode_partition."""

    chunk_map: Dict[str, List[str]]
    original_key_dims: Dict[str, int]
    rule_dict: Dict[str, torch.Tensor]
    nn_repr_info: Dict[str, Any]
    key_order: List[str]
    is_flat_input: bool
    input_parts_meta: Optional[Dict[str, int]]


class ActionPartitioner:
    """Manages key routing, merge_spec, and dimension chunking.

    Pipeline (encode direction)::
        component_dict → partition → merge → chunk

    Pipeline (decode direction)::
        chunked_recon + PartitionState → unchunk → unmerge → nn_dict
    """

    def __init__(
        self,
        key_dims: Dict[str, int],
        max_component_dim: int,
        rule_based_key_patterns: List[str],
        merge_preprocessor=None,
    ):
        self.key_dims = dict(key_dims)
        self.max_component_dim = max_component_dim
        self.rule_based_key_patterns = list(rule_based_key_patterns)
        self.merge_preprocessor = merge_preprocessor

    def is_rule_based_key(self, key: str) -> bool:
        return any(p in key for p in self.rule_based_key_patterns)

    def partition(
        self, component_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        nn_dict = {}
        rule_dict = {}
        for k, v in component_dict.items():
            if self.is_rule_based_key(k):
                rule_dict[k] = v
            else:
                nn_dict[k] = v
        return nn_dict, rule_dict

    def unpartition(
        self,
        nn_dict: Dict[str, torch.Tensor],
        rule_dict: Dict[str, torch.Tensor],
        key_order: List[str],
    ) -> Dict[str, torch.Tensor]:
        combined = {}
        combined.update(nn_dict)
        combined.update(rule_dict)
        return {k: combined[k] for k in key_order if k in combined}

    def merge(
        self,
        nn_dict: Dict[str, torch.Tensor],
        *,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        action_op_mask: Optional[torch.Tensor] = None,
        input_parts_meta: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if self.merge_preprocessor is None:
            return nn_dict, {}

        actual_parts_meta = input_parts_meta or self.key_dims
        actual_nn_parts_meta = {
            k: v for k, v in actual_parts_meta.items() if not self.is_rule_based_key(k)
        }

        ordered_keys = list(actual_nn_parts_meta.keys())
        flat = torch.cat([nn_dict[k] for k in ordered_keys], dim=-1)
        B, T, D = flat.shape

        nn_dim_is_pad = None
        if action_dim_is_pad is not None:
            cum = 0
            nn_idx = []
            for k, d in actual_parts_meta.items():
                if not self.is_rule_based_key(k):
                    nn_idx.extend(range(cum, cum + d))
                cum += d
            nn_dim_is_pad = action_dim_is_pad[:, nn_idx]

        nn_op_mask = None
        if action_op_mask is not None:
            cum = 0
            nn_idx = []
            for k, d in actual_parts_meta.items():
                if not self.is_rule_based_key(k):
                    nn_idx.extend(range(cum, cum + d))
                cum += d
            nn_op_mask = action_op_mask[..., nn_idx]

        if nn_dim_is_pad is not None or nn_op_mask is not None:
            _op = nn_op_mask if nn_op_mask is not None else flat.new_ones(B, D, dtype=torch.bool)
            merged_parts, _ = self.merge_preprocessor.forward(
                flat, _op, nn_dim_is_pad, input_parts_meta=actual_nn_parts_meta
            )
        else:
            op_mask = flat.new_ones(B, D, dtype=torch.bool)
            merged_parts, _ = self.merge_preprocessor.forward(
                flat, op_mask, None, input_parts_meta=actual_nn_parts_meta
            )

        repr_info = self.merge_preprocessor.last_repr_info
        return merged_parts, repr_info

    def unmerge(
        self, merged_dict: Dict[str, torch.Tensor], repr_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.merge_preprocessor is None or not repr_info:
            return merged_dict

        device = next(iter(merged_dict.values())).device
        ri = {
            k: (
                torch.tensor(v, dtype=torch.bool, device=device)
                if not isinstance(v, torch.Tensor)
                else v.to(device)
            )
            for k, v in repr_info.items()
        }
        nn_flat = self.merge_preprocessor.backward(merged_dict, ri)
        nn_pm = {k: v for k, v in self.key_dims.items() if not self.is_rule_based_key(k)}
        nn_flat = nn_flat[..., : sum(nn_pm.values())]
        splits = torch.split(nn_flat, list(nn_pm.values()), dim=-1)
        return dict(zip(nn_pm.keys(), splits))

    def chunk(
        self, component_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        max_d = self.max_component_dim
        chunked = {}
        chunk_map = {}
        for k, v in component_dict.items():
            d = v.shape[-1]
            if self.is_rule_based_key(k):
                chunked[k] = v
                chunk_map[k] = [k]
                continue
            if d > max_d:
                chunk_keys = []
                for i, start in enumerate(range(0, d, max_d)):
                    end = min(start + max_d, d)
                    chunk_v = v[..., start:end]
                    pad_size = max_d - chunk_v.shape[-1]
                    if pad_size > 0:
                        chunk_v = F.pad(chunk_v, (0, pad_size))
                    chunk_key = f"{k}__chunk_{i}"
                    chunked[chunk_key] = chunk_v
                    chunk_keys.append(chunk_key)
                chunk_map[k] = chunk_keys
            else:
                chunked[k] = v
                chunk_map[k] = [k]
        return chunked, chunk_map

    def unchunk(
        self,
        chunked_dict: Dict[str, torch.Tensor],
        chunk_map: Dict[str, List[str]],
        original_key_dims: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        max_d = self.max_component_dim
        merged = {}
        for orig_key, chunk_keys in chunk_map.items():
            if len(chunk_keys) == 1 and chunk_keys[0] == orig_key:
                if orig_key in chunked_dict:
                    merged[orig_key] = chunked_dict[orig_key]
            else:
                orig_dim = original_key_dims.get(orig_key, max_d * len(chunk_keys))
                parts = []
                remaining = orig_dim
                for ck in chunk_keys:
                    if ck not in chunked_dict or remaining <= 0:
                        break
                    take = min(max_d, remaining)
                    parts.append(chunked_dict[ck][..., :take])
                    remaining -= take
                if parts:
                    merged[orig_key] = torch.cat(parts, dim=-1)
        return merged

    def encode_partition(
        self,
        component_dict: Dict[str, torch.Tensor],
        *,
        action_dim_is_pad=None,
        action_op_mask=None,
        input_parts_meta=None,
        is_flat_input: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], PartitionState]:
        nn_dict, rule_dict = self.partition(component_dict)
        merged_dict, nn_repr_info = self.merge(
            nn_dict,
            action_dim_is_pad=action_dim_is_pad,
            action_op_mask=action_op_mask,
            input_parts_meta=input_parts_meta,
        )
        original_key_dims = {k: v.shape[-1] for k, v in merged_dict.items()}
        nn_chunked, chunk_map = self.chunk(merged_dict)

        state = PartitionState(
            chunk_map=chunk_map,
            original_key_dims=original_key_dims,
            rule_dict=rule_dict,
            nn_repr_info=nn_repr_info,
            key_order=list(component_dict.keys()),
            is_flat_input=is_flat_input,
            input_parts_meta=input_parts_meta,
        )
        return nn_chunked, state

    def decode_partition(
        self,
        chunked_recon: Dict[str, torch.Tensor],
        state: PartitionState,
    ) -> Dict[str, torch.Tensor]:
        recon = self.unchunk(chunked_recon, state.chunk_map, state.original_key_dims)
        recon = self.unmerge(recon, state.nn_repr_info)
        return recon
