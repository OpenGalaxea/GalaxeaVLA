# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import Dict, Tuple, Union

import torch


class _DefaultNoneDict(dict):
    """Dict that returns None for missing keys without inserting them.

    Used by SparseKVCache for GatedDeltaNet conv/recurrent states:
    - Prefill: keys don't exist yet → returns None (GatedDeltaNet ignores None states in prefill)
    - After prefill: GatedDeltaNet writes actual tensors → subsequent reads return tensors
    """

    def __missing__(self, key):
        return None


class SparseKVCache:
    """Dict-based KV cache for hybrid architectures (e.g., Qwen3.5).

    Only full_attention layers produce KV cache; linear_attention layers
    (GatedDeltaNet) don't. Using a dict keyed by layer_idx avoids
    index-mismatch problems of a list-based cache.

    API: has_item / num_items / get / update /
    __getitem__ / detach all work the same way.
    """

    def __init__(
        self,
        key_cache: Dict[int, torch.Tensor] = None,
        value_cache: Dict[int, torch.Tensor] = None,
        last_linear_layer: int = None,
    ) -> None:
        self.key_cache: Dict[int, torch.Tensor] = key_cache if key_cache is not None else {}
        self.value_cache: Dict[int, torch.Tensor] = value_cache if value_cache is not None else {}
        # GatedDeltaNet (linear_attention) state — used for AR decode
        self.conv_states: Dict[int, torch.Tensor] = _DefaultNoneDict()
        self.recurrent_states: Dict[int, torch.Tensor] = _DefaultNoneDict()
        # Prefix-only recurrent states (written by GatedDeltaNet when split_idx is set).
        # Contains states at split_idx boundary — clean of AR action tokens.
        self.split_recurrent_states: Dict[int, torch.Tensor] = _DefaultNoneDict()
        # Index of the last linear_attention layer (for has_previous_state check)
        self.last_linear_layer = last_linear_layer

    @property
    def has_previous_state(self) -> bool:
        """True if the last linear_attention layer's conv_state was populated."""
        if self.last_linear_layer is None:
            return len(self.conv_states) > 0
        return self.conv_states.get(self.last_linear_layer) is not None

    def has_item(self, layer_idx: int) -> bool:
        return layer_idx in self.key_cache

    def num_items(self) -> int:
        """Return the cached sequence length (from first cached layer)."""
        if not self.key_cache:
            return 0
        first_key = next(iter(self.key_cache.values()))
        return first_key.shape[-2]

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __getitem__(self, index: Union[int, slice]):
        """
        1. cache[layer_idx: int] -> (k, v) for that layer
        2. cache[:N] slice -> new SparseKVCache with seq_len sliced
        """
        if isinstance(index, int):
            return self.key_cache[index], self.value_cache[index]
        elif isinstance(index, slice):
            new_keys = {idx: k[..., index, :] for idx, k in self.key_cache.items()}
            new_vals = {idx: v[..., index, :] for idx, v in self.value_cache.items()}
            return SparseKVCache(
                key_cache=new_keys,
                value_cache=new_vals,
                last_linear_layer=self.last_linear_layer,
            )
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def detach(self):
        new = SparseKVCache(
            key_cache={idx: k.detach() for idx, k in self.key_cache.items()},
            value_cache={idx: v.detach() for idx, v in self.value_cache.items()},
            last_linear_layer=self.last_linear_layer,
        )
        for idx, rs in self.recurrent_states.items():
            new.recurrent_states[idx] = rs.detach()
        for idx, rs in self.split_recurrent_states.items():
            new.split_recurrent_states[idx] = rs.detach()
        return new

    def repeat_batch(self, n: int) -> "SparseKVCache":
        """Repeat KV cache along batch dimension N times. [B,...] → [N*B,...].

        Handles all GatedDeltaNet states (conv, recurrent, split_recurrent)
        in addition to the standard KV cache entries.
        """
        if n == 1:
            return self
        new = SparseKVCache(
            key_cache={
                idx: k.repeat(n, *([1] * (k.ndim - 1))) for idx, k in self.key_cache.items()
            },
            value_cache={
                idx: v.repeat(n, *([1] * (v.ndim - 1))) for idx, v in self.value_cache.items()
            },
            last_linear_layer=self.last_linear_layer,
        )
        for idx, cs in self.conv_states.items():
            new.conv_states[idx] = cs.repeat(n, *([1] * (cs.ndim - 1)))
        for idx, rs in self.recurrent_states.items():
            new.recurrent_states[idx] = rs.repeat(n, *([1] * (rs.ndim - 1)))
        for idx, srs in self.split_recurrent_states.items():
            new.split_recurrent_states[idx] = srs.repeat(n, *([1] * (srs.ndim - 1)))
        return new

    def batch_slice(self, i: int) -> "SparseKVCache":
        """Slice the i-th sample along the batch dimension into a new SparseKVCache.

        Used by per-sample BAR inference to split the batch KV cache from prefill so
        each sample decodes independently. Copies GatedDeltaNet conv/recurrent states
        in addition to key/value cache.
        """
        new = SparseKVCache(
            key_cache={idx: k[i : i + 1] for idx, k in self.key_cache.items()},
            value_cache={idx: v[i : i + 1] for idx, v in self.value_cache.items()},
            last_linear_layer=self.last_linear_layer,
        )
        for idx, cs in self.conv_states.items():
            new.conv_states[idx] = cs[i : i + 1]
        for idx, rs in self.recurrent_states.items():
            new.recurrent_states[idx] = rs[i : i + 1]
        for idx, srs in self.split_recurrent_states.items():
            new.split_recurrent_states[idx] = srs[i : i + 1]
        return new

    def to_past_key_values(self, num_layers: int, device=None, dtype=None):
        """Convert to list[tuple] format for Mixture.forward() fast path.

        Layers without cache get a zero-length empty KV pair.
        Returns: list of (key, value) tuples, one per layer.
        """
        if not self.key_cache:
            return None

        sample_k = next(iter(self.key_cache.values()))
        B, H, _, D = sample_k.shape
        _device = device or sample_k.device
        _dtype = dtype or sample_k.dtype
        empty = torch.empty(B, H, 0, D, device=_device, dtype=_dtype)
        empty_pair = (empty, empty)

        return [
            (self.key_cache[i], self.value_cache[i]) if i in self.key_cache else empty_pair
            for i in range(num_layers)
        ]
