# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

import torch

from g05.data_processor import BaseActionStateTransform


class ConcatKeysTransform(BaseActionStateTransform):
    """
    Concatenates multiple keys into one along the last dimension.

    Applies to action, state, action_op_mask, and gt_action dicts in batch.
    Used for WBC configs: chassis_velocities(3D) + torso(4D) → lower_body(7D).

    invertible=True: backward splits output_key back into input_keys by input_sizes.
    """

    invertible = True

    def __init__(self, input_keys: List[str], input_sizes: List[int], output_key: str):
        assert len(input_keys) == len(input_sizes), (
            "input_keys and input_sizes must have same length"
        )
        self.input_keys = input_keys
        self.input_sizes = input_sizes
        self.output_key = output_key

    def _concat(self, d: Dict) -> Dict:
        present = [(k, d[k]) for k in self.input_keys if k in d]
        if not present:
            return d
        out = dict(d)
        out[self.output_key] = torch.cat([v for _, v in present], dim=-1)
        for k, _ in present:
            del out[k]
        return out

    def _split(self, d: Dict) -> Dict:
        if self.output_key not in d:
            return d
        out = dict(d)
        parts = out.pop(self.output_key).split(self.input_sizes, dim=-1)
        for k, t in zip(self.input_keys, parts):
            out[k] = t
        return out

    def forward(self, batch: Dict) -> Dict:
        batch = dict(batch)
        for field in ("action", "state", "action_op_mask", "gt_action"):
            if field in batch:
                batch[field] = self._concat(batch[field])
        return batch

    def backward(self, batch: Dict) -> Dict:
        batch = dict(batch)
        for field in ("action", "state", "action_op_mask", "gt_action"):
            if field in batch:
                batch[field] = self._split(batch[field])
        return batch


class WrapStateAngle(BaseActionStateTransform):
    """
    Wrap angle values to [-pi, pi] range using atan2(sin, cos).

    This transform is NOT invertible because wrapping loses information
    about the original angle's winding number.
    """

    invertible = False

    def __init__(self, keys: List[str]):
        self.keys = keys

    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch

    def backward(self, batch):
        return batch


class BinarizeTransform(BaseActionStateTransform):
    """
    Binarize selected action/state keys with strict threshold comparison.

    Values greater than threshold become max; all others become min. The
    thresholds/max/min parameters may be scalars shared by every selected key
    and dim, or dicts keyed by key name with per-dim lists.
    """

    invertible = False

    def __init__(
        self,
        keys: List[str],
        thresholds: Any = 0.5,
        max: Any = 1.0,
        min: Any = 0.0,
        fields: Optional[List[str]] = None,
    ):
        self.keys = keys
        self.fields = tuple(fields or ("action", "state"))
        self._thresholds = self._normalize_param("thresholds", thresholds)
        self._max = self._normalize_param("max", max)
        self._min = self._normalize_param("min", min)

    def _normalize_param(self, name: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            missing = [key for key in self.keys if key not in value]
            if missing:
                raise ValueError(f"BinarizeTransform: {name} missing keys {missing}")
            return dict(value)
        return value

    @staticmethod
    def _is_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    def _value_for_key(self, name: str, params: Any, key: str, x: torch.Tensor) -> torch.Tensor:
        value = params[key] if isinstance(params, Mapping) else params
        if torch.is_tensor(value):
            tensor = value.to(device=x.device, dtype=x.dtype)
        elif self._is_sequence(value):
            tensor = torch.tensor(list(value), device=x.device, dtype=x.dtype)
        else:
            tensor = torch.tensor(value, device=x.device, dtype=x.dtype)

        if tensor.ndim > 1:
            raise ValueError(
                f"BinarizeTransform: {name}[{key!r}] must be a scalar or 1D list, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.ndim == 1 and tensor.numel() not in (1, x.shape[-1]):
            raise ValueError(
                f"BinarizeTransform: {name}[{key!r}] last dim {tensor.numel()} "
                f"does not match input last dim {x.shape[-1]}"
            )
        return tensor

    def _apply_dict(self, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(values)
        for key in self.keys:
            if key not in out:
                continue
            x = out[key]
            threshold = self._value_for_key("thresholds", self._thresholds, key, x)
            high = self._value_for_key("max", self._max, key, x)
            low = self._value_for_key("min", self._min, key, x)
            out[key] = torch.where(x > threshold, high, low)
        return out

    def forward(self, batch: Dict) -> Dict:
        out_batch = dict(batch)
        for field in self.fields:
            if field in batch:
                out_batch[field] = self._apply_dict(batch[field])
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        return batch


class ArcSinhTransform(BaseActionStateTransform):
    """
    Apply arcsinh normalization to action keys.

    For each key in bounds, applies: x = arcsinh(x / s), where s = 1.5 * bound.
    This is an invertible transform that compresses large values while preserving
    the sign and scale of small values.
    """

    invertible = True

    def __init__(self, bounds: Dict[str, List[float]]):
        for key, bound_list in bounds.items():
            for v in bound_list:
                if v <= 0:
                    raise ValueError(f"bounds must be positive, got bounds[{key!r}] = {bound_list}")
        self._bounds: Dict[str, torch.Tensor] = {
            key: torch.tensor(bound_list, dtype=torch.float32) for key, bound_list in bounds.items()
        }

    def forward(self, batch: Dict) -> Dict:
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key, bound in self._bounds.items():
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != bound.shape[0]:
                raise ValueError(
                    f"ArcSinhTransform: action[{key!r}] has shape {tuple(x.shape)}, "
                    f"but bounds has {bound.shape[0]} elements"
                )
            # .to(device) called per-batch because this class is not an nn.Module
            s = 1.5 * bound.to(x.device)
            out_batch["action"][key] = torch.asinh(x / s)
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key, bound in self._bounds.items():
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != bound.shape[0]:
                raise ValueError(
                    f"ArcSinhTransform: action[{key!r}] has shape {tuple(x.shape)}, "
                    f"but bounds has {bound.shape[0]} elements"
                )
            # .to(device) called per-batch because this class is not an nn.Module
            s = 1.5 * bound.to(x.device)
            out_batch["action"][key] = s * torch.sinh(x)
        return out_batch


class LinearTailTransform(BaseActionStateTransform):
    """
    Piecewise transform: values inside [q01, q99] stay linear; outside values
    use log1p to compress extremes.

    forward:
      x > q99[i]:  f = q99[i]  + c_pos[i] * log1p((x - q99[i])  / c_pos[i])
      q01[i] ≤ x ≤ q99[i]: f = x
      x < q01[i]:  f = q01[i]  - c_neg[i] * log1p((q01[i] - x)  / c_neg[i])

    Where:
      c_pos[i] = tail_scale * (q99[i] - mean[i])    # per-dim positive-side compression coefficient
      c_neg[i] = tail_scale * (mean[i] - q01[i])    # per-dim negative-side compression coefficient

    backward, exact inverse:
      y > q99[i]:  x = q99[i]  + c_pos[i] * expm1((y - q99[i])  / c_pos[i])
      q01[i] ≤ y ≤ q99[i]: x = y
      y < q01[i]:  x = q01[i]  - c_neg[i] * expm1((q01[i] - y)  / c_neg[i])

    tail_scale reference table, default 0.075:
      0.075 → mean + 2*sigma → mean + 1.20*sigma
      0.100 → mean + 2*sigma → mean + 1.24*sigma
      0.050 → mean + 2*sigma → mean + 1.15*sigma
    """

    invertible = True

    def __init__(
        self,
        mean: Dict[str, List[float]],
        q01: Dict[str, List[float]],
        q99: Dict[str, List[float]],
        tail_scale: float = 0.075,
    ):
        if tail_scale <= 0:
            raise ValueError(f"tail_scale must be > 0, got {tail_scale}")

        self._q01: Dict[str, torch.Tensor] = {}
        self._q99: Dict[str, torch.Tensor] = {}
        self._c_pos: Dict[str, torch.Tensor] = {}
        self._c_neg: Dict[str, torch.Tensor] = {}

        keys = set(mean) | set(q01) | set(q99)
        for key in keys:
            if key not in mean or key not in q01 or key not in q99:
                raise ValueError(
                    f"LinearTailTransform: key {key!r} must appear in all of mean/q01/q99"
                )
            m = torch.tensor(mean[key], dtype=torch.float32)
            lo = torch.tensor(q01[key], dtype=torch.float32)
            hi = torch.tensor(q99[key], dtype=torch.float32)
            if not (lo < hi).all():
                raise ValueError(
                    f"LinearTailTransform: q01 must be < q99 for key {key!r}, "
                    f"got q01={lo.tolist()}, q99={hi.tolist()}"
                )
            if not ((m > lo) & (m < hi)).all():
                raise ValueError(
                    f"LinearTailTransform: mean must be strictly inside (q01, q99) for key {key!r}, "
                    f"got mean={m.tolist()}, q01={lo.tolist()}, q99={hi.tolist()}"
                )
            sigma_pos = hi - m  # > 0
            sigma_neg = m - lo  # > 0
            self._q01[key] = lo
            self._q99[key] = hi
            self._c_pos[key] = tail_scale * sigma_pos
            self._c_neg[key] = tail_scale * sigma_neg

    def _apply_forward(self, x: torch.Tensor, key: str) -> torch.Tensor:
        q01 = self._q01[key].to(x.device)
        q99 = self._q99[key].to(x.device)
        c_pos = self._c_pos[key].to(x.device)
        c_neg = self._c_neg[key].to(x.device)

        pos_tail = q99 + c_pos * torch.log1p(torch.clamp((x - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.log1p(torch.clamp((q01 - x) / c_neg, min=0.0))

        return torch.where(x > q99, pos_tail, torch.where(x < q01, neg_tail, x))

    def _apply_backward(self, y: torch.Tensor, key: str) -> torch.Tensor:
        q01 = self._q01[key].to(y.device)
        q99 = self._q99[key].to(y.device)
        c_pos = self._c_pos[key].to(y.device)
        c_neg = self._c_neg[key].to(y.device)

        pos_tail = q99 + c_pos * torch.expm1(torch.clamp((y - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.expm1(torch.clamp((q01 - y) / c_neg, min=0.0))

        return torch.where(y > q99, pos_tail, torch.where(y < q01, neg_tail, y))

    def forward(self, batch: Dict) -> Dict:
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key in self._q99:
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != self._q99[key].shape[0]:
                raise ValueError(
                    f"LinearTailTransform: action[{key!r}] last dim {x.shape[-1]} "
                    f"!= bounds shape {self._q99[key].shape[0]}"
                )
            out_batch["action"][key] = self._apply_forward(x, key)
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key in self._q99:
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != self._q99[key].shape[0]:
                raise ValueError(
                    f"LinearTailTransform: action[{key!r}] last dim {x.shape[-1]} "
                    f"!= bounds shape {self._q99[key].shape[0]}"
                )
            out_batch["action"][key] = self._apply_backward(x, key)
        return out_batch
