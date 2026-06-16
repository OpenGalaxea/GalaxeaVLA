# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import Dict, List, Optional

import torch
from torch.nn.functional import pad

from g05.data_processor import BaseActionStateTransform
from g05.tokenizer.utils.parts_meta_utils import compute_grouped_layout, compute_grouped_dims


class ConcatLeftAlign(BaseActionStateTransform):
    """
    Concatenates action/state from dict format to a single tensor with left-aligned padding.

    Forward: Dict[key, Tensor] -> Tensor (concatenation + left-aligned padding)
    Backward: Tensor -> Dict[key, Tensor] (cropping + splitting)

    This transform is invertible.
    """

    invertible = True

    def __init__(
        self,
        action_target_dim: int | None = None,
        state_target_dim: int | None = None,
        **kwargs,
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        """Set shape_meta to determine concatenation order and dimensions"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """Dict → tensor: concatenate all keys and pad to target dimension"""
        if "action" in batch:
            # Concatenate action dict to tensor [action_size, sum(shape)]
            batch["action"] = self._concat(batch["action"], self.action_meta)
            if "action_op_mask" in batch:
                batch["action_op_mask"] = self._concat(batch["action_op_mask"], self.action_meta)
                batch["action_op_mask"], _ = self._pad(
                    batch["action_op_mask"], self.action_target_dim
                )

            # Pad to target_dim and generate padding mask [action_size, target_dim]
            batch["action"], batch["action_dim_is_pad"] = self._pad(
                batch["action"], self.action_target_dim
            )

        # Process state in the same way
        if "state" in batch:
            batch["state"] = self._concat(batch["state"], self.state_meta)
            batch["state"], batch["proprio_dim_is_pad"] = self._pad(
                batch["state"], self.state_target_dim
            )

        return batch

    def backward(self, batch):
        """Tensor → dict: crop padding and split back to original keys"""
        # Verify dimensions and crop state
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)

        # Verify dimensions and crop action
        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)

        # Synchronously process action_op_mask
        if "action_op_mask" in batch:
            batch["action_op_mask"] = self._crop(batch["action_op_mask"], self.action_meta)
            batch["action_op_mask"] = self._split(batch["action_op_mask"], self.action_meta)

        return batch

    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        """Pad right to specified dimension, returns padded tensor and padding mask.

        Supports both 2D (T, D) for action and 1D (D,) for action_op_mask.
        """
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim in (1, 2) and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        if x.ndim == 2:
            mask = torch.zeros_like(x[0]).bool()
            mask = pad(mask, (0, pad_dim), value=True)
        else:
            mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
            mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        """Crop padding portion, keep dimensions defined by meta.

        Supports 3D (B, T, D) for action and 2D (B, D) for action_op_mask.
        """
        assert x.ndim in (2, 3)
        dim = sum([m["shape"] for m in meta])
        if x.ndim == 3:
            x = x[:, :, :dim]
        else:
            x = x[:, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        """Concatenate tensors in dict according to meta order.

        Supports both 2D (T, D) for action and 1D (D,) for action_op_mask.
        """
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim in (1, 2)
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        """Split tensor back to dict according to shapes defined in meta.

        Supports 3D (B, T, D) for action and 2D (B, D) for action_op_mask.
        """
        assert x.ndim in (2, 3)
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            if x.ndim == 3:
                y[key] = x[:, :, idx : idx + dim]
            else:
                y[key] = x[:, idx : idx + dim]
            idx += dim

        return y


class DummyActionStateMerger(BaseActionStateTransform):
    """
    No-op merger that performs no data transformation.

    Forward/backward are identity operations, directly returning input batch.

    This transform is invertible (identity operation).
    """

    invertible = True

    def __init__(self, action_target_dim: int | None = None, state_target_dim: int | None = None):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        """Receives shape_meta but does not use it"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """Identity operation, directly returns batch"""
        return batch

    def backward(self, batch):
        """Identity operation, directly returns batch"""
        return batch

    # Methods below are for interface compatibility only, not called
    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        """(Unused) Pad to specified dimension"""
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        mask = torch.zeros_like(x[0]).bool()
        mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        """(Unused) Crop padding portion"""
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        """(Unused) Concatenate tensors in dict"""
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        """(Unused) Split tensor back to dict"""
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx : idx + dim]
            idx += dim

        return y


class PaddingActionMerger(BaseActionStateTransform):
    """
    Aligns action/state dicts from different embodiments (robot morphologies) for batching.

    Forward aligns to max_shape, backward restores to embodiment original shape.
    This transform is invertible.
    """

    invertible = True

    def __init__(
        self,
        max_action_shape_meta: Dict[str, int] | None = None,
        max_state_shape_meta: Dict[str, int] | None = None,
        merge: bool = False,
        **kwargs,
    ):
        # max_shape_meta format: {"left_arm": 6, "right_arm": 6, "gripper": 1, ...}
        self.max_action_shape_meta = max_action_shape_meta
        self.max_state_shape_meta = max_state_shape_meta
        self.merge = merge

    def set_shape_meta(self, shape_meta):
        """Set current embodiment's shape_meta for backward restoration"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """Align to max_shape: unify keys and dimensions, optionally merge to tensor"""
        if self.max_action_shape_meta is not None:
            if "action" in batch:
                # Align action and action_op_mask
                has_op_mask = "action_op_mask" in batch
                batch["action"], aligned_op_mask, action_padding_info = self._align_dict(
                    batch["action"], batch.get("action_op_mask", {}), self.max_action_shape_meta
                )
                if has_op_mask:
                    batch["action_op_mask"] = aligned_op_mask

                # Optionally merge aligned dict to single tensor
                if self.merge:
                    batch["action"], batch["action_dim_is_pad"] = self._concat_aligned_dict(
                        batch["action"], action_padding_info, self.max_action_shape_meta
                    )
                    # Also merge action_op_mask to tensor format
                    if has_op_mask:
                        batch["action_op_mask"], _ = self._concat_aligned_dict(
                            batch["action_op_mask"], {}, self.max_action_shape_meta
                        )

            if "gt_action" in batch:
                # Ground truth action also needs alignment (e.g., target during training)
                batch["gt_action"], _, gt_action_padding_info = self._align_dict(
                    batch["gt_action"], {}, self.max_action_shape_meta
                )

                # Optionally merge gt_action
                if self.merge:
                    batch["gt_action"], _ = self._concat_aligned_dict(
                        batch["gt_action"], gt_action_padding_info, self.max_action_shape_meta
                    )

        if self.max_state_shape_meta is not None and "state" in batch:
            # Align state
            batch["state"], _, state_padding_info = self._align_dict(
                batch["state"], {}, self.max_state_shape_meta
            )

            # Optionally merge state
            if self.merge:
                batch["state"], batch["proprio_dim_is_pad"] = self._concat_aligned_dict(
                    batch["state"], state_padding_info, self.max_state_shape_meta
                )

        return batch

    def backward(self, batch):
        """Restore to this embodiment's original keys and dimensions"""
        if self.max_action_shape_meta is not None:
            if "action" in batch:
                # If merged, split tensor back to dict first
                if self.merge:
                    batch["action"] = self._split_aligned_dict(
                        batch["action"], self.max_action_shape_meta
                    )

                # Restore to embodiment's original shape
                batch["action"] = self._restore_dict(batch["action"], self.action_meta)

            if "action_op_mask" in batch:
                # If merged, split action_op_mask tensor back to dict first
                if self.merge:
                    batch["action_op_mask"] = self._split_aligned_dict(
                        batch["action_op_mask"], self.max_action_shape_meta
                    )

                # Restore action_op_mask to embodiment's original shape
                batch["action_op_mask"] = self._restore_dict(
                    batch["action_op_mask"], self.action_meta
                )

            if "gt_action" in batch:
                if self.merge:
                    batch["gt_action"] = self._split_aligned_dict(
                        batch["gt_action"], self.max_action_shape_meta
                    )
                batch["gt_action"] = self._restore_dict(batch["gt_action"], self.action_meta)

        if self.max_state_shape_meta is not None and "state" in batch:
            if self.merge:
                batch["state"] = self._split_aligned_dict(batch["state"], self.max_state_shape_meta)
            batch["state"] = self._restore_dict(batch["state"], self.state_meta)

        return batch

    def _align_dict(
        self,
        data_dict: Dict[str, torch.Tensor],
        mask_dict: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """
        Align dict keys and dimensions to max_shape_meta.

        Processing logic:
        1. key exists + insufficient dimension → pad to target_dim
        2. key exists + exceeds dimension → truncate to target_dim
        3. key doesn't exist → create zero tensor + all-False mask

        Returns:
            aligned_data: Aligned data dict
            aligned_mask: Aligned mask dict (action_op_mask)
            padding_info: Dict indicating which dimensions are padded due to alignment
        """
        if not data_dict:
            return data_dict, mask_dict, {}

        # Get temporal dimension h and tensor properties
        h = next(iter(data_dict.values())).shape[0]
        device = next(iter(data_dict.values())).device
        dtype = next(iter(data_dict.values())).dtype

        aligned_data = {}
        aligned_mask = {}
        padding_info = {}

        for key, target_dim in max_shape_meta.items():
            if key in data_dict:
                # Case 1-2: key exists, need to align dimensions
                current_data = data_dict[key]  # (h, current_dim)
                current_dim = current_data.shape[-1]

                if current_dim < target_dim:
                    # Case 1: insufficient dimension, pad
                    pad_size = target_dim - current_dim
                    aligned_data[key] = torch.nn.functional.pad(current_data, (0, pad_size))

                    # Padding info: original dims are False, padded dims are True
                    padding_info[key] = torch.cat(
                        [
                            torch.zeros(current_dim, dtype=torch.bool, device=device),
                            torch.ones(pad_size, dtype=torch.bool, device=device),
                        ]
                    )

                    # Mask handling: extended dims don't exist for this embodiment → always False
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (current_dim,)
                        aligned_mask[key] = torch.nn.functional.pad(
                            current_mask, (0, pad_size), value=False
                        )
                    else:
                        # No mask, default to all True (indicates all dimensions are valid)
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

                elif current_dim > target_dim:
                    # Case 2: exceeds dimension, truncate
                    aligned_data[key] = current_data[..., :target_dim]

                    # Padding info: no padding (all truncated to fit)
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                    # Synchronously truncate mask
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (current_dim,)
                        aligned_mask[key] = current_mask[..., :target_dim]
                    else:
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

                else:
                    # Case 3: same dimension, use directly
                    aligned_data[key] = current_data

                    # Padding info: no padding
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                    # But mask dimension might differ, need independent alignment
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (mask_dim,)
                        mask_dim = current_mask.shape[-1]

                        if mask_dim < target_dim:
                            # Mask dimension insufficient, pad; extended dims don't exist → False
                            aligned_mask[key] = torch.nn.functional.pad(
                                current_mask, (0, target_dim - mask_dim), value=False
                            )
                        elif mask_dim > target_dim:
                            # Mask dimension exceeds, truncate
                            aligned_mask[key] = current_mask[..., :target_dim]
                        else:
                            # Mask dimension also same
                            aligned_mask[key] = current_mask
                    else:
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)
            else:
                # Case 4: key doesn't exist, create virtual tensor
                # Zero data + all-False mask (indicates this embodiment doesn't have this component)
                aligned_data[key] = torch.zeros((h, target_dim), dtype=dtype, device=device)
                aligned_mask[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                # Padding info: all dimensions are padded (virtual key)
                padding_info[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

        return aligned_data, aligned_mask, padding_info

    def _restore_dict(self, aligned_dict: Dict[str, torch.Tensor], meta: List[Dict]):
        """Restore to original keys and dimensions, discard padding and virtual keys"""
        restored = {}
        for m in meta:
            key = m["key"]
            original_dim = m["shape"]
            if key in aligned_dict:
                # Crop to original dimension
                restored[key] = aligned_dict[key][..., :original_dim]
        return restored

    def _concat_aligned_dict(
        self,
        aligned_dict: Dict[str, torch.Tensor],
        padding_info: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """
        Concatenate aligned dict into a single tensor following max_shape_meta key order.

        Args:
            aligned_dict: Aligned data dict with unified keys and dimensions
            padding_info: Dict indicating which dimensions are padded due to alignment
            max_shape_meta: Defines key order and dimensions

        Returns:
            concatenated_tensor: Single tensor [temporal_dim, sum(max_shape_meta.values())]
            dim_is_pad: Boolean mask indicating which dimensions are padded [sum(max_shape_meta.values())]
                        True = padded due to alignment (virtual key or extended dimension)
                        False = original data from embodiment
        """
        if not aligned_dict:
            return aligned_dict, None

        # Concatenate in max_shape_meta key order
        tensors = []
        padding_masks = []
        for key in max_shape_meta.keys():
            assert key in aligned_dict, f"Key '{key}' missing from aligned_dict"
            tensors.append(aligned_dict[key])

            # Use padding_info to determine which dimensions are padded
            if key in padding_info:
                padding_masks.append(padding_info[key])
            else:
                # No padding info, assume all dimensions are original (not padded)
                dim = aligned_dict[key].shape[-1]
                device = aligned_dict[key].device
                padding_masks.append(torch.zeros(dim, dtype=torch.bool, device=device))

        # Concatenate along last dimension
        concatenated = torch.cat(tensors, dim=-1)  # [temporal_dim, sum(dims)]

        # Concatenate padding masks
        dim_is_pad = torch.cat(padding_masks, dim=-1)  # [sum(dims)], True = padded

        return concatenated, dim_is_pad

    def _split_aligned_dict(
        self, concatenated_tensor: torch.Tensor, max_shape_meta: Dict[str, int]
    ):
        """
        Split concatenated tensor back to aligned dict following max_shape_meta key order.

        Args:
            concatenated_tensor: Concatenated tensor [batch, temporal_dim, sum(dims)] or [temporal_dim, sum(dims)]
            max_shape_meta: Defines key order and dimensions for splitting

        Returns:
            aligned_dict: Dict with keys from max_shape_meta
        """
        # Handle both 2D [temporal, dim] and 3D [batch, temporal, dim]
        assert concatenated_tensor.ndim in (2, 3)

        aligned_dict = {}
        idx = 0
        for key, dim in max_shape_meta.items():
            aligned_dict[key] = concatenated_tensor[..., idx : idx + dim]
            idx += dim

        return aligned_dict


class GroupedPaddingMerger(BaseActionStateTransform):
    """
    Merges action/state dict to a compact flat tensor using two-level merge_spec.

    merge_spec format: {replacement_name: [raw_key, ...]}
    For each replacement_name, selects the first present (non-virtual) alternative key,
    pads to max(alternative_dims), and concatenates.
    Keys not in any merge_spec group are appended as residuals in parts_meta order.

    Example with dual_arm_grouped merge_spec:
        left_control: left_arm(8) OR left_ee_pose(9) → 9D
        left_gripper: left_gripper → 1D
        right_control: right_arm(8) OR right_ee_pose(9) → 9D
        right_gripper: right_gripper → 1D
        Total flat tensor = 20D

    If no merge_spec is configured, no mutual exclusion is applied;
    all keys are simply concatenated in parts_meta order.

    Forward:  Dict[raw_key, Tensor] → Tensor [T, total_grouped_dim]
    Backward: Tensor → Dict[raw_key, Tensor], restored to per-embodiment original dims

    This transform is invertible.
    """

    invertible = True

    def __init__(
        self,
        max_action_shape_meta: Optional[Dict[str, int]] = None,
        max_state_shape_meta: Optional[Dict[str, int]] = None,
        merge_spec: Optional[Dict] = None,
        merge: bool = True,
        **kwargs,
    ):
        self._raw_max_action_shape_meta = max_action_shape_meta
        self._raw_max_state_shape_meta = max_state_shape_meta
        self.merge_spec = merge_spec
        self.merge = merge

        self._action_layout, self._action_residual_keys = self._precompute(
            max_action_shape_meta, merge_spec
        )
        self._state_layout, self._state_residual_keys = self._precompute(
            max_state_shape_meta, merge_spec
        )

        if max_action_shape_meta is not None and merge_spec is not None:
            self.max_action_shape_meta = compute_grouped_dims(max_action_shape_meta, merge_spec)
        else:
            self.max_action_shape_meta = max_action_shape_meta

        if max_state_shape_meta is not None and merge_spec is not None:
            self.max_state_shape_meta = compute_grouped_dims(max_state_shape_meta, merge_spec)
        else:
            self.max_state_shape_meta = max_state_shape_meta

    @staticmethod
    def _precompute(shape_meta, merge_spec):
        if shape_meta is None or merge_spec is None:
            return None, []
        layout = compute_grouped_layout(shape_meta, merge_spec)
        merged_keys = set()
        for group in layout:
            merged_keys.update(group.part_names)
        residual_keys = [k for k in shape_meta if k not in merged_keys]
        return layout, residual_keys

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if self._raw_max_action_shape_meta is not None:
            if "action" in batch:
                has_op_mask = "action_op_mask" in batch
                batch["action"], batch["action_dim_is_pad"] = self._forward_one(
                    batch["action"],
                    self._raw_max_action_shape_meta,
                    self._action_layout,
                    self._action_residual_keys,
                )
                if has_op_mask:
                    batch["action_op_mask"], _ = self._forward_one(
                        batch["action_op_mask"],
                        self._raw_max_action_shape_meta,
                        self._action_layout,
                        self._action_residual_keys,
                    )
            if "gt_action" in batch:
                batch["gt_action"], _ = self._forward_one(
                    batch["gt_action"],
                    self._raw_max_action_shape_meta,
                    self._action_layout,
                    self._action_residual_keys,
                )

        if self._raw_max_state_shape_meta is not None and "state" in batch:
            batch["state"], batch["proprio_dim_is_pad"] = self._forward_one(
                batch["state"],
                self._raw_max_state_shape_meta,
                self._state_layout,
                self._state_residual_keys,
            )

        return batch

    def backward(self, batch):
        if self._raw_max_action_shape_meta is not None:
            if "action" in batch:
                batch["action"] = self._backward_one(
                    batch["action"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )
            if "action_op_mask" in batch:
                batch["action_op_mask"] = self._backward_one(
                    batch["action_op_mask"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )
            if "gt_action" in batch:
                batch["gt_action"] = self._backward_one(
                    batch["gt_action"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )

        if self._raw_max_state_shape_meta is not None and "state" in batch:
            batch["state"] = self._backward_one(
                batch["state"],
                self._state_layout,
                self._state_residual_keys,
                self.state_meta,
                self._raw_max_state_shape_meta,
            )

        return batch

    def _align_per_raw_key(
        self,
        data_dict: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """Align dict values to max_shape_meta dims (pad / truncate / create virtual zeros).

        Supports both 2D (h, D) values (action/state) and 1D (D,) values (action_op_mask).
        Virtual keys are created with the same ndim as existing keys.
        """
        if not data_dict:
            return {}, {}

        sample_val = next(iter(data_dict.values()))
        device, dtype = sample_val.device, sample_val.dtype
        is_1d = sample_val.ndim == 1
        if not is_1d:
            h = sample_val.shape[0]

        aligned, padding_info = {}, {}
        for key, target_dim in max_shape_meta.items():
            if key in data_dict:
                t = data_dict[key]
                d = t.shape[-1]
                if d < target_dim:
                    pad_size = target_dim - d
                    aligned[key] = pad(t, (0, pad_size))
                    padding_info[key] = torch.cat(
                        [
                            torch.zeros(d, dtype=torch.bool, device=device),
                            torch.ones(pad_size, dtype=torch.bool, device=device),
                        ]
                    )
                elif d > target_dim:
                    aligned[key] = t[..., :target_dim]
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)
                else:
                    aligned[key] = t
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)
            else:
                if is_1d:
                    aligned[key] = torch.zeros(target_dim, dtype=dtype, device=device)
                else:
                    aligned[key] = torch.zeros((h, target_dim), dtype=dtype, device=device)
                padding_info[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

        return aligned, padding_info

    def _forward_one(self, data_dict, max_shape_meta, layout, residual_keys):
        """Align raw keys, apply merge_spec grouping, concat to flat tensor."""
        aligned, padding_info = self._align_per_raw_key(data_dict, max_shape_meta)
        device = next(iter(aligned.values())).device

        tensors, pad_masks = [], []

        if layout is None:
            for key in max_shape_meta:
                tensors.append(aligned[key])
                pad_masks.append(padding_info[key])
        else:
            for group in layout:
                # First non-virtual alternative wins (alternatives order = priority)
                chosen = next(
                    (p for p in group.part_names if not padding_info[p].all()),
                    group.part_names[0],
                )
                t = aligned[chosen]
                is_pad = padding_info[chosen].clone()

                # chosen's raw max_dim may be < group.max_dim; pad the difference
                chosen_max_dim = max_shape_meta[chosen]
                if chosen_max_dim < group.max_dim:
                    extra = group.max_dim - chosen_max_dim
                    t = pad(t, (0, extra))
                    is_pad = torch.cat(
                        [
                            is_pad,
                            torch.ones(extra, dtype=torch.bool, device=device),
                        ]
                    )

                tensors.append(t)
                pad_masks.append(is_pad)

        for key in residual_keys:
            tensors.append(aligned[key])
            pad_masks.append(padding_info[key])

        flat = torch.cat(tensors, dim=-1)
        dim_is_pad = torch.cat(pad_masks)
        return flat, dim_is_pad

    def _backward_one(self, flat_tensor, layout, residual_keys, embodiment_meta, max_shape_meta):
        """Split flat tensor back to per-embodiment raw key dict."""
        assert flat_tensor.ndim in (2, 3)
        result = {}
        idx = 0

        for group in layout:
            slot_t = flat_tensor[..., idx : idx + group.max_dim]
            idx += group.max_dim
            for m in embodiment_meta:
                if m["key"] in group.part_names:
                    result[m["key"]] = slot_t[..., : m["shape"]]
                    break

        for key in residual_keys:
            dim = max_shape_meta[key]
            t = flat_tensor[..., idx : idx + dim]
            idx += dim
            for m in embodiment_meta:
                if m["key"] == key:
                    result[key] = t[..., : m["shape"]]
                    break

        return result
