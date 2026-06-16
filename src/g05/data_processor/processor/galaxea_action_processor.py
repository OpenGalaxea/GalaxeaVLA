# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import Dict, Any, Optional, Literal, List

import torch

from g05.data_processor.transforms.action_filter import BaseActionFilter


class R1LiteJointActionFilter(BaseActionFilter):
    """
    Per-dimension action filter for Galaxea robots.

    Each dimension is independently judged as active/stationary using its own
    threshold.  Thresholds are resolved with priority:
        1. dim_thresholds[key] → per-dim vector
        2. type-level fallback (joint/gripper/eef/velocity_threshold) → scalar broadcast

    Control Scheme Assumptions:
        - arm / eef / gripper: delta from first frame (absolute position control)
        - torso / chassis: absolute value (velocity control)
    """

    _VELOCITY_KEYS = ("torso", "chassis")

    def forward(self, batch):
        if "action" not in batch:
            return batch

        action_op_mask = {}
        for meta in self.action_meta:
            k, meta_shape = meta["key"], meta["raw_shape"]
            action = batch["action"][k]  # (H, D)
            actual_shape = action.shape[-1]
            half = action.shape[0] // 2

            threshold = self._resolve_threshold(k, actual_shape, action.device)

            if "hand" in k:
                flag = torch.ones(actual_shape, dtype=torch.bool, device=action.device)
            elif any(vk in k for vk in self._VELOCITY_KEYS):
                # velocity semantics: absolute value is meaningful
                deviation = torch.abs(action[:half])  # (half, D)
                flag = (deviation >= threshold).any(dim=0)  # (D,)
            else:
                # arm / eef / gripper: delta from first frame
                deviation = torch.abs(action[:half] - action[:1])  # (half, D)
                flag = (deviation >= threshold).any(dim=0)  # (D,)

            action_op_mask[k] = flag  # (D,)
            assert actual_shape == meta_shape, (
                f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
            )

        batch["action_op_mask"] = action_op_mask
        return batch

    def backward(self, batch):
        if "action_op_mask" not in batch:
            return batch
        if "action" in batch:
            for meta in self.action_meta:
                k = meta["key"]
                assert k in batch["action"], f"Missing action key in backward filter: {k}"
                assert k in batch["action_op_mask"], (
                    f"Missing action_op_mask key in backward filter: {k}"
                )
                batch["action"][k] = torch.where(batch["action_op_mask"][k], batch["action"][k], 0)
        return batch

    def _resolve_threshold(self, key: str, dim: int, device) -> torch.Tensor:
        """Resolve threshold for a key: per-dim vector or type-level scalar."""
        if key in self.dim_thresholds:
            t = torch.tensor(self.dim_thresholds[key], dtype=torch.float32, device=device)
            if t.shape[0] != dim:
                raise ValueError(f"dim_thresholds['{key}'] has {t.shape[0]} values, expected {dim}")
            return t

        # type-level fallback
        if "gripper" in key:
            val = self.gripper_threshold
        elif "eef" in key or "ee_pose" in key:
            val = self.eef_threshold
        elif any(vk in key for vk in self._VELOCITY_KEYS):
            val = self.velocity_threshold
        else:
            val = self.joint_threshold

        return torch.tensor(val or 0.0, dtype=torch.float32, device=device)
