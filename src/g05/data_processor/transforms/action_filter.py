# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import torch

from g05.data_processor import BaseActionStateTransform


class BaseActionFilter(BaseActionStateTransform):
    """
    Base class for action filters that mark operational dimensions.

    Action filters are NOT invertible - they modify action_op_mask which
    cannot be recovered from the action values alone.
    """

    invertible = False

    def __init__(
        self,
        joint_threshold: float | None = None,
        gripper_threshold: float | None = None,
        velocity_threshold: float | None = None,
        eef_threshold: float | None = 1e-3,
        dim_thresholds: dict | None = None,
    ):
        self.joint_threshold = joint_threshold
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
        self.eef_threshold = eef_threshold
        self.dim_thresholds = dim_thresholds or {}

    def set_shape_meta(self, shape_meta):
        processed_action_meta, processed_state_meta = [], []
        for meta in shape_meta["action"]:
            if meta["key"] is not None:
                processed_action_meta.append(meta)
        for meta in shape_meta["state"]:
            if meta["key"] is not None:
                processed_state_meta.append(meta)
        self.action_meta = processed_action_meta
        self.state_meta = processed_state_meta

    def forward(self, batch):
        if "action" not in batch:
            return batch

        action_op_mask = {}
        for meta in self.action_meta:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["action"][k].shape[-1]
            flag = torch.ones(actual_shape, dtype=torch.bool)
            action_op_mask[k] = flag
        batch["action_op_mask"] = action_op_mask  # A
        return batch

    def backward(self, batch):
        return batch


class DummyActionFilter(BaseActionFilter):
    """
    Action filter that marks all dimensions as operational (no masking).

    Sets action_op_mask to all True, equivalent to R1LiteJointActionFilter
    with all thresholds set to 0.
    """

    invertible = True

    def forward(self, batch):
        return super().forward(batch)

    def backward(self, batch):
        return batch
