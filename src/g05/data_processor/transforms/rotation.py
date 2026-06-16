# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is adapted from PyTorch3D rotation utilities.
# See THIRD_PARTY_NOTICES.md for the upstream BSD-style license notice.

from typing import Literal, List
import torch
import torch.nn.functional as F

from g05.utils.data.rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_rotation_9d,
    rotation_9d_to_matrix,
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    quaternion_to_rotation_6d_jit,
    quaternion_to_rotation_9d_jit,
    rotation_6d_to_quaternion_jit,
)
from g05.data_processor import BaseActionStateTransform
from copy import deepcopy


class PoseRotationTransform(BaseActionStateTransform):
    """
    Transform pose rotation representations between quaternion/euler and rotation_6d/rotation_9d.

    Forward: input_rotation_type → rotation_6d/rotation_9d
    Backward: rotation_6d/rotation_9d → input_rotation_type

    Input pose layout:
      - 7D (quaternion): [x, y, z, quat_x, quat_y, quat_z, quat_w]
      - 6D (euler): [x, y, z, euler_x, euler_y, euler_z]

    Output pose layout:
      - 9D (rotation_6d): [x, y, z, rotation_6d]
      - 12D (rotation_9d): [x, y, z, rotation_9d]

    Note: Gripper should be stored in a separate key (e.g., left_gripper), not in ee_pose.
    This transform is invertible.
    """

    invertible = True

    def __init__(
        self,
        rotation_type: Literal["quaternion", "rotation_6d", "rotation_9d"],
        category_keys: List[str],
        input_rotation_type: Literal["quaternion", "euler"] = "quaternion",
        euler_convention: str = "XYZ",
        fast_forward: bool = True,
    ):
        self.rotation_type = rotation_type
        self.category_keys = category_keys
        self.input_rotation_type = input_rotation_type
        self.euler_convention = euler_convention
        self.fast_forward = fast_forward

    def forward(self, batch):
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            if cat == "action" and "action" not in out_batch:
                continue

            for k in ks:
                if self.fast_forward:
                    out_batch[cat][k] = self._forward_fast(out_batch[cat][k])
                else:
                    out_batch[cat][k] = self._forward(out_batch[cat][k])

                if (
                    cat == "action"
                    and "action_op_mask" in out_batch
                    and k in out_batch["action_op_mask"]
                ):
                    out_batch["action_op_mask"][k] = self._forward_mask(
                        out_batch["action_op_mask"][k]
                    )

        return out_batch

    def backward(self, batch):
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            for k in ks:
                if self.fast_forward and out_batch[cat][k].ndim <= 2:
                    out_batch[cat][k] = self._backward_fast(out_batch[cat][k])
                else:
                    out_batch[cat][k] = self._backward(out_batch[cat][k])

                if (
                    cat == "action"
                    and "action_op_mask" in out_batch
                    and k in out_batch["action_op_mask"]
                ):
                    out_batch["action_op_mask"][k] = self._backward_mask(
                        out_batch["action_op_mask"][k]
                    )

        return out_batch

    # (total_pose_dim, rot_dim) for each rotation representation
    _ROT_DIMS = {
        "quaternion": (7, 4),
        "euler": (6, 3),
        "rotation_6d": (9, 6),
        "rotation_9d": (12, 9),
    }

    def _remap_mask(self, mask: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        """Remap op_mask between rotation representations, preserving pos and collapsing rot."""
        if src_type == dst_type:
            return mask

        in_dim, _ = self._ROT_DIMS[src_type]
        out_dim, out_rot_dim = self._ROT_DIMS[dst_type]

        if mask.shape[-1] < in_dim:
            mask = F.pad(mask, (0, in_dim - mask.shape[-1]), value=False)
        elif mask.shape[-1] > in_dim:
            mask = mask[..., :in_dim]

        pos_mask = mask[..., :3]
        rot_active = mask[..., 3:].any(dim=-1, keepdim=True)
        rot_out = rot_active.expand(*rot_active.shape[:-1], out_rot_dim)
        out = torch.cat([pos_mask, rot_out], dim=-1)
        assert out.shape[-1] == out_dim
        return out

    def _forward_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self._remap_mask(mask, self.input_rotation_type, self.rotation_type)

    def _backward_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self._remap_mask(mask, self.rotation_type, self.input_rotation_type)

    def _forward(self, pose):
        """
        Convert pose from input_rotation_type to rotation_type.

        Args:
            pose: (..., 6) or (..., 7)
                  6D (euler): [x, y, z, euler_x, euler_y, euler_z]
                  7D (quaternion): [x, y, z, quat_x, quat_y, quat_z, quat_w]

        Returns:
            rotated_pose: (..., 9) for rotation_6d or (..., 12) for rotation_9d
        """
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        if self.rotation_type == "quaternion":
            return pose

        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            quaternion = pose[..., [6, 3, 4, 5]]  # (x,y,z,w) -> (w,x,y,z)
            matrix = quaternion_to_matrix(quaternion)
        elif self.input_rotation_type == "euler":
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

        if self.rotation_type == "rotation_6d":
            rotation = matrix_to_rotation_6d(matrix)
        elif self.rotation_type == "rotation_9d":
            rotation = matrix_to_rotation_9d(matrix)
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        return torch.cat([position, rotation], dim=-1)

    def _backward(self, pose: torch.Tensor):
        """
        Inverse transformation: convert from rotation_type to input_rotation_type.

        Args:
            pose: (..., 9) for rotation_6d or (..., 12) for rotation_9d

        Returns:
            original_pose: (..., 6) for euler or (..., 7) for quaternion
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9  # 3 + 6
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12  # 3 + 9
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            matrix = rotation_6d_to_matrix(rotation)
        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

        if self.input_rotation_type == "quaternion":
            quaternion = matrix_to_quaternion(matrix)
            quaternion = quaternion[..., [1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
            return torch.cat([position, quaternion], dim=-1)
        elif self.input_rotation_type == "euler":
            euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
            return torch.cat([position, euler_angles], dim=-1)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def add_noise(self, pose: torch.Tensor, std_position=0.05, std_angle=0.05):
        """Add noise to 7D quaternion pose."""
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        position = pose[..., 0:3]
        quaternion = pose[..., [6, 3, 4, 5]]
        axis_angles = quaternion_to_axis_angle(quaternion)
        position = position + std_position * torch.randn_like(position)
        axis_angles = axis_angles + std_angle * torch.randn_like(axis_angles)
        quaternion = axis_angle_to_quaternion(axis_angles)
        quaternion = quaternion[..., [1, 2, 3, 0]]
        return torch.cat([position, quaternion], dim=-1)

    def _forward_fast(self, pose: torch.Tensor):
        """
        Optimized forward with JIT-compiled rotation conversions.
        For euler input, falls back to non-JIT path.
        """
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        if self.rotation_type == "quaternion":
            return pose

        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            quaternion = pose[..., [6, 3, 4, 5]]  # (x,y,z,w) -> (w,x,y,z)

            if self.rotation_type == "rotation_6d":
                rotation = quaternion_to_rotation_6d_jit(quaternion)
            elif self.rotation_type == "rotation_9d":
                rotation = quaternion_to_rotation_9d_jit(quaternion)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        elif self.input_rotation_type == "euler":
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)

            if self.rotation_type == "rotation_6d":
                rotation = matrix_to_rotation_6d(matrix)
            elif self.rotation_type == "rotation_9d":
                rotation = matrix_to_rotation_9d(matrix)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def _backward_fast(self, pose: torch.Tensor):
        """
        Optimized backward with JIT-compiled rotation conversions.
        For euler input, falls back to non-JIT path.
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12
        else:
            raise NotImplementedError

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            quaternion = rotation_6d_to_quaternion_jit(rotation)

            if self.input_rotation_type == "quaternion":
                quaternion = quaternion[..., [1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                matrix = quaternion_to_matrix(quaternion)
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

            if self.input_rotation_type == "quaternion":
                quaternion = matrix_to_quaternion(matrix)
                quaternion = quaternion[..., [1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")


class EulerPoseToQuaternionTransform(BaseActionStateTransform):
    """
    Convert Euler-angle pose format to quaternion format before RelativePoseTransform.

    Forward:  [..., 6]  [x, y, z, euler_x, euler_y, euler_z]
           → [..., 7]  [x, y, z, quat_x, quat_y, quat_z, quat_w]

    Backward: [..., 7] → [..., 6]

    Use case: OXE Euler-angle datasets should convert to quaternions before
    RelativePoseTransform, preserving geometric correctness of relative pose
    computation instead of using approximate elementwise differences.

    This transform is invertible.
    """

    invertible = True

    def __init__(self, category_keys: dict, euler_convention: str = "XYZ"):
        self.category_keys = category_keys
        self.euler_convention = euler_convention

    def forward(self, batch: dict) -> dict:
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            if cat == "action" and "action" not in out_batch:
                continue
            for k in ks:
                if k in out_batch.get(cat, {}):
                    out_batch[cat][k] = self._euler_to_quat(out_batch[cat][k])
        return out_batch

    def backward(self, batch: dict) -> dict:
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            for k in ks:
                if k in out_batch.get(cat, {}):
                    out_batch[cat][k] = self._quat_to_euler(out_batch[cat][k])
        return out_batch

    def _euler_to_quat(self, pose: torch.Tensor) -> torch.Tensor:
        """[x,y,z,ex,ey,ez] → [x,y,z,qx,qy,qz,qw]"""
        assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"
        position = pose[..., :3]
        euler = pose[..., 3:6]
        matrix = euler_angles_to_matrix(euler, self.euler_convention)
        quat_wxyz = matrix_to_quaternion(matrix)  # pytorch3d: (w,x,y,z)
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]  # -> (x,y,z,w), RelativePoseTransform format
        return torch.cat([position, quat_xyzw], dim=-1)

    def _quat_to_euler(self, pose: torch.Tensor) -> torch.Tensor:
        """[x,y,z,qx,qy,qz,qw] → [x,y,z,ex,ey,ez]"""
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        position = pose[..., :3]
        quat_xyzw = pose[..., 3:7]
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]  # (x,y,z,w) → (w,x,y,z)
        matrix = quaternion_to_matrix(quat_wxyz)
        euler = matrix_to_euler_angles(matrix, self.euler_convention)
        return torch.cat([position, euler], dim=-1)
