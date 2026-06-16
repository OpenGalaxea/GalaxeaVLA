# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import List, Dict

import torch
from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)
from copy import deepcopy
from g05.data_processor import BaseActionStateTransform

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


# JIT-compiled functions for fast quaternion operations
@torch.jit.script
def quaternion_conjugate_jit(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion in (i, j, k, r) format."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


@torch.jit.script
def quaternion_multiply_jit(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in (i, j, k, r) format."""
    i1, j1, k1, r1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    i2, j2, k2, r2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

    return torch.stack([i, j, k, r], dim=-1)


@torch.jit.script
def quaternion_rotate_vector_jit(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q in (i, j, k, r) format."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    # t = 2 * cross(q.xyz, v)
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)

    # v' = v + q.w * t + cross(q.xyz, t)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)

    return torch.stack([rx, ry, rz], dim=-1)


class RelativePoseTransform(BaseActionStateTransform):
    """
    Convert absolute poses to relative poses w.r.t. the last state.

    Forward: pose_relative = T_base^{-1} @ pose_absolute
    Backward: pose_absolute = T_base @ pose_relative

    This transform is invertible.
    """

    invertible = True

    def __init__(
        self,
        keys: List[str],
        input_rotation_type: str = "quaternion",
        euler_convention: str = "XYZ",
        fast_forward: bool = True,
    ):
        self.keys = keys
        self.input_rotation_type = input_rotation_type
        self.euler_convention = euler_convention
        self.fast_forward = fast_forward

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch

        # Stack all keys along new leading dim for batched matrix ops:
        # (K, ..., 7) → batched _pose_to_matrix, inv, @, _matrix_to_pose
        action_stacked = torch.stack([batch["action"][k] for k in self.keys])
        state_stacked = torch.stack([batch["state"][k] for k in self.keys])

        if self.fast_forward:
            result = self._forward_fast(action_stacked, state_stacked[..., -1:, :])
        else:
            result = self._forward(action_stacked, state_stacked[..., -1:, :])

        out_batch = deepcopy(batch)
        for i, k in enumerate(self.keys):
            out_batch["action"][k] = result[i]

        return out_batch

    def backward(self, batch: Dict):
        action_stacked = torch.stack([batch["action"][k] for k in self.keys])
        state_stacked = torch.stack([batch["state"][k] for k in self.keys])

        if self.fast_forward:
            result = self._backward_fast(action_stacked, state_stacked[..., -1:, :])
        else:
            result = self._backward(action_stacked, state_stacked[..., -1:, :])

        out_batch = deepcopy(batch)
        for i, k in enumerate(self.keys):
            out_batch["action"][k] = result[i]

        return out_batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0:3]
        quaternion = pose[..., [6, 3, 4, 5]]  # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0:3, 0:3] = rotation
        matrix[..., 0:3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0:3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0:3, 0:3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]]  # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose

    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix

    @staticmethod
    def _quaternion_conjugate(q: torch.Tensor):
        """Conjugate of quaternion in (i, j, k, r) format."""
        return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    @staticmethod
    def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
        """Multiply two quaternions in (i, j, k, r) format."""
        i1, j1, k1, r1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        i2, j2, k2, r2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
        i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
        j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
        k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

        return torch.stack([i, j, k, r], dim=-1)

    @staticmethod
    def _quaternion_rotate_vector(q: torch.Tensor, v: torch.Tensor):
        """Rotate vector v by quaternion q in (i, j, k, r) format."""
        i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        x, y, z = v[..., 0], v[..., 1], v[..., 2]

        # Using the formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        # This is more efficient than converting to matrix
        qx, qy, qz = i, j, k
        qw = r

        # t = 2 * cross(q.xyz, v)
        tx = 2 * (qy * z - qz * y)
        ty = 2 * (qz * x - qx * z)
        tz = 2 * (qx * y - qy * x)

        # v' = v + q.w * t + cross(q.xyz, t)
        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)

        return torch.stack([rx, ry, rz], dim=-1)

    def _forward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """Optimized forward with JIT-compiled quaternion operations."""
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )

        # Extract position and quaternion (i, j, k, r format)
        p_target = pose[..., :3]
        q_target = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # Compute relative pose with JIT functions
        # p_rel = R_base^T * (p_target - p_base)
        q_base_conj = quaternion_conjugate_jit(q_base)
        delta_p = p_target - p_base
        p_rel = quaternion_rotate_vector_jit(q_base_conj, delta_p)

        # q_rel = q_base^{-1} * q_target
        q_rel = quaternion_multiply_jit(q_base_conj, q_target)

        return torch.cat([p_rel, q_rel], dim=-1)

    def _backward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """Optimized backward with JIT-compiled quaternion operations."""
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )

        # Extract position and quaternion (i, j, k, r format)
        p_rel = pose[..., :3]
        q_rel = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # Compute absolute pose with JIT functions
        # p_target = R_base * p_rel + p_base
        p_target = quaternion_rotate_vector_jit(q_base, p_rel) + p_base

        # q_target = q_base * q_rel
        q_target = quaternion_multiply_jit(q_base, q_rel)

        return torch.cat([p_target, q_target], dim=-1)

    def _forward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """Optimized forward for euler poses using 3x3 tensor ops."""
        assert pose.shape[-1] == 6, f"Pose shape must be (..., 6), but got {pose.shape}"
        assert base_pose.shape[-1] == 6, (
            f"Base pose shape must be (..., 6), but got {base_pose.shape}"
        )

        p_target = pose[..., :3]
        e_target = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        r_target = euler_angles_to_matrix(e_target, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)
        r_base_t = r_base.transpose(-1, -2)

        delta_p = (r_base_t @ (p_target - p_base).unsqueeze(-1)).squeeze(-1)
        r_rel = r_base_t @ r_target
        e_rel = matrix_to_euler_angles(r_rel, self.euler_convention)
        return torch.cat([delta_p, e_rel], dim=-1)

    def _backward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """Optimized backward for euler poses using 3x3 tensor ops."""
        assert pose.shape[-1] == 6, f"Pose shape must be (..., 6), but got {pose.shape}"
        assert base_pose.shape[-1] == 6, (
            f"Base pose shape must be (..., 6), but got {base_pose.shape}"
        )

        p_rel = pose[..., :3]
        e_rel = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        r_rel = euler_angles_to_matrix(e_rel, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)

        p_target = (r_base @ p_rel.unsqueeze(-1)).squeeze(-1) + p_base
        r_target = r_base @ r_rel
        e_target = matrix_to_euler_angles(r_target, self.euler_convention)
        return torch.cat([p_target, e_target], dim=-1)


class DeltaPoseTransform(BaseActionStateTransform):
    """
    Convert absolute poses to delta (frame-to-frame difference) poses.

    Forward:  delta[0] = action[0] - state[-1]
             delta[t] = action[t] - action[t-1]   for t >= 1
    Backward: action[0] = delta[0] + state[-1]
              action[t] = delta[t] + action[t-1]   for t >= 1  (cumulative sum)

    Simple element-wise difference on all dimensions of the specified keys.
    This transform is invertible.
    """

    invertible = True

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch

        out_batch = {
            "action": {},
            "state": batch["state"],
        }
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                action = batch["action"][k]
                state_last = batch["state"][k][..., -1:, :]
                first_delta = action[..., 0:1, :] - state_last
                if action.shape[-2] > 1:
                    rest_delta = action[..., 1:, :] - action[..., :-1, :]
                    out_batch["action"][k] = torch.cat([first_delta, rest_delta], dim=-2)
                else:
                    out_batch["action"][k] = first_delta
            elif k in batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch

    def backward(self, batch: Dict):
        if "action" not in batch:
            return batch

        out_batch = {
            "action": {},
            "state": batch["state"],
        }
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                delta = batch["action"][k]
                state_last = batch["state"][k][..., -1:, :]
                first_action = delta[..., 0:1, :] + state_last
                if delta.shape[-2] > 1:
                    rest_cumsum = torch.cumsum(delta[..., 1:, :], dim=-2)
                    out_batch["action"][k] = torch.cat(
                        [first_action, first_action + rest_cumsum], dim=-2
                    )
                else:
                    out_batch["action"][k] = first_action
            elif k in batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch


class RelativeJointTransform(BaseActionStateTransform):
    """
    Convert absolute joint positions to relative positions w.r.t. the last state.

    Forward: action_relative = action_absolute - state_last
    Backward: action_absolute = action_relative + state_last

    This transform is invertible.
    """

    invertible = True

    def __init__(self, keys: List[str], fast_forward: bool = True):
        self.keys = keys
        self.fast_forward = fast_forward

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch or "state" not in batch:
            return batch

        if self.fast_forward:
            return self._forward_fast(batch)
        else:
            return self._forward(batch)

    def _forward(self, batch: Dict):
        """Original implementation with deepcopy."""
        # Cat all keys along last dim for one batched subtraction
        # (keys may have different D, so cat+split instead of stack)
        action_parts = [batch["action"][k] for k in self.keys if k in batch["action"]]
        state_parts = [batch["state"][k] for k in self.keys if k in batch["state"]]
        dims = [t.shape[-1] for t in action_parts]

        if len(action_parts) == 0 or len(state_parts) == 0:
            return batch

        action_merged = torch.cat(action_parts, dim=-1)
        state_merged = torch.cat(state_parts, dim=-1)

        result = action_merged - state_merged[..., -1:, :]

        out_batch = deepcopy(batch)
        for k, t in zip(self.keys, result.split(dims, dim=-1)):
            out_batch["action"][k] = t

        return out_batch

    def _forward_fast(self, batch: Dict):
        """Optimized forward: avoid deepcopy, direct per-key operations."""
        # Shallow copy batch structure (only copy dict references, not tensors)
        out_batch = {
            "action": {},
            "state": batch["state"],  # Share state reference (not modified)
        }

        # Copy other keys if present (observation, etc.)
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        # Process keys that exist in both action and state
        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                # Direct per-key operation, no cat/split overhead
                out_batch["action"][k] = batch["action"][k] - batch["state"][k][..., -1:, :]
            elif k in batch["action"]:
                # Key not in state, keep original
                out_batch["action"][k] = batch["action"][k]

        # Copy remaining action keys not in self.keys
        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch

    def backward(self, batch: Dict):
        if "action" not in batch or "state" not in batch:
            return batch

        if self.fast_forward:
            return self._backward_fast(batch)
        else:
            return self._backward(batch)

    def _backward(self, batch: Dict):
        """Original implementation with deepcopy."""
        action_parts = [batch["action"][k] for k in self.keys if k in batch["action"]]
        state_parts = [batch["state"][k] for k in self.keys if k in batch["state"]]
        dims = [t.shape[-1] for t in action_parts]

        if len(action_parts) == 0 or len(state_parts) == 0:
            return batch

        action_merged = torch.cat(action_parts, dim=-1)
        state_merged = torch.cat(state_parts, dim=-1)

        result = action_merged + state_merged[..., -1:, :]

        out_batch = deepcopy(batch)
        for k, t in zip(self.keys, result.split(dims, dim=-1)):
            out_batch["action"][k] = t

        return out_batch

    def _backward_fast(self, batch: Dict):
        """Optimized backward: avoid deepcopy, direct per-key operations."""
        # Shallow copy batch structure
        out_batch = {
            "action": {},
            "state": batch["state"],  # Share state reference
        }

        # Copy other keys if present
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        # Process keys that exist in both action and state
        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                # Direct per-key operation
                out_batch["action"][k] = batch["action"][k] + batch["state"][k][..., -1:, :]
            elif k in batch["action"]:
                # Key not in state, keep original
                out_batch["action"][k] = batch["action"][k]

        # Copy remaining action keys not in self.keys
        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch


class ReorderLowerBodyTransform(BaseActionStateTransform):
    """Reorder lower_body dims from [base_qvel(3), trunk_qpos(4)] to [trunk_qpos(4), base_qvel(3)].

    Raw b1k action[0:7] is [base_qvel(3), trunk_qpos(4)].
    WBC convention (and cholesky matrix) uses [trunk_qpos(4), base_qvel(3)].
    This transform aligns the ordering.
    """

    invertible = True

    def __init__(self, key: str = "lower_body", trunk_dim: int = 4, base_dim: int = 3):
        self.key = key
        self.trunk_dim = trunk_dim
        self.base_dim = base_dim

    def forward(self, batch: Dict):
        for domain in ("action", "state"):
            if domain in batch and self.key in batch[domain]:
                t = batch[domain][self.key]
                # [base_qvel(3), trunk_qpos(4)] → [trunk_qpos(4), base_qvel(3)]
                batch[domain][self.key] = torch.cat(
                    [t[..., self.base_dim :], t[..., : self.base_dim]], dim=-1
                )
        return batch

    def backward(self, batch: Dict):
        for domain in ("action", "state"):
            if domain in batch and self.key in batch[domain]:
                t = batch[domain][self.key]
                # [trunk_qpos(4), base_qvel(3)] → [base_qvel(3), trunk_qpos(4)]
                batch[domain][self.key] = torch.cat(
                    [t[..., self.trunk_dim :], t[..., : self.trunk_dim]], dim=-1
                )
        return batch


class PartialRelativeTransform(BaseActionStateTransform):
    """Apply relative transform (action -= state) only on specified dims of a key.

    Used for trunk_qpos where only dims [0:3] should be relative while dim[3] stays absolute.
    After this transform, the specified action dims become increments relative to state.
    """

    invertible = True

    def __init__(self, key: str, relative_dims: List[int]):
        self.key = key
        self.relative_dims = relative_dims

    def forward(self, batch: Dict):
        if "action" not in batch or "state" not in batch:
            return batch
        if self.key not in batch["action"] or self.key not in batch["state"]:
            return batch
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        state_last = state[..., -1:, :] if state.ndim > action.ndim - 1 else state
        if state.ndim == action.ndim:
            state_last = state[..., -1:, :]
        for d in self.relative_dims:
            action[..., d] = action[..., d] - state_last[..., d]
        batch["action"][self.key] = action
        return batch

    def backward(self, batch: Dict):
        if "action" not in batch or "state" not in batch:
            return batch
        if self.key not in batch["action"] or self.key not in batch["state"]:
            return batch
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        state_last = state[..., -1:, :] if state.ndim > action.ndim - 1 else state
        if state.ndim == action.ndim:
            state_last = state[..., -1:, :]
        for d in self.relative_dims:
            action[..., d] = action[..., d] + state_last[..., d]
        batch["action"][self.key] = action
        return batch


class BehaviorPerKeyTransform(BaseActionStateTransform):
    """Per-key transform for b1k behavior data (compatible with per-key shape_meta).

    Input keys expected:
      action: {left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), lower_body(7)}
      state:  {left_arm(7), left_gripper(2), right_arm(7), right_gripper(2), trunk_qpos(4), base_qvel(3)}

    Operations:
      1. Gripper state: collapse 2D → 1D (sum → rescale to [-1,1])
      2. Relative: left_arm, right_arm (action -= state)
      3. Lower body reorder: action lower_body [base(3),trunk(4)] → [trunk(4),base(3)]
      4. Trunk partial relative: action lower_body dims[0:N] -= state trunk_qpos[0:N]
      5. Merge state trunk_qpos + base_qvel → state lower_body (remove originals)

    Output keys:
      action: {left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), lower_body(7)}
      state:  {left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), lower_body(7)}
    """

    invertible = True
    alters_key_structure = True

    def __init__(self, gripper_max_width: float = 0.1, trunk_relative_dims: List[int] = None):
        self.gripper_max_width = gripper_max_width
        self.trunk_relative_dims = trunk_relative_dims or [0, 1, 2]

    def forward(self, batch: Dict):
        if "state" not in batch:
            return batch

        state = batch["state"]
        action = batch.get("action", {})

        # 1. Gripper collapse 2D → 1D (both action and state)
        for gk in ("left_gripper", "right_gripper"):
            for domain in (state, action):
                if gk in domain and domain[gk].shape[-1] == 2:
                    width = domain[gk].sum(dim=-1, keepdim=True)
                    domain[gk] = 2.0 * (width / self.gripper_max_width) - 1.0

        # 2. Merge action trunk_qpos + base_qvel → lower_body [trunk(4), base(3)]
        if "trunk_qpos" in action and "base_qvel" in action:
            action["lower_body"] = torch.cat([action["trunk_qpos"], action["base_qvel"]], dim=-1)
            del action["trunk_qpos"]
            del action["base_qvel"]

        # 3. Relative: left_arm, right_arm (action -= state_last)
        for k in ("left_arm", "right_arm"):
            if k in action and k in state:
                state_last = state[k][..., -1:, :] if state[k].ndim == action[k].ndim else state[k]
                action[k] = action[k] - state_last

        # 4. Trunk partial relative on action lower_body
        if "lower_body" in action and "trunk_qpos" in state:
            trunk_state = state["trunk_qpos"]
            trunk_state_last = (
                trunk_state[..., -1:, :]
                if trunk_state.ndim == action["lower_body"].ndim
                else trunk_state
            )
            for d in self.trunk_relative_dims:
                action["lower_body"][..., d] = (
                    action["lower_body"][..., d] - trunk_state_last[..., d]
                )

        # 5. Merge state: trunk_qpos + base_qvel → lower_body, remove originals
        if "trunk_qpos" in state and "base_qvel" in state:
            state["lower_body"] = torch.cat([state["trunk_qpos"], state["base_qvel"]], dim=-1)
            del state["trunk_qpos"]
            del state["base_qvel"]

        batch["state"] = state
        if action:
            batch["action"] = action
        return batch

    def backward(self, batch: Dict):
        state = batch["state"]
        action = batch.get("action", {})

        # Reverse 5: split state lower_body → trunk_qpos + base_qvel
        if "lower_body" in state:
            state["trunk_qpos"] = state["lower_body"][..., :4]
            state["base_qvel"] = state["lower_body"][..., 4:]

        # Reverse 4: undo trunk partial relative
        if "lower_body" in action and "trunk_qpos" in state:
            trunk_state = state["trunk_qpos"]
            trunk_state_last = (
                trunk_state[..., -1:, :]
                if trunk_state.ndim == action["lower_body"].ndim
                else trunk_state
            )
            for d in self.trunk_relative_dims:
                action["lower_body"][..., d] = (
                    action["lower_body"][..., d] + trunk_state_last[..., d]
                )

        # Reverse 3: reorder action lower_body back: [trunk(4), base(3)] → [base(3), trunk(4)]
        if "lower_body" in action:
            lb = action["lower_body"]
            action["lower_body"] = torch.cat([lb[..., 4:], lb[..., :4]], dim=-1)

        # Reverse 2: undo relative
        for k in ("left_arm", "right_arm"):
            if k in action and k in state:
                state_last = state[k][..., -1:, :] if state[k].ndim == action[k].ndim else state[k]
                action[k] = action[k] + state_last

        # Reverse 1: un-collapse gripper (not truly invertible, expand to 2D equal split)
        for gk in ("left_gripper", "right_gripper"):
            if gk in state and state[gk].shape[-1] == 1:
                width = (state[gk] + 1.0) * self.gripper_max_width / 2.0
                state[gk] = torch.cat([width / 2, width / 2], dim=-1)

        # Remove merged lower_body from state, keep trunk_qpos + base_qvel
        if "lower_body" in state and "trunk_qpos" in state:
            del state["lower_body"]

        batch["state"] = state
        if action:
            batch["action"] = action
        return batch
