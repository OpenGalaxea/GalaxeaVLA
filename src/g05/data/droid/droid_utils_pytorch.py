"""
DROID Dataset Utils - PyTorch version

Coordinate transforms and rotation representation utilities.
"""

import torch


def euler_to_rmat(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrices.
    Uses XYZ order, matching tensorflow_graphics.

    Args:
        euler: (..., 3) Euler angle tensor [roll, pitch, yaw].

    Returns:
        (..., 3, 3) rotation matrices.
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    r00 = cos_y * cos_p
    r01 = cos_y * sin_p * sin_r - sin_y * cos_r
    r02 = cos_y * sin_p * cos_r + sin_y * sin_r
    r10 = sin_y * cos_p
    r11 = sin_y * sin_p * sin_r + cos_y * cos_r
    r12 = sin_y * sin_p * cos_r - cos_y * sin_r
    r20 = -sin_p
    r21 = cos_p * sin_r
    r22 = cos_p * cos_r

    rmat = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return rmat


def rmat_to_euler(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles.

    Args:
        rot_mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw].
    """
    pitch = -torch.asin(torch.clamp(rot_mat[..., 2, 0], -1.0, 1.0))
    roll = torch.atan2(rot_mat[..., 2, 1], rot_mat[..., 2, 2])
    yaw = torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])

    return torch.stack([roll, pitch, yaw], dim=-1)


def invert_rmat(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Invert rotation matrices, i.e. transpose them.

    Args:
        rot_mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3, 3) inverse rotation matrices.
    """
    return rot_mat.transpose(-1, -2)


def rotmat_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to R6 rotation representation: the first two rows.

    Args:
        mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) 6D rotation vectors.
    """
    r6 = mat[..., :2, :]  # (..., 2, 3)
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]  # (..., 3), (..., 3)
    r6_flat = torch.cat([r6_0, r6_1], dim=-1)  # (..., 6)
    return r6_flat


def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert R6 rotation representation back to rotation matrices with Gram-Schmidt orthogonalization.

    Args:
        rot6d: (..., 6) 6D rotation vectors.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Gram-Schmidt orthogonalization.
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


def velocity_act_to_wrist_frame(
    velocity: torch.Tensor,
    wrist_in_robot_frame: torch.Tensor
) -> torch.Tensor:
    """
    Transform velocity actions (translation + rotation) from robot base frame to wrist frame.

    Args:
        velocity: (batch, 6) velocity actions (3 x translation, 3 x rotation).
        wrist_in_robot_frame: (batch, 6) end-effector pose in robot base frame.

    Returns:
        (batch, 9) velocity actions in robot wrist frame (3 x translation, 6 x R6 rotation).
    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3].unsqueeze(-1)).squeeze(-1)

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)

    return torch.cat([vel_t, dR_r6], dim=-1)


# ===============================================================================
# DROID-specific constants.
# ===============================================================================

# DROID dataset normalization parameters: 1%/99% quantiles for Cartesian velocity.
DROID_Q01 = torch.tensor([
    -0.7776297926902771,
    -0.5803514122962952,
    -0.5795090794563293,
    -0.6464047729969025,
    -0.7041108310222626,
    -0.8895104378461838,
])

DROID_Q99 = torch.tensor([
    0.7597932070493698,
    0.5726242214441299,
    0.7351000607013702,
    0.6705610305070877,
    0.6464948207139969,
    0.8897542208433151,
])
