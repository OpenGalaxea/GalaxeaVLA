"""DROID Dataset Module.

Provides a LeRobot-format loader for DROID datasets.

All DROID-specific behavior is encapsulated in DroidLerobotDataset, keeping the
processor side unaware of it:
- gripper flip (1 - x): handled in _slice_meta_feature
- 2-choose-1 exterior swap: handled by _swap_exterior_images in __getitem__
- 3-choose-1 language augmentation: _pick_lang_alternative
- dummy zero images: injected by _inject_dummy_images for shape_meta entries marked dummy: true
- failed trajectory / idle-frame filtering: filter_failed_trajectories + DroidIdleActionFilter
"""

from g05.data.droid.droid_lerobot_dataset import (
    DroidLerobotDataset,
    DroidActionSpace,
    create_droid_dataset,
)
from g05.data.droid.droid_utils_pytorch import (
    euler_to_rmat,
    rmat_to_euler,
    invert_rmat,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    velocity_act_to_wrist_frame,
    DROID_Q01,
    DROID_Q99,
)

__all__ = [
    # Dataset
    "DroidLerobotDataset",
    "DroidActionSpace",
    "create_droid_dataset",
    # Utils - Rotation
    "euler_to_rmat",
    "rmat_to_euler",
    "invert_rmat",
    "rotmat_to_rot6d",
    "rot6d_to_rotmat",
    "velocity_act_to_wrist_frame",
    # Constants
    "DROID_Q01",
    "DROID_Q99",
]
