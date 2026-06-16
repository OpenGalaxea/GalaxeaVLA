# DROID Dataset Module

LeRobot-format loader for the DROID dataset.

## Architecture

All DROID-specific behavior is encapsulated in `DroidLerobotDataset`. The processor remains unaware of these details. Mixed batches with r1lite, r1pro, agibot, and other embodiments follow the same `GalaxeaCoTProcessor` path.

```text
DroidLerobotDataset.__getitem__:
  super, with image_meta injected with cam2_left for loading
    └─ read cam1 / cam2 / wrist as uint8 [T,3,H,W]
  _swap_exterior_images: 50% training-time cam1 <- cam2, then drop cam2
  _inject_dummy_images: inject zero tensors according to dummy:true in shape_meta
  _pick_lang_alternative: 3-way choice over task, lang2, and lang3
  processor.preprocess using the generic GalaxeaCoTProcessor
```

## File Layout

```text
droid/
├── __init__.py              # module entry
├── droid_lerobot_dataset.py # LeRobot-format data loader with all Droid-specific behavior
├── droid_utils_pytorch.py   # coordinate transform utilities
└── README.md                # this document
```

## Configuration

The processor only declares cameras the model actually sees. Auxiliary exterior cameras are injected into `image_meta` by the dataset for loading and are not exposed to the processor.

```yaml
# configs/data/droid.yaml
processor:
  # Omit _target_ and use the task config default, usually GalaxeaCoTProcessor.
  shape_meta:
    images:
      - key: exterior_image     # primary exterior camera; may be replaced by cam2 during training
        camera_type: exterior
        lerobot_key: observation.images.exterior_1_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: wrist_image
        camera_type: wrist_left
        lerobot_key: observation.images.wrist_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: dummy_wrist_right  # optional: align pixel_values keys with 3-camera embodiments
        camera_type: wrist_right
        dummy: true             # dataset injects zeros and does not read lerobot for this key
        raw_shape: [3, 256, 256]
        shape: [3, 256, 160]

Droid_Franka:
  type: g05.data.droid.droid_lerobot_dataset.DroidLerobotDataset
  random_swap_exterior_images: true
  random_select_instruction: true
  filter_failed_trajectories: true
  shape_meta: ${processor.shape_meta}
  ...
```

## DROID-Specific Behavior

| Behavior | Implementation |
|----------|----------------|
| Gripper inversion, `1 - x` | `_slice_meta_feature` |
| 2-way exterior swap | `__getitem__` -> `_swap_exterior_images` |
| 3-way language augmentation | `__getitem__` -> `_pick_lang_alternative` |
| Dummy zero-image injection | `__getitem__` -> `_inject_dummy_images` |
| Failed trajectory filtering | `_build_valid_local_indices` with `filter_failed_trajectories` |
| Idle frame filtering | `R1LiteJointActionFilter` in `processor.action_filter` config |

## Column Mapping

| Sample key | LeRobot column |
|------------|----------------|
| `exterior_image`, primary camera after internal dataset swap | `observation.images.exterior_1_left` |
| `exterior_image_2`, loaded internally by the dataset and dropped after swap | `observation.images.exterior_2_left` |
| `wrist_image` | `observation.images.wrist_left` |
| `joint_position` as state | `observation.state.joint_position` |
| `gripper` as state | `observation.state.gripper_position` |

## Coordinate Transform Utilities

```python
from g05.data.droid import (
    euler_to_rmat,
    rmat_to_euler,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    velocity_act_to_wrist_frame,
)
```
