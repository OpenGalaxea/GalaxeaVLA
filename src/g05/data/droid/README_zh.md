# DROID Dataset Module

DROID 数据集的 LeRobot 格式加载器。

## 架构设计

DROID 全部特殊性封装在 `DroidLerobotDataset` 内，processor 端零知识 — 跨 emb
混 batch 时与 r1lite/r1pro/agibot 等走完全相同的 `GalaxeaCoTProcessor` 路径。

```
DroidLerobotDataset.__getitem__:
  super (image_meta 已注入 cam2_left 用于加载)
    └─ 读 cam1 / cam2 / wrist (uint8 [T,3,H,W])
  _swap_exterior_images:  训练 50% cam1 ← cam2, drop cam2
  _inject_dummy_images:   按 shape_meta 中 dummy:true 注入 zero tensors
  _pick_lang_alternative: 3-choose-1 over (task, lang2, lang3)
  processor.preprocess (通用 GalaxeaCoTProcessor)
```

## 文件结构

```
droid/
├── __init__.py              # 模块入口
├── droid_lerobot_dataset.py # LeRobot 格式数据加载器（含全部 Droid-specific 行为）
├── droid_utils_pytorch.py   # 坐标变换工具
└── README.md                # 本文档
```

## 配置使用

processor 端只需声明模型实际看到的相机 — aux exterior 由 dataset 自己注入到
`image_meta` 用于加载，不暴露给 processor。

```yaml
# configs/data/droid.yaml
processor:
  # 不写 _target_, 走 task config 默认 (通常 GalaxeaCoTProcessor)
  shape_meta:
    images:
      - key: exterior_image     # 主 exterior (训练时可能被 cam2 替换)
        camera_type: exterior
        lerobot_key: observation.images.exterior_1_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: wrist_image
        camera_type: wrist_left
        lerobot_key: observation.images.wrist_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: dummy_wrist_right  # 可选: 让 pixel_values 键集与 3-cam emb 对齐
        camera_type: wrist_right
        dummy: true             # dataset 看到此标记会注入 zeros, 不读 lerobot
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

## DROID-specific 行为

| 行为 | 实现位置 |
|---|---|
| Gripper 翻转 (`1 - x`) | `_slice_meta_feature` |
| 2-choose-1 exterior swap | `__getitem__` → `_swap_exterior_images` |
| 3-choose-1 lang augmentation | `__getitem__` → `_pick_lang_alternative` |
| dummy zero-image 注入 | `__getitem__` → `_inject_dummy_images` |
| 失败轨迹过滤 | `_build_valid_local_indices` (`filter_failed_trajectories`) |
| Idle 帧过滤 | `R1LiteJointActionFilter` (在 processor.action_filter 配置) |

## 列名映射

| sample key | LeRobot 列名 |
|----------|--------------|
| `exterior_image` (主, dataset 内部 swap 后) | `observation.images.exterior_1_left` |
| `exterior_image_2` (dataset 自己加载, swap 后丢弃) | `observation.images.exterior_2_left` |
| `wrist_image` | `observation.images.wrist_left` |
| `joint_position` (state) | `observation.state.joint_position` |
| `gripper` (state) | `observation.state.gripper_position` |

## 坐标变换工具

```python
from g05.data.droid import (
    euler_to_rmat,
    rmat_to_euler,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    velocity_act_to_wrist_frame,
)
```
