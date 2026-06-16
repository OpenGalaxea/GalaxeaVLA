# 数据 Schema

> 统一 `shape_meta` 定义 action/state/images 维度，驱动所有数据加载。

## 1. Schema 结构

```yaml
shape_meta:
  action:
    - key: left_arm
      lerobot_key: action
      start_index: 0
      raw_shape: 6
      shape: 6
      time_offset: 0
    - key: left_gripper
      lerobot_key: action
      start_index: 6
      raw_shape: 1
      shape: 1
      time_offset: 0

  state:
    - key: left_arm
      lerobot_key: observation.state
      start_index: 0
      raw_shape: 6
      shape: 6
      time_offset: 0

  images:
    - key: head_rgb
      lerobot_key: observation.images.head_rgb
      start_index: 0
      raw_shape: [3, 720, 1280]
      shape: [3, 224, 224]
      time_offset: 0
```

## 2. 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | str | 内部 key（训练使用） |
| `lerobot_key` | str | parquet 原始列名 |
| `start_index` | int | 从原始列第几维开始切 |
| `raw_shape` | int/list | 变换前维度 |
| `shape` | int/list | 变换后维度 |
| `time_offset` | int | 时间偏移（默认 0） |

### raw_shape vs shape

- `raw_shape`: 从 parquet 切出来的维度
- `shape`: 经过 `action_state_transforms` 后的维度

**通常两者相同**，只有做维度变换的 transform 才不同。

### time_offset

- `state/images`: `(time_offset + step) / fps`
- `action`: `(time_offset + step) / fps`

`time_offset=1` 用于构造 `t+1` target（state-as-action）。

## 3. 同列切分

多个 part 共享一个 `lerobot_key` 时，靠 `start_index + raw_shape` 显式切片：

```yaml
# 原始 action 10 维，取最后 7 维
action:
  - key: arm
    lerobot_key: action
    start_index: 3
    raw_shape: 6
    shape: 6
  - key: gripper
    lerobot_key: action
    start_index: 9
    raw_shape: 1
    shape: 1
```

**禁止** `key: null` 占位写法，直接用 `start_index` 跳过不需要的维度。

## 4. 数据集版本

| 版本 | 类 | 说明 |
|------|-----|------|
| v2.1 | `BaseLerobotDataset` | 逐 episode 迭代计算 stats |
| v3.0 | `BaseLerobotDatasetV3` | pyarrow 批量读取，快 10-50× |

**v3.0 推荐用于新数据集**。

## 5. 新增 Embodiment 步骤

### 5.1 创建 Config

```yaml
# configs/data/<emb>/pretrain.yaml
MyDataset:
  type: g05.data.base_lerobot_datasetV3.BaseLerobotDatasetV3
  embodiment_type: my_robot
  lerobot_ds_version: "3.0"
  action_size: 32
  obs_size: 1

  shape_meta: &shape_def
    action: [...]
    state: [...]
    images: [...]

  dataset_groups:
    - weight: 1.0
      dataset_dirs:
        - /path/to/lerobot/dataset

processor:
  shape_meta: *shape_def
  train_transforms:
    head_rgb: ${oc.load:configs/data/_transforms.yaml,train_head}
  norm_default_mode: "z-score"
```

### 5.2 添加到 Data Config

```yaml
# configs/data/<your_task>.yaml
embodiment_datasets:
  my_robot:
    type: ...
    shape_meta: ...
processors:
  my_robot:
    shape_meta: ...
    action_state_merger: ...
```

## 6. State-as-Action

将 action 的 `lerobot_key` 指到 observation/state 列：

```yaml
action:
  - key: left_arm
    lerobot_key: observation.state.left_arm
    start_index: 0
    raw_shape: 6
    shape: 6
    time_offset: 1  # t+1
```

## 7. 常见错误

| 错误 | 原因 |
|------|------|
| `shape_meta.images` key 与 `train_transforms` 不一致 | key 必须匹配 |
| 新增 part 未更新 `action_state_merger.max_*_shape_meta` | merger 输出维度或 mask 会出错 |
| `shape` 未随 transform 修改 | processor 断言失败 |
| `time_offset` 取到未来帧 | 检查偏移值 |

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| [samples_builders.md](samples_builders.md) | SamplesBuilder 字段依赖与模板 |
| [../architecture/parts_meta.md](../architecture/parts_meta.md) | 按 embodiment 配置的动作空间定义 |

---
*最后修改: 2026.03.09*
