# Data Schema

> `shape_meta` is the shared definition of action, state, and image dimensions. It drives all data loading.

## 1. Schema Structure

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

## 2. Fields

| Field | Type | Description |
|-------|------|-------------|
| `key` | str | Internal key used during training. |
| `lerobot_key` | str | Raw parquet column name. |
| `start_index` | int | Start dimension when slicing from the raw column. |
| `raw_shape` | int/list | Dimension before transforms. |
| `shape` | int/list | Dimension after transforms. |
| `time_offset` | int | Time offset; default is 0. |

### `raw_shape` vs `shape`

- `raw_shape`: dimension sliced from parquet.
- `shape`: dimension after `action_state_transforms`.

They are usually identical. They differ only when a transform changes the dimension.

### `time_offset`

- `state/images`: `(time_offset + step) / fps`
- `action`: `(time_offset + step) / fps`

Use `time_offset=1` to construct a `t+1` target, also called state-as-action.

## 3. Slicing Shared Columns

When multiple parts share one `lerobot_key`, use `start_index + raw_shape` for explicit slicing:

```yaml
# Raw action has 10 dims; use the last 7 dims.
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

Do not use `key: null` placeholders. Skip unused dimensions directly with `start_index`.

## 4. Dataset Versions

| Version | Class | Description |
|---------|-------|-------------|
| v2.1 | `BaseLerobotDataset` | Iterates by episode to compute stats. |
| v3.0 | `BaseLerobotDatasetV3` | Uses pyarrow batch reads and is 10-50x faster. |

Use v3.0 for new datasets.

## 5. Add A New Embodiment

### 5.1 Create A Config

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

### 5.2 Add It To A Data Config

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

## 6. State-As-Action

Point an action entry's `lerobot_key` to an observation/state column:

```yaml
action:
  - key: left_arm
    lerobot_key: observation.state.left_arm
    start_index: 0
    raw_shape: 6
    shape: 6
    time_offset: 1  # t+1
```

## 7. Common Errors

| Error | Cause |
|-------|-------|
| `shape_meta.images` key does not match `train_transforms` | Keys must match. |
| New part was added but `action_state_merger.max_*_shape_meta` was not updated | Merger output dimension or mask will be wrong. |
| `shape` was not updated after a transform changed dimensions | Processor assertions fail. |
| `time_offset` reaches a future frame | Check the offset value. |

## 8. Related Documents

| Document | Contents |
|----------|----------|
| [samples_builders.md](samples_builders.md) | SamplesBuilder field dependencies and templates. |
| [../architecture/parts_meta.md](../architecture/parts_meta.md) | Embodiment-specific action space definitions. |

---
Last modified: 2026-03-09
