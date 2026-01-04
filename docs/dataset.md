# Galaxea LeRobot Dataset Format

This repository uses a dataset format developed based on the LeRobot v2.1 dataset, with structure and fields adapted to `BaseLerobotDataset`. The following describes the format using the current data as an example.

### Directory Structure

```
data/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
  videos/
    chunk-000/
      observation.images.head_rgb/
        episode_000000.mp4
      observation.images.head_right_rgb/
        episode_000000.mp4
      observation.images.left_wrist_rgb/
        episode_000000.mp4
      observation.images.right_wrist_rgb/
        episode_000000.mp4
```

### meta Metadata

- `meta/info.json`: Dataset-level information and feature descriptions. Key fields include:
  - `codebase_version`: Version number (e.g., `v2.1`)
  - `robot_type`: Robot model (e.g., `r1lite`)
  - `fps`: Sampling frame rate (e.g., `15`)
  - `data_path`: Parquet path template (`data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`)
  - `video_path`: Video path template (`videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`)
  - `features`: Field definitions (dtype/shape/names), the authoritative source for data parsing
- `meta/tasks.jsonl`: Task index table, mapping `task_index` -> `task` text. Task text may use the format `Chinese@English` for processor language switching.
- `meta/episodes.jsonl`: Summary information for each episode, including:
  - `episode_index`: Episode number
  - `tasks`: Set of task texts involved in this episode (from `tasks.jsonl`)
  - `length`: Number of frames
  - `raw_file_name`: Original data file name
- `meta/episodes_stats.jsonl`: Statistics for each episode (min/max/mean/std/count), used for normalization or data checking. These statistics are raw representations and will be recalculated according to configuration at training startup.

### Data Files (parquet)

- Each episode corresponds to a parquet file, with the number of rows equal to the episode's `length`.
- All **non-video** fields are stored in parquet; video frames are stored as mp4 files under `videos/`.
- Main fields are as follows (refer to `meta/info.json`):

**Observation**
- Video streams (stored in `videos/`):  
  `observation.images.head_rgb` / `head_right_rgb` / `left_wrist_rgb` / `right_wrist_rgb`  
  Shape `(720, 1280, 3)`, fps=15, encoded as av1.
- State variables (float64):  
  `observation.state.left_arm` (6), `left_arm.velocities` (6)  
  `observation.state.right_arm` (6), `right_arm.velocities` (6)  
  `observation.state.chassis` (3), `chassis.velocities` (3), `chassis.imu` (10)  
  `observation.state.torso` (4), `torso.velocities` (4)  
  `observation.state.left_gripper` (1), `right_gripper` (1)  
  `observation.state.left_ee_pose` (7), `right_ee_pose` (7)

**Action (float64)**
- `action.left_arm` (6), `action.right_arm` (6)  
- `action.left_gripper` (1), `action.right_gripper` (1)  
- `action.torso.velocities` (6), `action.chassis.velocities` (6)

**Index and Annotation**
- `timestamp` (float32, seconds)
- `frame_index` / `episode_index` / `index` (int64)
- `coarse_task_index` / `task_index` / `coarse_quality_index` / `quality_index` (int64)

### Video Files (mp4)

Video paths follow the `video_path` template in `meta/info.json`, for example:

```
videos/chunk-000/observation.images.head_rgb/episode_000123.mp4
```

Here, `{video_key}` corresponds to the `observation.images.*` keys in `features`. When loading, use indices such as `task_index` to map back to the task text (see `meta/tasks.jsonl`).
