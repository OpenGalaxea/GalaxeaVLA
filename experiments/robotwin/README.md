# RoboTwin Evaluation

This directory contains the evaluation scripts for running GalaxeaVLA on the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) simulation benchmark. It supports single-task evaluation and multi-task scheduling across multiple GPUs.

## Contents

| File | Purpose |
|------|---------|
| `run_robotwin_manager.py` | Multi-task evaluation manager: reads the task list, distributes tasks across GPUs, and aggregates results. |
| `eval_robotwin_single.py` | Single-task Hydra entry point, spawned by the manager as a subprocess. |
| `setup_robotwin_venv.sh` | Creates `.venv-robotwin`, syncs project dependencies into it, installs RoboTwin simulator dependencies, and verifies imports. |
| `galaxeafm_policy/deploy_policy.py` | Policy adapter: wraps GalaxeaVLA inference into the RoboTwin policy interface (`get_model` / `eval` / `reset_model`). |
| `galaxeafm_policy/deploy_policy.yml` | Default policy parameters such as action horizon, replan steps, precision, and dataset statistics path. |

## Prerequisites

> Install `uv` as described in the repository root [README.md](../../README.md). All commands below are run from the project root unless a command explicitly changes directory.

> RoboTwin uses `.venv-robotwin`, separate from the main `.venv`. Avoid `uv run python` for RoboTwin evaluation because it follows `pyproject.toml` / `uv.lock` and can replace RoboTwin's `sapien` 3.x with the main environment's `sapien` 2.x.

### 1. Clone RoboTwin

Clone the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) repository with submodules into `third_party/`:

```bash
git clone --recursive https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
```

If the repository was cloned without submodules, initialize them:

```bash
cd third_party/RoboTwin
git submodule update --init --recursive
cd -
```

The expected RoboTwin root is the directory that directly contains `script/eval_policy.py` and `task_config/_eval_step_limit.yml`:

```bash
test -f third_party/RoboTwin/script/eval_policy.py
test -f third_party/RoboTwin/task_config/_eval_step_limit.yml
```

If your RoboTwin checkout uses a different layout, pass that directory through `EVALUATION.robotwin_root=...` in the evaluation commands.

### 2. Download Assets

Download the RoboTwin assets from the RoboTwin root:

```bash
cd third_party/RoboTwin
bash script/_download_assets.sh
cd -
```

If assets already exist on shared storage, link the whole assets directory instead:

```bash
ln -sfn /absolute/path/to/RoboTwin/assets third_party/RoboTwin/assets
```

If direct Hugging Face access is unavailable, download the assets from a mirror:

```bash
mkdir -p third_party/RoboTwin/assets
cd third_party/RoboTwin/assets
wget https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/main/embodiments.zip
wget https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/main/background_texture.zip
wget https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/main/objects.zip
unzip -o embodiments.zip
unzip -o background_texture.zip
unzip -o objects.zip
cd -
```

### 3. Create the RoboTwin Virtual Environment

Create a dedicated `.venv-robotwin` environment:

```bash
bash experiments/robotwin/setup_robotwin_venv.sh
```

The script syncs the project dependencies into `.venv-robotwin`, then installs the RoboTwin-specific dependencies:

```text
sapien==3.0.0b1
warp-lang==0.11.0
nvidia-curobo from third_party/RoboTwin/envs/curobo
```

If `third_party/RoboTwin/envs/curobo` is missing, the script clones the validated `v0.7.5` source there before installation. The PyPI package named `curobo` is not the NVIDIA cuRobo library, and cuRobo v0.8.x is incompatible with RoboTwin 2.0.

Set `ROBOTWIN_VENV`, `ROBOTWIN_ROOT`, `PYTHON_VERSION`, or `CUROBO_VERSION` before running the script only if you need non-default paths or versions.

### 4. Generate cuRobo Config Files

RoboTwin embodiment assets include template files with `${ASSETS_PATH}` placeholders. Generate the concrete `curobo_left.yml` and `curobo_right.yml` files once after assets are in place:

```bash
cd third_party/RoboTwin
../../.venv-robotwin/bin/python script/update_embodiment_config_path.py
cd -
```

This step writes generated config files next to the templates under `third_party/RoboTwin/assets/embodiments/`. If `assets` is a read-only shared-storage symlink, verify the generated `curobo*.yml` files already exist there; otherwise use a writable local assets copy and rerun the command.

### 5. Verify Rendering

Run the RoboTwin render verification from the RoboTwin root:

```bash
cd third_party/RoboTwin
../../.venv-robotwin/bin/python script/test_render.py
cd -
```

The render check should finish with `Render Well`. Fix rendering before running policy evaluation.

## Checkpoint Layout

Download the shared processor and action tokenizer as described in the root README. If the RoboTwin checkpoint is present in the Hugging Face repository snapshot, download the RoboTwin subset with:

```bash
huggingface-cli download OpenGalaxea/G05 \
    --repo-type model \
    --local-dir checkpoints \
    --local-dir-use-symlinks False \
    --include "g05-robotwin20/*" "qwen3_5_2b_base_processor/*" "action_tokenizer.pt"
```

If `g05-robotwin20` is distributed separately from the public Hugging Face snapshot, place the files under the same local layout before running evaluation.

Expected local layout:

```text
checkpoints/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
└── g05-robotwin20/
    ├── .hydra/config.yaml
    ├── checkpoints/model_state_dict.pt
    └── dataset_stats.json
```

For `checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt`, the policy adapter loads model and tokenizer configuration from `checkpoints/g05-robotwin20/.hydra/config.yaml`. That config must reference the shared processor at `checkpoints/qwen3_5_2b_base_processor` and the shared action tokenizer at `checkpoints/action_tokenizer.pt`.

`dataset_stats.json` is searched from checkpoint parent directories. With the standard layout above, no explicit `EVALUATION.dataset_stats_path` override is needed. Pass it only when the stats file lives outside the checkpoint parent directories.

## Usage

### Full Benchmark (All Tasks)

Run all tasks from RoboTwin's `_eval_step_limit.yml`, distributed across GPUs:

```bash
.venv-robotwin/bin/python -u experiments/robotwin/run_robotwin_manager.py \
  task=robotwin \
  ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

Each task runs two phases sequentially: `clean` for deterministic initial placement and `random` for randomized initial placement. By default, each phase runs 50 episodes.

### Single Task Evaluation

Evaluate a specific task with the standard episode count:

```bash
.venv-robotwin/bin/python -u experiments/robotwin/run_robotwin_manager.py \
  task=robotwin \
  ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=50 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

## Parameters

### Required

| Parameter | Description |
|-----------|-------------|
| `task` | Hydra task config name, usually `robotwin`. |
| `ckpt` | Path to the model checkpoint file. |
| `EVALUATION.robotwin_root` | RoboTwin root containing `script/eval_policy.py`. |

### Common

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALUATION.dataset_stats_path` | auto-search | Optional path to `dataset_stats.json`. Leave unset for the standard `g05-robotwin20` layout; pass it only when the stats file lives elsewhere. |
| `EVALUATION.task_name` | `null` | Run a single task by name, for example `click_alarmclock`. If unset, all tasks are evaluated. |
| `EVALUATION.eval_num_episodes` | `50` | Number of episodes per phase. |
| `MULTIRUN.num_gpus` | `8` | Number of GPUs to use, with GPU IDs from `0` to `num_gpus - 1`. |
| `MULTIRUN.max_tasks_per_gpu` | `2` | Maximum concurrent workers per GPU. Start with `1` and increase only if VRAM allows. |

### Inference

These can be overridden via `EVALUATION.<key>=<value>`. Defaults are in `galaxeafm_policy/deploy_policy.yml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replan_steps` | `8` | Number of actions executed before re-inferring. |
| `num_inference_steps` | `10` | Flow-matching diffusion steps. |
| `action_horizon` | model config | Length of the predicted action chunk. |
| `mixed_precision` | `bf16` | Inference precision: `no`, `fp16`, or `bf16`. |
| `device` | `cuda` | Inference device. |
| `seed` | `0` | Random seed. |

## Output Layout

Aggregated GalaxeaVLA results are written to:

```text
evaluate_results/robotwin/<ckpt_tag>/<timestamp>/
├── summary.csv                         # Per-task success rates
├── summary.json                        # Per-task and overall success rates
├── manager.log                         # Manager scheduling log
├── failed_tasks.txt                    # Failed or unstarted tasks
├── eval_config_<task_name>.yaml        # Resolved single-task config snapshot
├── <task_name>/
│   ├── _result_clean.txt               # Clean phase success rate
│   └── _result_random.txt              # Random phase success rate
└── eval_<task_name>_<timestamp>.log    # Per-task evaluation log
```

RoboTwin-rendered videos remain under:

```text
third_party/RoboTwin/eval_result/<task_name>/galaxeafm_policy/<phase>/
```

## Architecture

```text
run_robotwin_manager.py          (multi-task scheduler)
  └── eval_robotwin_single.py    (single-task Hydra entry)
        └── third_party/RoboTwin/script/eval_policy.py
              └── galaxeafm_policy/deploy_policy.py
                    ├── get_model()     -> loads checkpoint + processor, returns policy
                    ├── eval()          -> executes one action step
                    └── reset_model()   -> clears action queue between episodes
```

- **Manager** reads the task list, assigns tasks to GPU workers, and terminates all workers on any failure. Results are aggregated into `summary.csv` and `summary.json`.
- **Single evaluator** resolves Hydra config, creates the policy symlink at `third_party/RoboTwin/policy/galaxeafm_policy -> experiments/robotwin/galaxeafm_policy`, patches RoboTwin's local episode count for the requested run, and launches `script/eval_policy.py`.
- **Policy adapter** converts RoboTwin observations into GalaxeaVLA input format, loads model and tokenizer config from the checkpoint `.hydra/config.yaml`, and executes predicted action chunks with `replan_steps` chunking.

## Training

The RoboTwin fine-tuning config is `configs/task/robotwin.yaml`, with data defined in `configs/data/robotwin.yaml`:

```bash
bash scripts/run/finetune.sh 8 robotwin
```
