# RoboTwin 评测

本目录包含在 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 仿真基准上评测 GalaxeaVLA 的脚本，支持单任务评测和多任务多 GPU 调度。

## 目录结构

| 文件 | 说明 |
|------|------|
| `run_robotwin_manager.py` | 多任务评测调度器：读取任务列表、分配 GPU、汇总结果。 |
| `eval_robotwin_single.py` | 单任务 Hydra 入口，由 manager 以子进程方式启动。 |
| `setup_robotwin_venv.sh` | 创建 `.venv-robotwin`，同步项目依赖，安装 RoboTwin 仿真依赖，并验证 import。 |
| `galaxeafm_policy/deploy_policy.py` | 策略适配层：将 GalaxeaVLA 推理封装为 RoboTwin 策略接口（`get_model` / `eval` / `reset_model`）。 |
| `galaxeafm_policy/deploy_policy.yml` | 默认策略参数，例如 action horizon、replan steps、推理精度和 dataset statistics 路径。 |

## 环境配置

> 请先按照项目根目录 [README_zh.md](../../README_zh.md) 安装 `uv`。除非命令显式切换目录，下面所有命令均在项目根目录执行。

> RoboTwin 使用独立的 `.venv-robotwin`，与主 `.venv` 分开。不要用 `uv run python` 跑 RoboTwin 评测，因为它会遵循 `pyproject.toml` / `uv.lock`，可能把 RoboTwin 所需的 `sapien` 3.x 替换为主环境的 `sapien` 2.x。

### 1. 克隆 RoboTwin

将 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 仓库和子模块克隆到 `third_party/`：

```bash
git clone --recursive https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
```

如果克隆时没有带子模块，补充初始化：

```bash
cd third_party/RoboTwin
git submodule update --init --recursive
cd -
```

期望的 RoboTwin root 是直接包含 `script/eval_policy.py` 和 `task_config/_eval_step_limit.yml` 的目录：

```bash
test -f third_party/RoboTwin/script/eval_policy.py
test -f third_party/RoboTwin/task_config/_eval_step_limit.yml
```

如果你的 RoboTwin checkout 使用了其他目录结构，请在评测命令中通过 `EVALUATION.robotwin_root=...` 指向实际 root。

### 2. 下载 Assets

在 RoboTwin root 下载 assets：

```bash
cd third_party/RoboTwin
bash script/_download_assets.sh
cd -
```

如果共享存储中已经有完整 assets，可以直接链接整个 assets 目录：

```bash
ln -sfn /absolute/path/to/RoboTwin/assets third_party/RoboTwin/assets
```

如果无法直接访问 Hugging Face，可以使用镜像手动下载：

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

### 3. 创建 RoboTwin 虚拟环境

创建专用 `.venv-robotwin` 环境：

```bash
bash experiments/robotwin/setup_robotwin_venv.sh
```

脚本会先把项目依赖同步到 `.venv-robotwin`，再安装 RoboTwin 专用依赖：

```text
sapien==3.0.0b1
warp-lang==0.11.0
nvidia-curobo from third_party/RoboTwin/envs/curobo
```

如果 `third_party/RoboTwin/envs/curobo` 不存在，脚本会先把已验证的 `v0.7.5` 源码克隆到该目录再安装。PyPI 上名为 `curobo` 的包不是 NVIDIA cuRobo，且 cuRobo v0.8.x 与 RoboTwin 2.0 不兼容。

只有需要使用非默认路径或版本时，才在运行脚本前设置 `ROBOTWIN_VENV`、`ROBOTWIN_ROOT`、`PYTHON_VERSION` 或 `CUROBO_VERSION`。

### 4. 生成 cuRobo 配置文件

RoboTwin embodiment assets 中包含带 `${ASSETS_PATH}` 占位符的模板文件。assets 放好后，运行一次脚本生成实际的 `curobo_left.yml` 和 `curobo_right.yml`：

```bash
cd third_party/RoboTwin
../../.venv-robotwin/bin/python script/update_embodiment_config_path.py
cd -
```

这一步会在 `third_party/RoboTwin/assets/embodiments/` 的模板旁边写入生成后的配置文件。如果 `assets` 是只读共享存储软链，请确认其中已经存在生成后的 `curobo*.yml`；如果缺失，需要换成本地可写 assets 副本后重新运行该命令。

### 5. 验证渲染

从 RoboTwin root 运行渲染验证：

```bash
cd third_party/RoboTwin
../../.venv-robotwin/bin/python script/test_render.py
cd -
```

该测试应以 `Render Well` 结束。请先修复渲染问题，再运行策略评测。

## Checkpoint 结构

请按根 README 下载共享 processor 和 action tokenizer。如果 Hugging Face 仓库 snapshot 中已经包含 RoboTwin checkpoint，可以只下载 RoboTwin 子集：

```bash
huggingface-cli download OpenGalaxea/G05 \
    --repo-type model \
    --local-dir checkpoints \
    --local-dir-use-symlinks False \
    --include "g05-robotwin20/*" "qwen3_5_2b_base_processor/*" "action_tokenizer.pt"
```

如果 `g05-robotwin20` 与公开 Hugging Face snapshot 分开发放，请按同样的本地目录结构放置文件后再运行评测。

期望本地目录结构：

```text
checkpoints/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
└── g05-robotwin20/
    ├── .hydra/config.yaml
    ├── checkpoints/model_state_dict.pt
    └── dataset_stats.json
```

对于 `checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt`，策略适配层会从 `checkpoints/g05-robotwin20/.hydra/config.yaml` 加载 model 和 tokenizer 配置。该配置必须指向共享 processor `checkpoints/qwen3_5_2b_base_processor` 和共享 action tokenizer `checkpoints/action_tokenizer.pt`。

`dataset_stats.json` 会从 checkpoint 父目录向上搜索。使用上面的标准目录结构时，不需要显式传 `EVALUATION.dataset_stats_path`；只有 stats 文件放在 checkpoint 父目录链之外时才需要指定。

## 使用方法

### 全量评测（所有任务）

运行 RoboTwin `_eval_step_limit.yml` 中的所有任务，并分配到多张 GPU：

```bash
.venv-robotwin/bin/python -u experiments/robotwin/run_robotwin_manager.py \
  task=robotwin \
  ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

每个任务会依次执行两个 phase：`clean` 表示确定性初始摆放，`random` 表示随机初始摆放。默认每个 phase 运行 50 个 episode。

### 单任务评测

按标准 episode 数评测一个指定任务：

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

## 参数说明

### 必选参数

| 参数 | 说明 |
|------|------|
| `task` | Hydra task 配置名称，通常为 `robotwin`。 |
| `ckpt` | 模型 checkpoint 文件路径。 |
| `EVALUATION.robotwin_root` | 包含 `script/eval_policy.py` 的 RoboTwin root。 |

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EVALUATION.dataset_stats_path` | 自动搜索 | 可选的 `dataset_stats.json` 路径。标准 `g05-robotwin20` 目录结构下保持未设置即可；只有 stats 文件放在其他位置时才需要传。 |
| `EVALUATION.task_name` | `null` | 指定单个任务名称，例如 `click_alarmclock`。未指定时评测全部任务。 |
| `EVALUATION.eval_num_episodes` | `50` | 每个 phase 的 episode 数量。 |
| `MULTIRUN.num_gpus` | `8` | 使用的 GPU 数量，GPU ID 为 `0` 到 `num_gpus - 1`。 |
| `MULTIRUN.max_tasks_per_gpu` | `2` | 每张 GPU 上的最大并发 worker 数。建议先设为 `1`，确认显存充足后再提高。 |

### 推理参数

以下参数可通过 `EVALUATION.<key>=<value>` 覆盖，默认值定义在 `galaxeafm_policy/deploy_policy.yml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `replan_steps` | `8` | 每次重新推理前实际执行的 action 步数。 |
| `num_inference_steps` | `10` | Flow-matching 扩散推理步数。 |
| `action_horizon` | 模型配置值 | 预测的 action chunk 长度。 |
| `mixed_precision` | `bf16` | 推理精度：`no`、`fp16` 或 `bf16`。 |
| `device` | `cuda` | 推理设备。 |
| `seed` | `0` | 随机种子。 |

## 输出结构

GalaxeaVLA 汇总结果写入：

```text
evaluate_results/robotwin/<ckpt_tag>/<timestamp>/
├── summary.csv                         # 各任务成功率
├── summary.json                        # 各任务和整体成功率
├── manager.log                         # 调度器日志
├── failed_tasks.txt                    # 失败或未启动的任务
├── eval_config_<task_name>.yaml        # 单任务解析后的配置快照
├── <task_name>/
│   ├── _result_clean.txt               # clean phase 成功率
│   └── _result_random.txt              # random phase 成功率
└── eval_<task_name>_<timestamp>.log    # 单任务评测日志
```

RoboTwin 渲染视频保留在：

```text
third_party/RoboTwin/eval_result/<task_name>/galaxeafm_policy/<phase>/
```

## 架构说明

```text
run_robotwin_manager.py          （多任务调度器）
  └── eval_robotwin_single.py    （单任务 Hydra 入口）
        └── third_party/RoboTwin/script/eval_policy.py
              └── galaxeafm_policy/deploy_policy.py
                    ├── get_model()     -> 加载 checkpoint 和 processor，返回 policy 实例
                    ├── eval()          -> 执行单步 action
                    └── reset_model()   -> episode 结束时清空 action 队列
```

- **Manager** 负责读取任务列表、分配 GPU worker，并在任何 worker 失败时终止所有运行中的任务。结果会汇总到 `summary.csv` 和 `summary.json`。
- **Single evaluator** 负责解析 Hydra 配置、创建 policy 软链接（`third_party/RoboTwin/policy/galaxeafm_policy -> experiments/robotwin/galaxeafm_policy`）、为当前 run patch RoboTwin 的 episode 数量，并启动 `script/eval_policy.py`。
- **策略适配层** 将 RoboTwin 观测转换为 GalaxeaVLA 输入格式，从 checkpoint 的 `.hydra/config.yaml` 加载 model 和 tokenizer 配置，并按 `replan_steps` 执行模型预测的 action chunk。

## 训练

RoboTwin 微调配置为 `configs/task/robotwin.yaml`，数据定义在 `configs/data/robotwin.yaml`：

```bash
bash scripts/run/finetune.sh 8 robotwin
```
