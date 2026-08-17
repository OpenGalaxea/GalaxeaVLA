# GalaxeaVLA: G0.5 Vision-Language-Action Model

[![Project Page](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=github)](https://opengalaxea.github.io/G05/)
[![Paper](https://img.shields.io/badge/Paper-8A2BE2?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2608.11739)
[![Videos](https://img.shields.io/badge/Videos-FF0000?style=for-the-badge&logo=youtube)](https://opengalaxea.github.io/G05/videos/introduction_g05.mp4)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF6B35?style=for-the-badge&logo=huggingface)](https://huggingface.co/OpenGalaxea/G05)

[English](README.md) | [Chinese](README_zh.md)

<div align="left">
  <img src="assets/r1_mascot.jpeg" alt="mascot" style="height: 160px; margin-right: 0px;">
  <img src="assets/g05_logo.png" alt="logo" style="height: 160px;">
</div>


## 📢 News

[Aug 12, 2026] Our G0.5 paper, [**G0.5: One Autoregressive Stream for Robot Reasoning and Action**](https://arxiv.org/abs/2608.11739), is now available on arXiv.

[Jul 11, 2026] We add an **G0.5** **R1 Pro** real-robot zero-shot inference deployment entrypoint under [experiments/r1pro](experiments/r1pro), using the same [`g05-base` pretrained checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base) as the R1 Lite zero-shot deployment.

[Jun 29, 2026] We add a **G0.5** **RoboTwin 2.0** evaluation entrypoint under [experiments/robotwin](experiments/robotwin), together with the [`g05-robotwin20` checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20). Evaluation support for more simulation benchmarks will be released soon.

[Jun 17, 2026] We add a **G0.5** **SO-100/101** zero-shot inference deployment entrypoint under [experiments/so100](experiments/so100), together with the [`g05-so101` checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-so101).

[Jun 16, 2026] We release the [`g05-base` pretrained checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base) for **G0.5**. The same checkpoint also supports zero-shot inference deployment on **R1 Lite** via the entrypoint in [experiments/r1lite](experiments/r1lite). We additionally add a **DROID** zero-shot inference deployment entrypoint under [experiments/droid](experiments/droid), using the [`g05-droid` checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-droid), a **LIBERO** simulation evaluation entrypoint under [experiments/libero](experiments/libero), using the [`g05-libero` checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-libero), and R1 Lite/R1 Pro **fine-tuning** through [scripts/run/finetune.sh](scripts/run/finetune.sh), initialized from `g05-base`.

[Jun 1, 2026] We introduce **G0.5**, our latest autoregressive VLA model with state-of-the-art performance. See the [Project Page](https://opengalaxea.github.io/G05/).

[Feb 12, 2026] Updated **G0Plus** pretrained weights trained on larger-scale teleoperation and web data. Released **G0Tiny** (250M, SmolVLM2 backbone) for R1 Pro Orin edge deployment. Added out-of-the-box demos: **Fold Towels** and **Handover Gift** (on-device G0Tiny inference via TensorRT at up to 10 Hz). Added [openpi](https://github.com/Physical-Intelligence/openpi)-based **pi0/pi0fast** fine-tuning support.

[Jan 4, 2026] We released **G0Plus**, our latest pretrained VLA model for multi-task robot manipulation.

[Oct 7, 2025] The Galaxea Open-World Dataset is now available in LeRobot format on [Hugging Face](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)!

[Sep 17, 2025] Released G0-VLA fine-tuning and real-robot inference code.

[Sep 9, 2025] Released G0-VLA pretrained model weights on [Hugging Face](https://huggingface.co/OpenGalaxea/G0-VLA) and [ModelScope](https://www.modelscope.cn/models/Galaxea/G0-VLA)!

[Sep 9, 2025] Released the Galaxea Open-World Dataset on [Hugging Face](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) and [ModelScope](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)!


## 📌 G0.5 Overview

**G0.5** is Galaxea's pretrained autoregressive vision-language-action model for general-purpose robot control. Instead of using the VLM only as a vision-language encoder for a separate action expert, G0.5 keeps the VLM as the actor: a single transformer decoder generates reasoning tokens and action tokens in one unified autoregressive stream under the same next-token prediction objective.

Key ideas from [**G0.5: One Autoregressive Stream for Robot Reasoning and Action**](https://arxiv.org/abs/2608.11739):

1. **Unified autoregressive VLA**
   - G0.5 conditions on multi-view RGB observations, an embodiment identifier, natural-language task instruction, and robot proprioceptive state.
   - The model can first emit embodied reasoning, then continue with structured action tokens in the same generation stream.
   - Action tokens are decoded into continuous robot controls, executed in chunks, and replanned closed-loop from new observations.

2. **Cross-embodiment ActionCodec**
   - Heterogeneous robot actions are mapped into a shared 27-dimensional action space:
     `left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7)`.
   - A learned residual vector-quantized action tokenizer represents semantically aligned motion groups such as left arm, right arm, and lower body.
   - Only active motion groups are emitted during action generation, avoiding unnecessary padding for idle degrees of freedom.

3. **Native chain-of-thought for control**
   - G0.5 trains reasoning and action as one token sequence, rather than treating reasoning as an external module or training-only auxiliary target.
   - The chain-of-thought span can include `Subtask`, `BBox`, `Trace`, and `ActionHint` fields for task decomposition, object grounding, 2D gripper traces, and frame-level motion hints.
   - This design improves grounding and long-horizon execution because generated reasoning is directly visible to the following action tokens.

4. **Visual memory**
   - G0.5 injects multi-second visual history through the vision encoder with factorized spatial-temporal attention.
   - Pre-training uses 6 frames sampled over a 5-second window, with stochastic history-frame dropout to reduce overfitting.
   - Historical tokens are discarded at the final layer to keep inference latency bounded.

5. **Pre-training and evaluation**
   - G0.5 is initialized from a Qwen3.5 2B VLM and pretrained on robot demonstrations from 14 embodiments together with large-scale web and embodied VQA data.
   - Robot action samples and VQA samples are optimized with the same cross-entropy objective, mixed at a VQA-to-action ratio of 1:4.
   - The model reports strong performance across diverse settings, including 82.5% zero-shot success on DROID, 87.3% on Bridge-SimplerEnv, 93.3% on RoboTwin 2.0, 98.9% on LIBERO, a 0.3136 task success score on BEHAVIOR-1K, and 76.7% average success after real-world fine-tuning on R1-Lite/R1-Pro.

<p align="center">
  <img src="assets/tokenizer.png" alt="G0.5 Tokenizer" width="700"/>
  <br>
  <em>G0.5 Tokenizer</em>
</p>

<p align="center">
  <img src="assets/token_template.png" alt="G0.5 Token Sequence Template" width="700"/>
  <br>
  <em>G0.5 Token Sequence Template</em>
</p>

## 📌 Legacy G0 / G0Plus Overview

This section preserves the G0Plus overview from earlier GalaxeaVLA releases for continuity. G0Plus and G0Tiny belong to the legacy G0/G0Plus release line. The current branch focuses on G0.5, and no longer includes the legacy G0Plus Dockerfiles, Fold Towels/Handover Gift demo guides, or pi0/pi0fast training configs. For the exact legacy README and file layout, use commit [13a16a9](https://github.com/OpenGalaxea/GalaxeaVLA/tree/13a16a9049aee8f1d799b56fccc0c5832a75fc2f). Historical checkpoints remain available in the [G0-VLA Hugging Face repository](https://huggingface.co/OpenGalaxea/G0-VLA), including: 

   - **G0Plus_3B-base**: pretrained with **2k+ hours** of real-world robot data for fine-tuning on custom tasks.
   - **G0Tiny_250M-base**: lightweight pretrained model with **1k hours** of R1 Pro VR teleoperation data, designed for R1 Pro Orin edge deployment.
   - **G0Plus_3B-pick_and_place**: post-trained checkpoint for pick-and-place deployment.
   - **Demos** included Pick Up Anything, Fold Towels, and Handover Gift.
   - openpi-based **pi0/pi0fast** fine-tuning support.

## 🚀 Galaxea Open-World Dataset

### **Key features**

- **500+ hours** of real-world mobile manipulation data.
- All data collected using **one uniform robotic embodiment** for consistency.
- Fine-grained **subtask language annotations**.
- Covers **residential**, **kitchen**, **retail**, and **office** settings.
- Dataset in **RLDS** and **LeRobot** format.

See more dataset (formats and examples) details [here](docs/data/schema.md).


## ⚙️ G0.5 Getting Started

### GPU Requirements

To run the pretrained G0.5 models in this repository, you need an NVIDIA GPU with at least the following specifications. These estimates assume a single GPU.

Multi-GPU fine-tuning launched through [scripts/run/finetune.sh](scripts/run/finetune.sh) uses `torchrun` distributed data parallelism by default. This improves throughput, but each GPU keeps a full model replica and it should not be described as model parallelism or as a way to reduce per-GPU model memory. Multi-node runs are controlled by environment variables such as `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

| Mode               | Memory Required | Example GPU              |
| ------------------ | --------------- | ------------------------ |
| Inference          | > 8 GB          | RTX 3090 / **RTX 4090 (Recommended)**      |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H20 (96GB) |

### Installation

Tested environment:

- Linux
- Python `>=3.10.16,<3.11`
- CUDA 12.8 with PyTorch 2.7.1 wheels from the `cu128` index
- Native CUDA extensions including `flash-attn-4` and `flash-linear-attention`; make sure your CUDA runtime, compiler toolchain, and PyTorch build match
- `ffmpeg` for video-backed datasets and evaluation

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA
cd GalaxeaVLA

# Runtime environment
uv sync --index-strategy unsafe-best-match

# Or include development/test dependencies instead
uv sync --extra dev --index-strategy unsafe-best-match

source .venv/bin/activate
```
Before installation:
1. We recommend [installing uv](https://docs.astral.sh/uv/getting-started/installation/) without using a conda environment.
2. If you encounter network issues, we recommend trying the following uv environment variables:
   ```bash
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
   export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
   ```


### Model Checkpoints

Download the Hugging Face repository into `checkpoints/` so the local paths match the checked-in configs:

```bash
huggingface-cli download OpenGalaxea/G05 \
    --repo-type model \
    --local-dir checkpoints
```

Expected local layout after downloading available checkpoints and placing any separately distributed RoboTwin checkpoint:

```text
checkpoints/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
├── g05-base/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-so101/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-droid/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-libero/
│   ├── .hydra/config.yaml
│   ├── model.pt
│   └── dataset_stats.json
└── g05-robotwin20/
    ├── .hydra/config.yaml
    ├── checkpoints/model_state_dict.pt
    └── dataset_stats.json
```

With the RoboTwin checkpoint included, the full set is about 55 GB. Each G0.5 model checkpoint is about 11 GB, the shared `action_tokenizer.pt` is about 484 MB, and `qwen3_5_2b_base_processor/` is about 22 MB.

| Model | Use | Local `--ckpt_path` |
| ----- | --- | ------------------- |
| [G05-base](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base) | Fine-tuning and R1 Lite/R1 Pro zero-shot deployment | `checkpoints/g05-base/checkpoints/model_state_dict.pt` |
| [G05-so101](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-so101) | SO-100/101 zero-shot deployment | `checkpoints/g05-so101/checkpoints/model_state_dict.pt` |
| [G05-droid](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-droid) | DROID zero-shot deployment | `checkpoints/g05-droid/checkpoints/model_state_dict.pt` |
| [G05-libero](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-libero) | LIBERO evaluation | `checkpoints/g05-libero/model.pt` |
| [G05-robotwin20](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20) | RoboTwin 2.0 evaluation | `checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt` |

Keep each checkpoint's `.hydra/config.yaml` and `dataset_stats.json` next to the downloaded weights. The inference and evaluation entrypoints resolve `dataset_stats.json` from the checkpoint parent directories, and the configs expect the shared processor at `checkpoints/qwen3_5_2b_base_processor` and the shared tokenizer at `checkpoints/action_tokenizer.pt`.

The RoboTwin checkpoint uses the same sidecar layout. If `g05-robotwin20` is distributed separately from the public Hugging Face snapshot, place it under `checkpoints/g05-robotwin20/` exactly as shown above before running evaluation.

For LIBERO, the exported config expects bundle-local sidecars. The `eval_libero.sh` command below requires these paths. If they are not included by your download tool, create symlinks:

```bash
ln -sf ../action_tokenizer.pt checkpoints/g05-libero/action_tokenizer.pt
ln -sfn ../qwen3_5_2b_base_processor checkpoints/g05-libero/hf_processor
```


### Inference on Real Robots

Real-robot deployment follows a server/client architecture. The GPU machine runs the G0.5 policy server from this repository, while the robot-side client collects raw observations, sends them over WebSocket + msgpack, receives action chunks, and executes them.

#### R1 Lite

Start the policy server on the GPU machine:

```bash
python scripts/serve_policy.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda \
    --action_steps 16 \
    eval_embodiment=galaxea_r1lite
```

Then start the R1 Lite client on the robot machine after setting the server address in `experiments/r1lite/config.toml`:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
cd experiments/r1lite
python run.py --config config.toml
```

See [experiments/r1lite/README.md](experiments/r1lite/README.md) for the ROS2 topics, client/server protocol, instruction file, recording, and testing details.

#### R1 Pro

R1 Pro deployment uses the multi-frame policy server and the ROS2 client under `experiments/r1pro`. Start the policy server on the GPU machine with an R1 Pro checkpoint:

```bash
PYTHONPATH="src:${PYTHONPATH:-}" python scripts/serve_policy_mem.py \
    --ckpt_path /path/to/r1pro/checkpoints/model_state_dict.pt \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda \
    --action_steps 16 \
    eval_embodiment=galaxea_r1pro \
    model.model_weights_to_bf16=true \
    model.use_torch_compile=true \
    model.model_arch.attn_implementation=sdpa \
    model.model_arch.discrete_action=true \
    model.model_arch.continuous_action=false
```

Then set the server address in `experiments/r1pro/config.toml` and start the client on the robot machine:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
cd experiments/r1pro
python run.py --config config.toml
```

See [experiments/r1pro/README.md](experiments/r1pro/README.md) for the ROS2 topics, client/server protocol, instruction management, recording, visualization, and testing details.

#### SO-100 / SO-101

SO-100/101 deployment uses the same G0.5 policy server with a lightweight LeRobot-based client. Start the policy server on the GPU machine:

```bash
bash experiments/so100/start_server.sh checkpoints/g05-so101/checkpoints/model_state_dict.pt
```

Then set up and start the robot-side client in a separate terminal or on the robot machine:

```bash
conda env create -f experiments/so100/environment.yml
conda activate lerobot
bash experiments/so100/start_client.sh
```

Before running the client, update the camera indices and camera-slot mapping in `experiments/so100/start_client.sh`. See [experiments/so100/README.md](experiments/so100/README.md) for details.

#### DROID / Franka

DROID deployment serves the G0.5 policy from this repo and uses the separate
[OpenGalaxea/droid-franka-client](https://github.com/OpenGalaxea/droid-franka-client)
repo for robot control. Start the policy server on the GPU machine:

```bash
CHECKPOINT_DIR=checkpoints/g05-droid \
POLICY_PORT=8000 \
POLICY_DEVICE=cuda:0 \
bash experiments/droid/start_server.sh \
    model.model_arch.discrete_action=true model.model_arch.continuous_action=false
```

Then clone and configure the robot-side client:

```bash
git clone git@github.com:OpenGalaxea/droid-franka-client.git
```

See [experiments/droid/README.md](experiments/droid/README.md) and [experiments/droid/PROTOCOL.md](experiments/droid/PROTOCOL.md) for the full setup and protocol contract.

### Training on LIBERO

For a minimal end-to-end LIBERO fine-tuning walkthrough — wiring the `g05-base` checkpoint,
the LIBERO LeRobot datasets, and `libero_stats.json` into the layout the configs expect, then
running a single-GPU smoke test — see [skills/setup-libero-training](skills/setup-libero-training/SKILL.md).
Training does **not** need the LIBERO simulator (that is only for evaluation rollouts).

### Evaluation on LIBERO

LIBERO evaluation also uses a server/client setup. Install the LIBERO simulator first — see
[experiments/libero/README.md](experiments/libero/README.md#prerequisites). This command starts a batched policy server in the background, launches one parallel client for each LIBERO suite, then stops the server after evaluation.

```bash
bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --num_trials 50 \
    --num_parallel 10 \
    --save_videos
```

By default, the script evaluates `libero_goal`, `libero_spatial`, `libero_object`, and `libero_10`, then writes per-suite logs and `summary.json` under `outputs/libero_eval_<checkpoint_name>/`. Use `--suites "libero_goal libero_10"` to run a subset, and append Hydra-style overrides such as `model.model_arch.discrete_action=false` when needed.

See [experiments/libero/README.md](experiments/libero/README.md) for LIBERO installation notes, generated path config, output layout, and single-suite debugging commands.

### Evaluation on RoboTwin

RoboTwin evaluation runs inside the RoboTwin simulator and requires extra simulation dependencies and assets. It uses a separate `.venv-robotwin` environment so the simulator's `sapien` 3.x dependency does not overwrite the main repository environment. Follow [experiments/robotwin/README.md](experiments/robotwin/README.md) to clone RoboTwin, download assets, create `.venv-robotwin`, and verify rendering.

Run the full RoboTwin task set:

```bash
.venv-robotwin/bin/python -u experiments/robotwin/run_robotwin_manager.py \
    task=robotwin \
    ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
    EVALUATION.robotwin_root=third_party/RoboTwin \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=1
```

This evaluates every task listed in RoboTwin's `_eval_step_limit.yml` with the configured episode count. `dataset_stats.json` is resolved automatically from the checkpoint parent directories for the standard `g05-robotwin20` layout. Pass `EVALUATION.dataset_stats_path=/path/to/dataset_stats.json` only when the stats file lives elsewhere. Results are written under `evaluate_results/robotwin/<ckpt_tag>/<timestamp>/`, while RoboTwin-rendered videos remain under `third_party/RoboTwin/eval_result/`.

### 🔥 Fine-Tuning Base Models on Galaxea Robots

To fine-tune our models with your own data, follow four steps. See [configs/QUICK_START.md](configs/QUICK_START.md) for the config hierarchy and common edit locations.

1. Create or adapt a task config under `configs/task/`, and update the matching dataset paths under `configs/data/`. For Galaxea robot fine-tuning, start from one of the provided configs:
   - R1 Lite: [configs/task/r1lite.yaml](configs/task/r1lite.yaml) with dataset paths in [configs/data/r1lite.yaml](configs/data/r1lite.yaml).
   - R1 Pro joint-space training: [configs/task/r1pro.yaml](configs/task/r1pro.yaml) with dataset paths in [configs/data/r1pro.yaml](configs/data/r1pro.yaml).
   - R1 Pro whole-body-control training with torso state/action: [configs/task/r1pro_wbc.yaml](configs/task/r1pro_wbc.yaml) with dataset paths in [configs/data/r1pro_wbc.yaml](configs/data/r1pro_wbc.yaml).

   All three Galaxea robot data configs contain example local dataset paths and
   must be edited before training:
   - R1 Lite: replace `data/r1lite/task*_lerobot` in `configs/data/r1lite.yaml`.
   - R1 Pro: replace `data/r1pro/fold_carton_lerobot` in `configs/data/r1pro.yaml`.
   - R1 Pro WBC: replace `data/r1pro_wbc/stack_box_lerobot` in `configs/data/r1pro_wbc.yaml`.

   The paths should point to your local LeRobot dataset directories.

2. Install the required packages.

   ```bash
   sudo apt install ffmpeg
   ```

3. Set your environment variables.
    - `G05_OUTPUT_DIR`: Required by `configs/train.yaml`; checkpoints and Hydra logs are written under this directory.
    - `HF_HOME` / `HF_HUB_CACHE`: Recommended cache locations for Hugging Face model and tokenizer snapshots.
    - `HF_DATASETS_CACHE`: Recommended cache location for Hugging Face datasets and LeRobot metadata.
    - `LIBERO_CONFIG_PATH`: Required only for LIBERO simulation evaluation. It points to the directory where LIBERO reads or writes `config.yaml` with `bddl_files`, `init_states`, `assets`, and dataset paths.
    - `SWANLAB_API_KEY`: Required only when `logger.type=swanlab`.
    - `WANDB_API_KEY`: Required only when `logger.type=wandb` and `logger.mode=online`.

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=<YOUR_HF_CACHE_ROOT>
    export HF_HUB_CACHE=$HF_HOME/hub
    export HF_DATASETS_CACHE=$HF_HOME/datasets
    export G05_OUTPUT_DIR=<YOUR_OUTPUT_DIR>
    export LIBERO_CONFIG_PATH=$(pwd)/experiments/libero
    export SWANLAB_API_KEY=<YOUR_SWANLAB_API_KEY>
    ```

    Copy the repository-local environment template and fill in machine-specific paths and optional API keys:

    ```bash
    cp .env.example .env
    source .env
    ```

4. Run fine-tuning.

   ```bash
   bash scripts/run/finetune.sh <num_of_gpu> <task_path>

   # dry-run: resolve config and exit before training
   bash scripts/run/finetune.sh 1 r1pro --dry-run --max_datasets 1

   # single-GPU smoke test
   bash scripts/run/finetune.sh 1 r1pro --test --max_datasets 1 model.max_steps=10

   # multi-GPU examples:
   bash scripts/run/finetune.sh 8 r1lite
   bash scripts/run/finetune.sh 8 r1pro
   bash scripts/run/finetune.sh 8 r1pro_wbc
   ```

   R1 Pro configs use the grouped 27D ActionCodec layout:
   `left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7)`.
   The WBC variant maps `torso` into the `lower_body` group through [configs/data/parts_meta/r1pro.yaml](configs/data/parts_meta/r1pro.yaml).

#### Fine-Tuning FAQ

1. Q: How do I convert my data to a [LeRobot](https://github.com/huggingface/lerobot) dataset?

   A: The [demo datasets](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_Finetune_LeRobot_Datasets_Demo) are provided on Hugging Face for quick testing.

2. Q: Why can't I view the training logs in SwanLab?

   A: Make sure you set your own SwanLab `workspace` in [train.yaml](configs/train.yaml).

3. Q: Why can't I find the pretrained model?

   A: The G0.5 model config is [g05.yaml](configs/model/g05.yaml). Update `model.model_arch.pretrained_model_path` or the task-level `model.pretrained_ckpt` if your checkpoints live in custom paths.

4. Q: Why am I getting an out-of-memory (OOM) error?

   A: Make sure you have enough GPU memory as mentioned above. Alternatively, reduce `model.batch_size` in the corresponding task config under [configs/task](configs/task).

### Minimal Validation

After installation and checkpoint download, run lightweight checks before launching expensive jobs:

```bash
python tools/resolve_config.py r1pro --key model.model_arch
python tools/resolve_config.py r1pro --key data.embodiment_datasets.galaxea_r1pro.dataset_groups
python -c "import torch, g05; print(torch.__version__); print(g05.__file__)"
```

Tests are split by their runtime requirements. Synthetic/unit tests can run without real robot data, while serving, pretrained-policy, and dataset tests need CUDA, downloaded checkpoints, or local LeRobot datasets. LIBERO batch evaluation additionally assumes Linux `ss` is available because [scripts/run/eval_libero.sh](scripts/run/eval_libero.sh) uses it to wait for the policy-server port.

## Acknowledgements

This project builds upon prior work from the open source community. The implementation was inspired by [open-pi-zero](https://github.com/allenzren/open-pi-zero), [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), [OpenPI](https://github.com/Physical-Intelligence/openpi), and [LeRobot](https://github.com/huggingface/lerobot), and the experiments make use of datasets including [OXE](https://github.com/google-deepmind/open_x_embodiment), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [BridgeV2](https://github.com/rail-berkeley/bridge_data_v2), and [DROID](https://github.com/droid-dataset/droid). We sincerely thank the authors of these projects for making their code and data publicly available.


## 📜 Citation

If you use our dataset or models, please cite:

```bibtex
@misc{liu2026g05autoregressivestreamrobot,
  title={G0.5: One Autoregressive Stream for Robot Reasoning and Action},
  author={Yicheng Liu and Zibin Dong and Baijun Ye and Tianyuan Yuan and Tao Jiang and Anqi Yang and Shicheng Cao and Haonan Liu and Yue Sun and Zihan Guo and Xiao Liu and Ke Dong and Changxun Pan and Chenru Wu and Tailai Cheng and Xiaoshu Ren and Xinlei Zhang and Jianning Cui and Zijie Zhao and Haoyu Zhang and Kaiming Xu and Haodong Yang and Bowen Zhang and Jiahui Niu and Shaoting Zhu and Shiduo Zhang and Hang Zhao},
  year={2026},
  eprint={2608.11739},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2608.11739}
}
```

## License

This repository contains materials released under different licenses depending on the commit date:
- Apache-2.0 (Legacy): All content committed before 2026-01-04 is licensed under the Apache License 2.0.
- G0 PLUS Community License Agreement: All content committed on or after 2026-01-04 and before 2026-06-16 is licensed under the G0 PLUS Community License (Non-Commercial + Limited Patent License). See [G0 Plus Community License Agreement](licenses/LICENSE-G0Plus).
- G0.5 Community License Agreement: All content committed on or after 2026-06-16 is licensed under the G0.5 Community License (Non-Commercial + Limited Patent License). See [G0.5 Community License Agreement](licenses/LICENSE-G0.5).
For avoidance of doubt, there are two licensing boundaries, each determined by the first commit that introduces the corresponding license switch:
- Boundary 1 (Apache-2.0 / G0 PLUS): First commit under the G0 PLUS license (introducing the G0 PLUS license switch): [38b31e4](https://github.com/OpenGalaxea/GalaxeaVLA/tree/38b31e4f732ef28719a5458a18e2836dd52f9d12)
- Boundary 2 (G0 PLUS / G0.5): First commit under the G0.5 license (introducing the G0.5 license switch): [dc0a1ef](https://github.com/OpenGalaxea/GalaxeaVLA/tree/dc0a1ef4531256adea4ee9f3d7d2fa44613cb866)
In the event of any inconsistency between the date descriptions and the commit hashes, the commit hashes shall prevail.

### What you can do under the G0.5 Community License

You may use, reproduce, modify, and distribute the G0.5 materials only for non-commercial purposes, such as academic research, personal use, education, and evaluation. Commercial use (including production deployment, providing services to third parties, or productization) requires a separate commercial license from us.

The G0.5 materials include model code, weights, configurations, training/inference scripts, documentation, and accompanying materials. See [LICENSE-G0.5](./LICENSE-G0.5) and [licenses/LICENSE-G0.5](licenses/LICENSE-G0.5).

### Notices and attribution

If you redistribute any part of the G0.5 materials, you must include:
- a copy/link of the G0.5 Community License Agreement,
- the NOTICE file in this repository, and
- prominent notices on modified files indicating changes.

### Third-party licenses

G0.5 uses Qwen3.5 as its pretrained VLM backbone and includes Qwen3.5-derived implementation components. Qwen3.5 model weights, configuration files, and related upstream materials are licensed by Qwen under Apache License 2.0. See [LICENSE_QWEN3_5.txt](./LICENSE_QWEN3_5.txt).

### Legal and safety compliance

This repository release does not itself provide a public-facing generative AI service. If you deploy, fine-tune, redistribute, or expose any model, service, output, or robot-control workflow, you are responsible for complying with applicable laws and regulations, including requirements for data rights, personal information protection, safety assessment or filing, content safety, generated-content labeling, and robotics safety in your jurisdiction.

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=OpenGalaxea/GalaxeaVLA&type=date&legend=top-left)](https://star-history.dera.page/#OpenGalaxea/GalaxeaVLA&type=date&legend=top-left)
