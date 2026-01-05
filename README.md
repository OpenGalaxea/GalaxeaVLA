# Galaxea Open-World Dataset & G0 Dual-System VLA Model

[![Project Page](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=github)](https://opengalaxea.github.io/GalaxeaVLA/)
[![Paper](https://img.shields.io/badge/Paper-8A2BE2?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2509.00576v1)
[![Videos](https://img.shields.io/badge/Videos-FF0000?style=for-the-badge&logo=youtube)](https://opengalaxea.github.io/GalaxeaVLA/)
[![Visualizer](https://img.shields.io/badge/Visualizer-FF8C00?style=for-the-badge&logo=airplayvideo)](https://opengalaxea.github.io/GalaxeaVLA/visualizer/index.html)
[![Huggingface](https://img.shields.io/badge/Huggingface-FF6B35?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
[![Modelscope](https://img.shields.io/badge/Modelscope-1890FF?style=for-the-badge&logo=alibabacloud)](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)
[![Twitter](https://img.shields.io/badge/Twitter-FF6B35?style=for-the-badge&logo=x)](https://x.com/Galaxea_x)
[![Linkedin](https://img.shields.io/badge/Linkedin-5865F2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/galaxeadynamics/posts/?feedView=all&viewAsMember=true)
[![Discord](https://img.shields.io/badge/Discord-1890FF?style=for-the-badge&logo=discord)](https://discord.gg/hB6BuUWZZA)

<div align="left">
  <img src="assets/r1_mascot.jpeg" alt="mascot" style="height: 160px; margin-right: 0px;">
  <img src="assets/g0plus_logo.png" alt="logo" style="height: 110px;">
</div>


## 📢 News

[Jan 4, 2026] We are releasing **G0Plus**, our latest pre-trained VLA model for multi-task robot manipulation.

[Oct 7, 2025] Now Lerobot Format Galaxea Open-World Dataset is available at [Huggingface](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)!

[Sep 17, 2025] Release G0-VLA fine-tuning and real-robot inference code.

[Sep 9, 2025] Release G0-VLA pretrained model weights. [Huggingface](https://huggingface.co/OpenGalaxea/G0-VLA) and [Modelscope](https://www.modelscope.cn/models/Galaxea/G0-VLA)!

[Sep 9, 2025] Release Galaxea Open-World Dataset. [Huggingface](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) and [Modelscope](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)!


## 📌 Overview

**GalaxeaVLA** is an open-source project dedicated to advancing real-world, long-horizon, and few-shot robot manipulation.

1. **Galaxea Open-World Dataset**
   - **500+ hours** of real-world mobile manipulation data.
   - All data collected using **one uniform robotic embodiment** for consistency.
   - Fine-grained **subtask language annotations**.
   - Covers **residential**, **kitchen**, **retail**, and **office** settings.
   - Dataset in **RLDS/LeRobot** format.

2. **Easy-to-Use Fine-Tuning Framework**
   - Fully compatible with the [LeRobot](https://github.com/huggingface/lerobot) dataset format and scalable to large, real-world datasets.
   - Modular design enables easy extension and adaptation for new tasks and environments.

3. **Model Checkpoints & An Out-of-the-Box Demo!**  
   - **G0Plus_3B_base**: A powerful pre-trained model with **2k hours+** real-world robot data for fine-tuning on custom tasks.
   - **G0Plus_3B_base-pick_and_place**: A deployment-ready checkpoint, post-trained for robust pick-and-place performance in the wild.
   - **Out-of-the-Box Pick Up Anything Demo**: a Dockerfile and step-by-step guides for quick setup and reproducible experiments.


<p align="center">
  <img src="assets/Galaxea_G0_Plus.png" alt="G0Plus Overview" width="700"/>
</p>


## 🚀 Galaxea Open-World Dataset

### **Key features**

- **500+ hours** of real-world mobile manipulation data.
- All data collected using **one uniform robotic embodiment** for consistency.
- Fine-grained **subtask language annotations**.
- Covers **residential**, **kitchen**, **retail**, and **office** settings.
- Dataset in **RLDS** and **LeRobot** format.

See more dataset (formats and examples) details [here](docs/dataset.md).

## ⚙️ GalaxeaVLA Getting Started

### GPU Requirements

To run our pretrained models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `--nnodes` and`--nproc-per-node` in the fine-tune start shell script. 

| Mode               | Memory Required | Example GPU              |
| ------------------ | --------------- | ------------------------ |
| Inference          | > 8 GB          | RTX 3090 / **RTX 4090 (Recommended)**      |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H20 (96GB) |

### Installation

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA
cd GalaxeaVLA
uv sync --index-strategy unsafe-best-match
source .venv/bin/activate

uv pip install -e .
uv pip install -e .[dev]
```
Note that before you run the installation:
1. recommend to [install uv](https://docs.astral.sh/uv/getting-started/installation/) without using a conda environment.
2. recommend to add env variables at the beginning of your terminal, if you are in the country:
   ```bash
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
   export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
   ```


### Model Checkpoints

| Model                  | Use Case    | Description                       | Checkpoint Path                                              |
| ---------------------- | ----------- | --------------------------------- | ------------------------------------------------------------ |
| G0_3B_base              | Fine-Tuning | Base G0-VLA Model for fine-tuning | https://huggingface.co/OpenGalaxea/G0-VLA/blob/main/G0_3B_base.pt |
| G0Plus_3B_base              | Fine-Tuning | Base G0Plus-VLA Model for fine-tuning | https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_3B_base |
| G0Plus_3B_base-pick_and_place | Deployment | Pick-and-Place Demo in the Wild | https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_PP_CKPT |

### 🔥 Fine-Tuning Base Models on Galaxea R1Lite Robot

To fine-tune our models with your own data, you should follow three steps:

1. Create your own task configs in `configs/tasks/real/`. You can adapt it from our [configs demo](configs/task/real/r1lite_g0plus_finetune_demo.yaml).

2. Install the required packages

   ```bash
   sudo apt install ffmpeg
   ```

3. Set your environment variables
    - `HF_DATASETS_CACHE`: An empty directory for HF-related caches.
    - `GALAXEA_FM_OUTPUT_DIR`: An empty directory for checkpoints and logs output.
    - `SWANLAB_API_KEY`: Your SwanLab API key.
    
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_DATASETS_CACHE=<YOUR_HF_CACHE_PATH>
    export GALAXEA_FM_OUTPUT_DIR=<YOUR_OUTPUT_DIR>
    export SWANLAB_API_KEY=<YOUR_SWANLAB_API_KEY>
    ```

4. Running fine-tuning

   ```bash
   bash scripts/run/finetune.sh <num_of_gpu> <task_path>
 
   # example:
   bash scripts/run/finetune.sh 8 real/r1lite_g0plus_finetune_demo
   ```

#### FAQs of Fine-tuning

1. Q: How to convert my data to a [LeRobot](https://github.com/huggingface/lerobot) dataset? 

   A: The [demo datasets](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_Finetune_LeRobot_Datasets_Demo) are provided on HuggingFace for easy trying.

2. Q: Cannot view the training logs in the SwanLab? 
   
   A: Make sure you set your own swanlab `workspace` in [train.yaml](configs/train.yaml).

3. Q: Cannot find the pre-trained model? 

   A: We use `google/paligemma-3b-pt-224` as the pre-trained model, you should modify it twice in [g0plus.yaml](configs/model/vla/g0plus.yaml) the same as your actual path (default: `/data/google/paligemma-3b-pt-224`).

4. Q: Out of Memory (OOM) error? 

   A: Make sure you have enough GPU memory as mentioned above. Or, reduce the `batch_size` in [g0plus.yaml](configs/model/vla/g0plus.yaml) (default: `4`).

### 🔥🔥 Out-of-the-Box Pick Up Anything Demo

We provide you with a [detailed user guide](docs/pick_up_anything_user_guideline.md) for setting up and running the Pick Up Anything Demo from scratch. 

Feel free to raise an issue if you have any questions.


## Acknowledgement

This project builds upon prior work from the open-source community. The implementation was inspired by [open-pi-zero](https://github.com/allenzren/open-pi-zero), [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), and [Openpi](https://github.com/Physical-Intelligence/openpi), and the experiments make use of datasets including [OXE](https://github.com/google-deepmind/open_x_embodiment), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [BridgeV2](https://github.com/rail-berkeley/bridge_data_v2), and [DROID](https://github.com/droid-dataset/droid). We sincerely thank the authors of these projects for making their code and data publicly available.


## 📜 Citation

If you use our dataset or models, please cite:

```bibtex
@article{galaxea2025,
  title={Galaxea G0: Open-World Dataset and Dual-System VLA Model},
  author={Galaxea Team},
  journal={arXiv preprint arXiv:2509.00576v1},
  year={2025}
}
```

## License

This repository contains materials released under different licenses depending on the commit date:
- **Apache-2.0 (Legacy)**: All content committed **before 2026-01-04** is licensed under the Apache License 2.0.
- **G0 PLUS Community License Agreement (Current)**: All content **committed on or after 2026-01-04** is licensed under the **G0 PLUS Community License (Non-Commercial + Limited Patent License)**. See [G0 Plus Community License Agreement](./LICENSE-G0-PLUS).

For avoidance of doubt, the licensing boundary is the first commit that introduced the G0 PLUS license switch:
- Boundary commit (first under G0 PLUS license): `318207fe6d994d0ecaf8f7d7ebb9b96fec5ebf56`.


### What you can do under the G0 Plus Community License

You may use, reproduce, modify, and distribute the G0 Plus materials **only for non-commercial purposes**, such as academic research, personal use, education, and evaluation. Commercial use (including production deployment, providing services to third parties, or productization) requires a separate commercial license from us.

### Notices and attribution

If you redistribute any part of the G0 Plus materials, you must include:
- a copy/link of G0 Plus Community License Agreement, and
- the NOTICE file in this repository, and
- prominent notices on modified files indicating changes.