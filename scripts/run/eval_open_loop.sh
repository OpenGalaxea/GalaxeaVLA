#!/bin/bash
# Open-loop evaluation — auto-loads config from the ckpt's run directory
#
# Usage:
#   bash scripts/run/eval_open_loop.sh --ckpt_path <ckpt> [--task_config <yaml>] [key=value ...] [--max_datasets N]
#
# Examples:
#   # Basic eval (config auto-detected from run_dir)
#   bash scripts/run/eval_open_loop.sh --ckpt_path runs/.../last.pt
#
#   # Eval with a custom task config (overrides config from run_dir)
#   bash scripts/run/eval_open_loop.sh \
#       --ckpt_path runs/.../checkpoints/step_24000.pt \
#       --task_config configs/task/libero.yaml
#
#   # Eval only r1lite
#   bash scripts/run/eval_open_loop.sh --ckpt_path runs/.../last.pt eval_embodiment=galaxea_r1lite
#
#   # Debug mode: 1 dataset per group, 1 episode
#   bash scripts/run/eval_open_loop.sh --ckpt_path runs/.../last.pt --max_datasets 1 eval_episodes_num=1
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

python scripts/eval_open_loop.py "$@"
