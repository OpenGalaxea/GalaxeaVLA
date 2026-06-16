#!/bin/bash
# R1_LITE client environment configuration


export TZ=Asia/Shanghai
echo $TZ
date

# Mirror sources
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
export HF_ENDPOINT=https://hf-mirror.com

# Paths, inferred automatically from this script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TORCH_HOME=$HF_HOME/torch

# Display
export LC_CTYPE=en_US.UTF-8
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false

# Network
unset http_proxy https_proxy
export MUJOCO_GL=egl

# CPU thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HYDRA_FULL_ERROR=1

# ROS2
set +u
_oldpwd=$PWD
cd /opt/ros/humble
source setup.bash
cd "$_oldpwd"
unset _oldpwd
set -u

if [[ ! -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    echo "Missing repository virtual environment: $REPO_ROOT/.venv" >&2
    echo "Run 'uv sync' from the repository root first." >&2
    return 1 2>/dev/null || exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
cd "$SCRIPT_DIR"

echo "R1_LITE client environment activated"
