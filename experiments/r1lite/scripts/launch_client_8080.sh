#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SCRIPT_DIR}/../:${PYTHONPATH:-}"

source "${SCRIPT_DIR}/../start.sh"

python3 "${SCRIPT_DIR}/../run.py"   --binarize-gripper --gripper-threshold 70 --visualize
# python3 "${SCRIPT_DIR}/../run.py" --visualize

