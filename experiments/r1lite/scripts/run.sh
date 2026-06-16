#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SCRIPT_DIR}/../:${PYTHONPATH:-}"

python3 "${SCRIPT_DIR}/../run.py" "$@"
