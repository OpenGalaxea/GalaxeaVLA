#!/bin/bash
# Start the G05 DROID policy server (WebSocket :PORT).
#
# Usage:
#   CHECKPOINT_DIR=/abs/path/to/g05_droid POLICY_PORT=8000 \
#     bash experiments/droid/start_server.sh
set -euo pipefail

# Must run from the repository root; relative resources under configs/ and .venv live there.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set to the checkpoint directory}"
PORT="${POLICY_PORT:-8000}"
DEVICE="${POLICY_DEVICE:-cuda}"

# Network/proxy independent; CUDA runtime is provided by the environment.
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# g05 is not pip-installed; this is a src-layout repo (g05 is under src/), so
# PYTHONPATH must be set explicitly.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec .venv/bin/python scripts/serve_policy.py \
  --ckpt_path "${CHECKPOINT_DIR}/checkpoints/model_state_dict.pt" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device "${DEVICE}" \
  eval_embodiment=Droid_Franka \
  "$@"
