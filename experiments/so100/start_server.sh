#!/usr/bin/env bash
# Start the SO100 inference server on a GPU training machine using the g05 training environment.
#
# Usage:
#   bash experiments/so100/start_server.sh /path/to/checkpoint.pt
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT="$HERE/../.."                       # repository root

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/so100/start_server.sh /path/to/checkpoint.pt" >&2
  exit 2
fi

CKPT="$1"

export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"

echo "Starting SO100 policy server on $(hostname) ..."
echo "  ckpt = $CKPT"
cd "$PROJECT"
python "$PROJECT/scripts/serve_policy.py" \
  --ckpt_path "$CKPT" \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda \
  --action_steps 32 \
  eval_embodiment=so100 \
  model.model_weights_to_bf16=true \
  model.use_torch_compile=true \
  model.model_arch.attn_implementation=sdpa
