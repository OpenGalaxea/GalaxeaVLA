#!/usr/bin/env bash
# Local SO100 server startup helper.
#
# Usage: bash experiments/so100/start_server_local.sh /path/to/checkpoint.pt
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT="$HERE/../.."                       # repository root

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/so100/start_server_local.sh /path/to/checkpoint.pt" >&2
  exit 2
fi

CKPT="$1"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$PROJECT/src:$PROJECT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[server] ckpt = $CKPT"
echo "[server] venv = $PROJECT/.venv"
cd "$PROJECT"
exec ./.venv/bin/python scripts/serve_policy.py \
  --ckpt_path "$CKPT" \
  --host 0.0.0.0 --port 8765 --device cuda --action_steps 32 \
  eval_embodiment=so100 \
  model.model_weights_to_bf16=true \
  model.use_torch_compile=false \
  model.model_arch.attn_implementation=sdpa
