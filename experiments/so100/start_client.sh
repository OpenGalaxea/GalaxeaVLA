#!/usr/bin/env bash
# Start the SO100 robot client on the local machine connected to the arm, using
# the lerobot conda environment.
# Adjust --camera-index / --camera-map to match your lerobot camera names.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT="$HERE/../.."                       # repository root
# Python from the lerobot environment; adjust as needed. See environment.yml in this directory.
CONDA_PYTHON="${CONDA_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}"

# Note: the client indirectly imports g05.utils.websocket through
# scripts/utils/policy_ws_client.py, so PYTHONPATH must include both $PROJECT
# for scripts.* and $PROJECT/src for g05.*.
export PYTHONPATH="$PROJECT/src:$PROJECT:${PYTHONPATH:-}"

"$CONDA_PYTHON" "$HERE/so100_policy_client.py" \
  --host localhost \
  --port 8765 \
  --robot-port /dev/ttyACM0 \
  --robot-id g05_so100_follower \
  --camera-index exterior:4 wrist_right:0 \
  --camera-map exterior:exterior wrist_right:wrist_right \
  --action-fps 15 \
  --max-step-deg 10
