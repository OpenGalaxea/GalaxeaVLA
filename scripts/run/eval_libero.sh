#!/bin/bash
# One-click LIBERO batch evaluation: server + 4 task suite clients + summary
#
# Usage:
#   bash scripts/run/eval_libero.sh <ckpt_path> [options]
#
# Options:
#   --output_dir DIR        Root output dir (default: outputs/<ckpt_basename>)
#   --port PORT             Server port (default: 12345)
#   --num_trials N          Trials per task (default: 10)
#   --num_parallel N        Parallel envs per client (default: 5)
#   --num_steps_wait N      Warmup steps (default: 20)
#   --max_batch_size N      Server max batch (default: 30)
#   --max_wait_ms N         Server max wait ms (default: 2000)
#   --env_resolution N      Camera resolution (default: 256)
#   --save_videos           Save rollout videos
#   --suites SUITES         Space-separated suite names (default: libero_goal libero_spatial libero_object libero_10)
#   Any additional key=value pairs are forwarded to the server as Hydra overrides
#
# Example:
#   bash scripts/run/eval_libero.sh \
#       runs/pretrain/libero/libero_g05_qwen_no_6d/2026-05-08_12-45-07/last.pt \
#       eval_embodiment=libero --action_steps 10 model.model_arch.discrete_action=false

set -euo pipefail

# ── Parse args ──
CKPT_PATH=""
OUTPUT_DIR=""
PORT=12345
NUM_TRIALS=50
NUM_PARALLEL=10
NUM_STEPS_WAIT=20
MAX_BATCH_SIZE=30
MAX_WAIT_MS=1000
ENV_RESOLUTION=256
SAVE_VIDEOS=false
SUITES=(libero_goal libero_spatial libero_object libero_10)
OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --num_trials)   NUM_TRIALS="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --num_steps_wait) NUM_STEPS_WAIT="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_wait_ms)  MAX_WAIT_MS="$2"; shift 2 ;;
        --env_resolution) ENV_RESOLUTION="$2"; shift 2 ;;
        --save_videos)  SAVE_VIDEOS=true; shift ;;
        --suites)       read -ra SUITES <<< "$2"; shift 2 ;;
        *=*)            OVERRIDES+=("$1"); shift ;;
        *)
            if [[ -z "$CKPT_PATH" ]]; then
                CKPT_PATH="$1"
            else
                echo "Unknown argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts/run/eval_libero.sh <ckpt_path> [options]" >&2
    echo "  --output_dir, --port, --num_trials, --num_parallel, --num_steps_wait," >&2
    echo "  --max_batch_size, --max_wait_ms, --save_videos, --suites" >&2
    exit 1
fi

# Derive output dir from checkpoint name if not set
if [[ -z "$OUTPUT_DIR" ]]; then
    CKPT_BASENAME=$(basename "$(dirname "$CKPT_PATH")")
    OUTPUT_DIR="outputs/libero_eval_${CKPT_BASENAME}"
fi
mkdir -p "$OUTPUT_DIR"

SERVER_URI="ws://127.0.0.1:${PORT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo " LIBERO Batch Evaluation"
echo "=============================================="
echo " Checkpoint:    $CKPT_PATH"
echo " Output dir:    $OUTPUT_DIR"
echo " Server:        $SERVER_URI"
echo " Suites:        ${SUITES[*]}"
echo " Trials/task:   $NUM_TRIALS"
echo " Parallel:      $NUM_PARALLEL"
echo " Steps wait:    $NUM_STEPS_WAIT"
echo " Max batch:     $MAX_BATCH_SIZE"
echo " Max wait ms:   $MAX_WAIT_MS"
echo " Env resolution: $ENV_RESOLUTION"
echo " Save videos:   $SAVE_VIDEOS"
echo " Overrides:     ${OVERRIDES[*]}"
echo "=============================================="

# ── Source environment ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Try sourcing startg05.sh if it exists
if [[ -f "$PROJECT_ROOT/startg05.sh" ]]; then
    source "$PROJECT_ROOT/startg05.sh"
fi

cd "$PROJECT_ROOT"

# ── Start server in background ──
echo ""
echo "[1/3] Starting batched policy server on port $PORT ..."
python scripts/serve_policy_batched.py \
    --ckpt_path "$CKPT_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    eval_embodiment=libero \
    --action_steps 10 \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_wait_ms "$MAX_WAIT_MS" \
    "${OVERRIDES[@]}" \
    &> "$OUTPUT_DIR/server.log" &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID (log: $OUTPUT_DIR/server.log)"

# Wait for server to be ready (check port in LISTEN state via ss)
echo "  Waiting for server to be ready ..."
SERVER_READY=false
for i in $(seq 1 300); do
    if ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT"; then
        echo "  Server is ready!"
        SERVER_READY=true
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died! Check $OUTPUT_DIR/server.log" >&2
        exit 1
    fi
    sleep 2
done
if [[ "$SERVER_READY" != "true" ]]; then
    echo "ERROR: Server not ready after 600s. Check $OUTPUT_DIR/server.log" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# ── Launch all clients in parallel ──
echo ""
echo "[2/3] Launching ${#SUITES[@]} client(s) in parallel ..."

declare -A CLIENT_PIDS
for suite in "${SUITES[@]}"; do
    SUITE_OUT="$OUTPUT_DIR/$suite"
    mkdir -p "$SUITE_OUT"

    VIDEO_FLAG=""
    if [[ "$SAVE_VIDEOS" == "true" ]]; then
        VIDEO_FLAG="--save_videos"
    fi

    echo "  Starting client: $suite -> $SUITE_OUT"
    python experiments/libero/eval_libero_parallel.py \
        --server_uri "$SERVER_URI" \
        --task_suite_name "$suite" \
        --num_trials_per_task "$NUM_TRIALS" \
        --num_steps_wait "$NUM_STEPS_WAIT" \
        --num_parallel "$NUM_PARALLEL" \
        --env_resolution "$ENV_RESOLUTION" \
        --output_dir "$SUITE_OUT" \
        $VIDEO_FLAG \
        &> "$SUITE_OUT/client.log" &
    CLIENT_PIDS[$suite]=$!
done

# ── Wait for all clients ──
echo "  Waiting for all clients to finish ..."
FAILED_SUITES=()
for suite in "${SUITES[@]}"; do
    PID=${CLIENT_PIDS[$suite]}
    if wait "$PID"; then
        echo "  [DONE] $suite (PID $PID)"
    else
        echo "  [FAIL] $suite (PID $PID) — check $OUTPUT_DIR/$suite/client.log" >&2
        FAILED_SUITES+=("$suite")
    fi
done

# ── Stop server ──
echo ""
echo "  Stopping server (PID $SERVER_PID) ..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# ── Generate summary ──
echo ""
echo "[3/3] Generating summary ..."

python - <<'PYTHON_SCRIPT' "$OUTPUT_DIR" "${SUITES[*]}"
import json
import sys
import os
from pathlib import Path

output_dir = Path(sys.argv[1])
suite_names = sys.argv[2].split()

all_suite_results = {}
per_task_rows = []

for suite in suite_names:
    json_path = output_dir / suite / f"{suite}_parallel_results.json"
    if not json_path.exists():
        print(f"  WARNING: No results file for {suite} at {json_path}")
        continue

    with open(json_path) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    suite_successes = 0
    suite_total = 0

    for t in tasks:
        tid = t["task_id"]
        desc = t.get("task_description", "")
        succ = t["successes"]
        total = t["total_episodes"]
        rate = t["success_rate"]
        suite_successes += succ
        suite_total += total
        per_task_rows.append((suite, tid, desc, succ, total, rate))

    sr = suite_successes / suite_total if suite_total > 0 else 0.0
    all_suite_results[suite] = {
        "successes": suite_successes,
        "total": suite_total,
        "success_rate": sr,
    }

# ── Print per-task table ──
print("\n" + "=" * 90)
print("  LIBERO EVALUATION RESULTS — PER-TASK BREAKDOWN")
print("=" * 90)
header = f"{'Suite':<18} {'TaskID':>6}  {'Success':>7}  {'Total':>5}  {'Rate':>7}  Description"
print(header)
print("-" * 90)

current_suite = None
for suite, tid, desc, succ, total, rate in sorted(per_task_rows, key=lambda x: (x[0], x[1])):
    if suite != current_suite:
        if current_suite is not None:
            s = all_suite_results[current_suite]
            avg = s["success_rate"]
            print(f"  {'─' * 38}  Suite Avg: {avg:6.1%}  ({s['successes']}/{s['total']})")
            print()
        current_suite = suite
    desc_short = desc[:35] + "..." if len(desc) > 38 else desc
    print(f"{suite:<18} {tid:>6}  {succ:>5}/{total:<3d}  {total:>5}  {rate:>6.1%}  {desc_short}")

if current_suite is not None:
    s = all_suite_results[current_suite]
    avg = s["success_rate"]
    print(f"  {'─' * 38}  Suite Avg: {avg:6.1%}  ({s['successes']}/{s['total']})")

# ── Print suite summary ──
print("\n" + "=" * 60)
print("  SUITE SUMMARY")
print("=" * 60)
print(f"{'Suite':<18} {'Success':>8}  {'Total':>6}  {'Rate':>8}")
print("-" * 60)

grand_successes = 0
grand_total = 0
for suite in suite_names:
    if suite not in all_suite_results:
        continue
    s = all_suite_results[suite]
    print(f"{suite:<18} {s['successes']:>5}/{s['total']:<3d}  {s['total']:>6}  {s['success_rate']:>7.1%}")
    grand_successes += s["successes"]
    grand_total += s["total"]

print("-" * 60)
if grand_total > 0:
    grand_rate = grand_successes / grand_total
else:
    grand_rate = 0.0
print(f"{'OVERALL':<18} {grand_successes:>5}/{grand_total:<3d}  {grand_total:>6}  {grand_rate:>7.1%}")
print("=" * 60)

# ── Save structured summary ──
summary = {
    "ckpt_path": str(output_dir),
    "suites": all_suite_results,
    "grand_successes": grand_successes,
    "grand_total": grand_total,
    "grand_success_rate": grand_rate,
    "per_task": [
        {"suite": s, "task_id": t, "description": d, "successes": su, "total": to, "success_rate": r}
        for s, t, d, su, to, r in per_task_rows
    ],
}
summary_path = output_dir / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nSummary saved to {summary_path}")
PYTHON_SCRIPT

echo ""
echo "All done!"
if [[ ${#FAILED_SUITES[@]} -gt 0 ]]; then
    echo "WARNING: Failed suites: ${FAILED_SUITES[*]}" >&2
    exit 1
fi
