#!/bin/bash
# Finetune entry for VLA training.
#
# Usage:
#   # Single-node
#   bash scripts/run/finetune.sh <num_gpus> <task_config> [options...] [hydra_overrides...]
#
#   # Multi-node / cluster env mode (GPU count from $NPROC_PER_NODE or $MLP_WORKER_GPU)
#   bash scripts/run/finetune.sh <task_config> [options...] [hydra_overrides...]
#
# Positional args:
#   <num_gpus>
#       Number of local GPUs passed to torchrun `--nproc-per-node`.
#   <task_config>
#       Supports all of the following forms as long as the yaml is under
#       `configs/task/`:
#         - Hydra shorthand: `libero`
#         - repo-relative yaml: `configs/task/libero.yaml`
#         - task-relative yaml: `task/libero.yaml`
#         - absolute yaml: `/abs/path/to/configs/task/libero.yaml`
#
# Options:
#   --test
#       Quick debug mode. Sets `EXP_NAME=test`, `logger.mode=offline`,
#       `MAX_EMBODIMENTS=17`, and defaults `MAX_DATASETS=1` unless overridden.
#   --dry-run
#       Print the resolved Hydra config and exit without training.
#       Implies --test (offline logging, truncated data).
#   --max_datasets N
#       Keep only the first N `dataset_dirs` for each dataset group.
#       This also truncates VLM dataset lists for faster startup.
#   --overfit_batch N
#       Friendly wrapper for Hydra `++overfit_batch=N`.
#   --overfit_mode MODE
#       Friendly wrapper for Hydra `+overfit_mode=MODE`. Default is
#       `per_dataset`, so multi-dataset overfit pins one stable subset per
#       inner dataset unless you explicitly override it.
#   --dataset PATH
#       Override task data with a single dataset yaml under `configs/data/`.
#       Example: `libero` or `configs/data/libero.yaml`.
#   --mixture PATH
#       Override task data with a data yaml under `configs/data/`.
#       Example: `libero`.
#
# Hydra overrides:
#   Any trailing `key=value` or `+key=value` / `++key=value` arguments are passed
#   through to Hydra unchanged. Common training knobs above also support `--...`
#   wrappers so you do not need to remember Hydra's `+` / `++` prefixes.
#
# Common examples:
#   # Standard shorthand task
#   bash scripts/run/finetune.sh 1 libero --overfit_batch 5
#
#   # Task yaml path + fast debug loading
#   bash scripts/run/finetune.sh 1 configs/task/libero.yaml --max_datasets 1
#
#   # Data override + per-dataset overfit for multi-dataset training
#   bash scripts/run/finetune.sh 1 libero \
#       --mixture libero \
#       --max_datasets 1 --overfit_batch 1

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

resolve_task_config() {
  local input="$1"
  local resolved=""
  local rel=""
  local task_root="$PROJECT_ROOT/configs/task"
  local normalized="$input"

  normalized="${normalized#./}"
  normalized="${normalized#configs/task/}"
  normalized="${normalized#task/}"
  normalized="${normalized#configs/}"

  local candidates=()

  if [[ "$input" == /* ]]; then
    candidates+=("$input")
    if [[ "$input" != *.yaml ]]; then
      candidates+=("${input}.yaml")
    fi
  else
    candidates+=(
      "$input"
      "$PROJECT_ROOT/$input"
      "$task_root/$input"
      "$task_root/$normalized"
    )
    if [[ "$input" != *.yaml ]]; then
      candidates+=(
        "${input}.yaml"
        "$PROJECT_ROOT/${input}.yaml"
        "$task_root/${input}.yaml"
        "$task_root/${normalized}.yaml"
      )
    fi
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      resolved="$(realpath "$candidate")"
      break
    fi
  done

  if [[ -n "$resolved" ]]; then
    if [[ "$resolved" != "$task_root/"* ]]; then
      echo "Error: task config yaml must be under $task_root"
      echo "Got: $resolved"
      exit 1
    fi

    rel="${resolved#$task_root/}"
    rel="${rel%.yaml}"
    echo "$rel"
    return
  fi

  rel="$normalized"
  rel="${rel%.yaml}"
  echo "$rel"
}

if [[ $# -lt 1 ]]; then
  echo "Error: Insufficient arguments"
  echo "Usage:"
  echo "  Single-node: $0 <num_gpus> <task_config> [options...]"
  echo "  Multi-node:  $0 <task_config> [options...]"
  echo "               (GPU count from \$NPROC_PER_NODE / \$MLP_WORKER_GPU or local GPU detection)"
  echo ""
  echo "Options:"
  echo "  --test              Run in test mode (offline logging)"
  echo "  --dry-run           Print resolved config and exit (implies --test)"
  echo "  --max_datasets N    Truncate each dataset config to first N dataset_dirs"
  echo "  --overfit_batch N   Set overfit batch count (wrapper for Hydra override)"
  echo "  --overfit_mode M    Set overfit mode, default is per_dataset"
  echo "  --dataset PATH      Data config (e.g. libero or configs/data/libero.yaml)"
  echo "  --mixture PATH      Data config (same accepted format as --dataset)"
  echo "  hydra_override      Any Hydra config override (key=value)"
  echo ""
  echo "Examples:"
  echo "  # Use task shorthand"
  echo "  $0 1 libero --dataset libero --test"
  echo "  $0 libero --dataset libero --test"
  echo ""
  echo "  # Use task yaml path"
  echo "  $0 1 configs/task/libero.yaml --mixture libero --test"
  exit 1
fi

if [[ "$1" =~ ^[0-9]+$ ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Error: Missing <task_config>"
    exit 1
  fi
  GPU=$1
  config=$2
  shift 2
else
  GPU="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
  if [[ -z "$GPU" || "$GPU" -lt 1 ]]; then
    GPU=1
  fi
  config=$1
  shift 1
fi

config="$(resolve_task_config "$config")"

# Handle --test flag:
# - export EXP_NAME=test
# - append logger.mode=offline (unless user already set logger.mode=...)
TEST_MODE=0
DRY_RUN_MODE=0
HAS_LOGGER_MODE=0
MAX_DATASETS_VALUE=""
OVERFIT_BATCH_VALUE=""
OVERFIT_MODE_VALUE=""
DATASET_PATH=""
MIXTURE_PATH=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)
      TEST_MODE=1
      shift
      ;;
    --dry-run)
      DRY_RUN_MODE=1
      TEST_MODE=1
      shift
      ;;
    --max_datasets=*)
      MAX_DATASETS_VALUE="${1#--max_datasets=}"
      shift
      ;;
    --max_datasets)
      MAX_DATASETS_VALUE="$2"
      shift 2
      ;;
    --overfit_batch=*)
      OVERFIT_BATCH_VALUE="${1#--overfit_batch=}"
      shift
      ;;
    --overfit_batch)
      OVERFIT_BATCH_VALUE="$2"
      shift 2
      ;;
    --overfit_mode=*)
      OVERFIT_MODE_VALUE="${1#--overfit_mode=}"
      shift
      ;;
    --overfit_mode)
      OVERFIT_MODE_VALUE="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET_PATH="${1#--dataset=}"
      shift
      ;;
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --mixture=*)
      MIXTURE_PATH="${1#--mixture=}"
      shift
      ;;
    --mixture)
      MIXTURE_PATH="$2"
      shift 2
      ;;
    logger.mode=*)
      HAS_LOGGER_MODE=1
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$MAX_DATASETS_VALUE" ]]; then
  export MAX_DATASETS="$MAX_DATASETS_VALUE"
fi

if [[ -n "$OVERFIT_BATCH_VALUE" ]]; then
  ARGS+=("++overfit_batch=$OVERFIT_BATCH_VALUE")
fi

if [[ -n "$OVERFIT_MODE_VALUE" ]]; then
  ARGS+=("+overfit_mode=$OVERFIT_MODE_VALUE")
fi

if [[ $DRY_RUN_MODE -eq 1 ]]; then
  export DRY_RUN=1
fi

if [[ $TEST_MODE -eq 1 ]]; then
  export EXP_NAME=test
  if [[ -z "$MAX_DATASETS_VALUE" ]]; then
    export MAX_DATASETS=1
  fi
  export MAX_EMBODIMENTS=5
  if [[ $HAS_LOGGER_MODE -eq 0 ]]; then
    ARGS+=("logger.mode=offline")
  fi
fi

# Handle data override using Hydra config group syntax
if [[ -n "$DATASET_PATH" && -n "$MIXTURE_PATH" ]]; then
  echo "Error: --dataset and --mixture are mutually exclusive"
  exit 1
fi

if [[ -n "$DATASET_PATH" ]]; then
  # Normalize path: remove configs/data/ prefix and .yaml suffix
  DATASET_PATH="${DATASET_PATH#configs/data/}"
  DATASET_PATH="${DATASET_PATH%.yaml}"
  # Set environment variable for finetune.py to handle dataset override
  export OVERRIDE_DATASET="$DATASET_PATH"
  echo "[Data Override] Using dataset: $DATASET_PATH (will be patched in Python)"
elif [[ -n "$MIXTURE_PATH" ]]; then
  # Normalize path: remove configs/data/ prefix and .yaml suffix
  MIXTURE_PATH="${MIXTURE_PATH#configs/data/}"
  MIXTURE_PATH="${MIXTURE_PATH%.yaml}"
  ARGS+=("data=$MIXTURE_PATH")
  echo "[Data Override] Using mixture: $MIXTURE_PATH"
fi

MULTINODE=0
NNODES="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
TORCHRUN_ARGS=()

if [[ "$NNODES" =~ ^[0-9]+$ ]] && [[ "$NNODES" -gt 1 ]]; then
  MULTINODE=1
  NODE_RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
  MASTER_ADDR_VAL="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
  MASTER_PORT_VAL="${MASTER_PORT:-${MLP_WORKER_0_PORT:-23456}}"
  TORCHRUN_ARGS=(
    --nproc-per-node "$GPU"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR_VAL"
    --master_port "$MASTER_PORT_VAL"
  )
else
  TORCHRUN_ARGS=(
    --standalone
    --nnodes 1
    --nproc-per-node "$GPU"
  )
fi

if [[ $MULTINODE -eq 1 ]]; then
  echo "[Launch Mode] multi-node: nnodes=$NNODES, node_rank=$NODE_RANK, master=$MASTER_ADDR_VAL:$MASTER_PORT_VAL, gpus_per_node=$GPU"
else
  echo "[Launch Mode] single-node: gpus=$GPU"
fi

python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" scripts/finetune.py "task=$config" "${ARGS[@]}"
