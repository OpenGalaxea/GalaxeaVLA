#!/bin/bash
# This script is used to eval model in a open-loop manner
# arguments:
#   task config
#   checkpoint path
#   other hydra overrides

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

config=$1
ckpt_path=$2
ARGS=${@:3}

config="${config#configs/}" # delete prefix configs/
config="${config#task/}" # delete prefix task/
config="${config%.yaml}" # delete suffix .yaml

python scripts/eval_open_loop.py task=$config ckpt_path=$ckpt_path $ARGS
