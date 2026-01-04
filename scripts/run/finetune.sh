#!/bin/bash
# This script is used to finetune the model
# arguments:
#   GPU number
#   task config
#   other hydra overrides

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

GPU=$1
config=$2
ARGS=${@:3}

config="${config#configs/}" # delete prefix configs/
config="${config#task/}" # delete prefix task/
config="${config%.yaml}" # delete suffix .yaml

torchrun --standalone --nnodes 1 --nproc-per-node $GPU scripts/finetune.py task=$config $ARGS
