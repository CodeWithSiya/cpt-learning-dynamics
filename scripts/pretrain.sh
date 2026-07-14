#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=47:59:00
#SBATCH --job-name="cpt-pretrain"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/pretrain_%j.log
#SBATCH --error=logs/pretrain_%j.log

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables from .env
set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

# HPC paths
export SCRATCH=/home/mdnsiy014/scratch
export HF_HOME=${SCRATCH}/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for CPT
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr")

# Language subset to train on
LANGUAGE="xho"

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

for model in "${MODELS[@]}"; do
    echo "=== Running CPT for model: ${model} ==="

    uv run accelerate launch \
        --num_processes ${SLURM_GPUS_ON_NODE:-1} \
        --num_machines 1 \
        --dynamo_backend no \
        --mixed_precision bf16 \
        --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
        src/pretraining/pretrain.py \
        --model-config configs/models/${model}.yaml \
        --train-corpus datasets/processed/corpus/${model}/${LANGUAGE}/train \
        --validation-corpus datasets/processed/corpus/${model}/${LANGUAGE}/validation
done