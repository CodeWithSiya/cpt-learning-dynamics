#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
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
export DATA_DIR=${SCRATCH}/cpt-learning-dynamics/datasets
mkdir -p "${HF_HOME}" "${DATA_DIR}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for CPT
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All language subsets to train on
ALL_LANGUAGES=("xho" "zul")

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Second script argument selects a single language; if omitted, loop through all languages
LANGUAGE_ARG="$2"
if [ -n "${LANGUAGE_ARG}" ]; then
    LANGUAGES=("${LANGUAGE_ARG}")
else
    LANGUAGES=("${ALL_LANGUAGES[@]}")
fi

# W&B run ID prefixes per model, suffixed with the language below, so resubmitted jobs resume the same run
declare -A WANDB_RUN_IDS=(
    ["roberta"]="roberta-large-cpt-200k"
    ["xlmr"]="xlmr-large-cpt-200k"
    ["nguni-xlmr"]="nguni-xlmr-large-200k"
    ["afriberta"]="afriberta-large-cpt-200k"
)

for model in "${MODELS[@]}"; do
    for language in "${LANGUAGES[@]}"; do
        echo "=== Running CPT for model: ${model} (${language}) ==="

        uv run accelerate launch \
            --num_processes ${SLURM_GPUS_ON_NODE:-1} \
            --num_machines 1 \
            --dynamo_backend no \
            --mixed_precision bf16 \
            --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
            src/pretraining/pretrain.py \
            --model-config configs/models/${model}.yaml \
            --train-corpus ${DATA_DIR}/processed/corpus/${model}/${language}/train \
            --validation-corpus ${DATA_DIR}/processed/corpus/${model}/${language}/validation \
            --output-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language} \
            --wandb-run-id "${WANDB_RUN_IDS[$model]}-${language}"
    done
done