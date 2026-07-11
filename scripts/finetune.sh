#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --job-name="cpt-finetune"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.log

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables
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

# All models available for fine-tuning
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr")

# All evaluation tasks
ALL_TASKS=("ner" "pos" "ntc")

# Language subset being evaluated
LANGUAGE="xho"

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

for model in "${MODELS[@]}"; do
    for task in "${ALL_TASKS[@]}"; do
        echo "=== Fine-tuning and evaluating ${model} on ${task} ==="

        uv run accelerate launch \
            --num_processes ${SLURM_GPUS_ON_NODE:-1} \
            --num_machines 1 \
            --dynamo_backend no \
            --mixed_precision bf16 \
            --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
            src/evaluation/finetune.py \
            --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}/checkpoints \
            --eval-config configs/evaluation/${task}.yaml \
            --preprocessed-dir datasets/processed/evaluation/${model}/${LANGUAGE}/${task} \
            --output-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}/finetuning
    done
done