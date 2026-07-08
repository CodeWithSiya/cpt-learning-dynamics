#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

# Fine-tune and evaluate on NER
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    src/evaluation/finetune.py \
    --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/checkpoints \
    --eval-config configs/evaluation/ner.yaml \
    --preprocessed-dir datasets/eval/processed/xlmr/xho/ner \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/finetuning \
    --language xho

# Fine-tune and evaluate on POS
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    src/evaluation/finetune.py \
    --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/checkpoints \
    --eval-config configs/evaluation/pos.yaml \
    --preprocessed-dir datasets/eval/processed/xlmr/xho/pos \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/finetuning \
    --language xho

# Fine-tune and evaluate on NTC
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    src/evaluation/finetune.py \
    --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/checkpoints \
    --eval-config configs/evaluation/ntc.yaml \
    --preprocessed-dir datasets/eval/processed/xlmr/xho/ntc \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/xlmr/finetuning \
    --language xho