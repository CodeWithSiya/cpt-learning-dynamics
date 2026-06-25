#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name="cpt-pretrain"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/pretrain_%j.log
#SBATCH --error=logs/pretrain_%j.log

# Sress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables from .env
set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

# Redirect HuggingFace cache to scratch
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# Launch CPT with accelerate for multi-GPU
accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    src/pretraining/pretrain.py \
    --config config/models/xlmr.yaml \
    --corpus datasets/corpus/xho