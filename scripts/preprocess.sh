#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name="cpt-preprocess"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/preprocess_%j.log
#SBATCH --error=logs/preprocess_%j.log

# Suppress uv hardlink warning
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

# Preprocess train split 
uv run python src/data/preprocess.py \
    --input datasets/corpus/xho \
    --config configs/models/xlmr.yaml \
    --output datasets/processed/xlmr/xho/train \
    --split train \
    --nproc ${SLURM_CPUS_PER_TASK}

# Preprocess validation split 
uv run python src/data/preprocess.py \
    --input datasets/corpus/xho \
    --config configs/models/xlmr.yaml \
    --output datasets/processed/xlmr/xho/validation \
    --split validation \
    --nproc ${SLURM_CPUS_PER_TASK}