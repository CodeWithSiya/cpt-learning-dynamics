#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name="cpt-preprocess-corpus"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/preprocess_corpus_%j.log
#SBATCH --error=logs/preprocess_corpus_%j.log

# Update to latest commit
git pull
git log -l

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

# Preprocess train split 
uv run python src/data/preprocess_corpus.py \
    --input datasets/corpus/xho \
    --model-config configs/models/xlmr.yaml \
    --language xho \
    --output datasets/processed/xlmr/xho/train \
    --split train \
    --nproc ${SLURM_CPUS_PER_TASK}

# Preprocess validation split 
uv run python src/data/preprocess_corpus.py \
    --input datasets/corpus/xho \
    --model-config configs/models/xlmr.yaml \
    --output datasets/processed/xlmr/xho/validation \
    --split validation \
    --nproc ${SLURM_CPUS_PER_TASK}