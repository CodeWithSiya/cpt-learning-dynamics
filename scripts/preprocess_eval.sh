#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name="cpt-preprocess-eval"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/preprocess_eval_%j.log
#SBATCH --error=logs/preprocess_eval_%j.log

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

# Preprocess NER evaluation dataset
uv run python src/data/preprocess_eval.py \
    --model-config configs/models/xlmr.yaml \
    --eval-config configs/eval/ner.yaml \
    --language xho \
    --output datasets/eval/processed/xlmr/xho/ner \
    --nproc ${SLURM_CPUS_PER_TASK}

# Preprocess POS evaluation dataset
uv run python src/data/preprocess_eval.py \
    --model-config configs/models/xlmr.yaml \
    --eval-config configs/eval/pos.yaml \
    --language xho \
    --output datasets/eval/processed/xlmr/xho/pos \
    --nproc ${SLURM_CPUS_PER_TASK}

# Preprocess NTC evaluation dataset
uv run python src/data/preprocess_eval.py \
    --model-config configs/models/xlmr.yaml \
    --eval-config configs/eval/ntc.yaml \
    --language xho \
    --output datasets/eval/processed/xlmr/xho/ntc \
    --nproc ${SLURM_CPUS_PER_TASK}