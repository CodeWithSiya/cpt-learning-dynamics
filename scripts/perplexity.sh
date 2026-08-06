#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name="cpt-perplexity"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/perplexity_%j.log
#SBATCH --error=logs/perplexity_%j.log

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# HPC paths
export SCRATCH=/home/mdnsiy014/scratch
export HF_HOME=${SCRATCH}/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load environment variables
set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for perplexity computation
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# Language subset to evaluate on
LANGUAGE="xho_Latn"

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

for model in "${MODELS[@]}"; do
    echo "=== Computing pseudo-perplexity for ${model} ==="

    uv run python src/evaluation/perplexity.py \
        --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/checkpoints \
        --flores-dir datasets/raw/flores \
        --language ${LANGUAGE} \
        --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/perplexity/${LANGUAGE}_perplexity.json \
        --batch-size 32
done