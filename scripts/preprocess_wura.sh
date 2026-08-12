#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --job-name="cpt-preprocess-corpus"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/preprocess_wura_%j.log
#SBATCH --error=logs/preprocess_wura_%j.log

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

# All models available for preprocessing
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All language subsets to preprocess
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

for model in "${MODELS[@]}"; do
    for language in "${LANGUAGES[@]}"; do
        echo "=== Preprocessing corpus for model: ${model} (${language}) ==="

        # Preprocess train split
        uv run python src/data/preprocess_wura.py \
            --input ${DATA_DIR}/raw/corpus/${language} \
            --model-config configs/models/${model}.yaml \
            --output ${DATA_DIR}/processed/corpus/${model}/${language}/train \
            --split train \
            --nproc ${SLURM_CPUS_PER_TASK}

        # Preprocess validation split
        uv run python src/data/preprocess_wura.py \
            --input ${DATA_DIR}/raw/corpus/${language} \
            --model-config configs/models/${model}.yaml \
            --output ${DATA_DIR}/processed/corpus/${model}/${language}/validation \
            --split validation \
            --nproc ${SLURM_CPUS_PER_TASK}
    done
done