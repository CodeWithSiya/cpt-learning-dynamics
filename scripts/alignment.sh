#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name="cpt-alignment"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/alignment_%j.log
#SBATCH --error=logs/alignment_%j.log

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# HPC paths
export SCRATCH=/home/mdnsiy014/scratch
export HF_HOME=${SCRATCH}/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export DATA_DIR=${SCRATCH}/cpt-learning-dynamics/datasets
mkdir -p "${HF_HOME}" "${DATA_DIR}"

# Load environment variables
set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for alignment computation
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All language subsets to evaluate on
ALL_LANGUAGES=("xho" "zul")

# FLORES-200 language codes
declare -A FLORES_CODES=(
    ["xho"]="xho_Latn"
    ["zul"]="zul_Latn"
)

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
        flores_code="${FLORES_CODES[$language]}"
        echo "=== Computing cross-lingual alignment for ${model} (${language}) ==="

        uv run python src/evaluation/alignment.py \
            --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/checkpoints \
            --flores-dir ${DATA_DIR}/raw/flores \
            --language ${flores_code} \
            --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/alignment/${flores_code}_alignment.json \
            --batch-size 64
    done
done