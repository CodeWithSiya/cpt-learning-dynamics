#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=47:59:00
#SBATCH --job-name="cpt-push-checkpoints"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/push_checkpoints_%j.log
#SBATCH --error=logs/push_checkpoints_%j.log

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables from .env (HF_TOKEN)
set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

# HPC paths
export SCRATCH=/home/mdnsiy014/scratch
export HF_HOME=${SCRATCH}/hf
mkdir -p "${HF_HOME}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for CPT
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All language subsets to push
ALL_LANGUAGES=("xho" "zul")

# Hub repo owner/org and naming prefix per model
HF_NAMESPACE="your-username"
declare -A REPO_PREFIXES=(
    ["roberta"]="roberta-large-cpt"
    ["xlmr"]="xlmr-large-cpt"
    ["nguni-xlmr"]="nguni-xlmr-large-cpt"
    ["afriberta"]="afriberta-large-cpt"
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
        echo "=== Pushing ${model} (${language}) checkpoints to the Hub ==="

        uv run python src/pretraining/push_to_hub.py \
            --checkpoint-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/checkpoints \
            --repo-id "${HF_NAMESPACE}/${REPO_PREFIXES[$model]}-${language}"
    done
done
