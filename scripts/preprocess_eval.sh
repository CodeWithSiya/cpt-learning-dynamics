#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
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

# All models available for preprocessing
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr")

# All evaluation tasks
ALL_TASKS=("ner" "pos" "ntc")

# Language subset to preprocess
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
        echo "=== Preprocessing ${task} eval dataset for model: ${model} ==="

        uv run python src/data/preprocess_eval.py \
            --model-config configs/models/${model}.yaml \
            --task-config configs/evaluation/${task}.yaml \
            --language ${LANGUAGE} \
            --output datasets/processed/evaluation/${model}/${LANGUAGE}/${task} \
            --nproc ${SLURM_CPUS_PER_TASK}
    done
done