#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-aggregate"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/aggregate_%j.log
#SBATCH --error=logs/aggregate_%j.log

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

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# All models available for aggregation
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All evaluation tasks
ALL_TASKS=("ner" "pos" "ntc")

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

for model in "${MODELS[@]}"; do
    for task in "${ALL_TASKS[@]}"; do
        echo "=== Aggregating ${task} results for ${model} across seeds ==="

        uv run python src/finetuning/aggregate.py \
            --results-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/finetuning \
            --task ${task} \
            --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/aggregated/${task}_aggregated.json
    done
done