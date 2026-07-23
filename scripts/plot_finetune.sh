#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name="cpt-plot-finetune"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_finetune_%j.log
#SBATCH --error=logs/plot_finetune_%j.log

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

# All models available for plotting
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr")

# All evaluation tasks
ALL_TASKS=("ner" "pos" "ntc")

# Seeds available for plotting
ALL_SEEDS=(42)

# Display names used in plot titles
declare -A MODEL_DISPLAY_NAMES=(
    ["roberta"]="RoBERTa"
    ["xlmr"]="XLMR"
    ["nguni-xlmr"]="Nguni-XLMR"
)

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Second script argument selects a single task; if omitted, loop through all tasks
TASK_ARG="$2"
if [ -n "${TASK_ARG}" ]; then
    TASKS=("${TASK_ARG}")
else
    TASKS=("${ALL_TASKS[@]}")
fi

# Third script argument selects a single seed; if omitted, loop through all seeds
SEED_ARG="$3"
if [ -n "${SEED_ARG}" ]; then
    SEEDS=("${SEED_ARG}")
else
    SEEDS=("${ALL_SEEDS[@]}")
fi

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "=== Plotting fine-tuning grid for ${model}, ${task}, seed ${seed} ==="

            uv run python src/visualisation/plot_finetune.py \
                --results-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/finetuning \
                --task ${task} \
                --seed ${seed} \
                --model-name "${MODEL_DISPLAY_NAMES[$model]}" \
                --output-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/plots
        done
    done
done