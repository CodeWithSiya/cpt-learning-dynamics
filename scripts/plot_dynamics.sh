#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-plot-dynamics"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_dynamics_%j.log
#SBATCH --error=logs/plot_dynamics_%j.log

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
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")

# All language subsets available for plotting
ALL_LANGUAGES=("xho" "zul")

# Evaluation tasks available for each language
declare -A LANGUAGE_TASKS=(
    ["xho"]="ner pos ntc_xho"
    ["zul"]="ner pos ntc_zul"
)

# Display names used in plot titles
declare -A MODEL_DISPLAY_NAMES=(
    ["roberta"]="RoBERTa"
    ["xlmr"]="XLMR"
    ["nguni-xlmr"]="Nguni-XLMR"
    ["afriberta"]="AfriBERTa"
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
        # Aggregate results across seeds, one file per task
        for task in ${LANGUAGE_TASKS[$language]}; do
            echo "=== Aggregating ${task} (${language}) results for ${model} ==="

            uv run python src/finetuning/aggregate.py \
                --results-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/finetuning \
                --task ${task} \
                --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/aggregated/${task}_aggregated.json
        done

        # Plot learning dynamics from the aggregated results
        echo "=== Plotting learning dynamics for ${model} (${language}) ==="

        uv run python src/visualisation/plot_dynamics.py \
            --aggregated-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/aggregated \
            --model-name "${MODEL_DISPLAY_NAMES[$model]}" \
            --output-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/plots/dynamics
    done
done