#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-plot-perplexity"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_perplexity_%j.log
#SBATCH --error=logs/plot_perplexity_%j.log

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

# Display names used in plot titles
declare -A MODEL_DISPLAY_NAMES=(
    ["roberta"]="RoBERTa"
    ["xlmr"]="XLMR"
    ["nguni-xlmr"]="Nguni-XLMR"
)

# Language subset to plot
LANGUAGE="xho_Latn"

# First script argument selects a single model; if omitted, loop through all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

for model in "${MODELS[@]}"; do
    echo "=== Plotting pseudo-perplexity dynamics for ${model} ==="

    uv run python src/visualisation/plot_perplexity.py \
        --perplexity-path ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/perplexity/${LANGUAGE}_perplexity.json \
        --model-name "${MODEL_DISPLAY_NAMES[$model]}" \
        --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/plots/${LANGUAGE}_perplexity.png
done
