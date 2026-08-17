#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name="cpt-plot-loss"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_cpt_%j.log
#SBATCH --error=logs/plot_cpt_%j.log

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

# All language subsets to plot
ALL_LANGUAGES=("xho" "zul")

# First script argument selects a single model; if omitted, plot all models
MODEL_ARG="$1"
if [ -n "${MODEL_ARG}" ]; then
    MODELS=("${MODEL_ARG}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Second script argument selects a single language; if omitted, plot all languages
LANGUAGE_ARG="$2"
if [ -n "${LANGUAGE_ARG}" ]; then
    LANGUAGES=("${LANGUAGE_ARG}")
else
    LANGUAGES=("${ALL_LANGUAGES[@]}")
fi

# Plot CPT loss curves for all models, for every loss
for language in "${LANGUAGES[@]}"; do
    echo "=== Plotting CPT loss curves for ${MODELS[@]} (${language}) ==="

    uv run python src/visualisation/plot_cpt.py \
        --results-dir ${SCRATCH}/cpt-learning-dynamics/results \
        --models "${MODELS[@]}" \
        --language ${language} \
        --output-dir ${SCRATCH}/cpt-learning-dynamics/results/plots/cpt
done