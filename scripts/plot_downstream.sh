#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-plot-downstream"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_downstream_%j.log
#SBATCH --error=logs/plot_downstream_%j.log

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

# All models and languages available for plotting
ALL_MODELS=("roberta" "xlmr" "nguni-xlmr" "afriberta")
ALL_LANGUAGES=("xho" "zul")

# Evaluation tasks available for each language
declare -A LANGUAGE_TASKS=(
    ["xho"]="ner pos ntc_xho"
    ["zul"]="ner pos ntc_zul"
)

# Aggregate across seeds first, since the figures read the aggregated results
for model in "${ALL_MODELS[@]}"; do
    for language in "${ALL_LANGUAGES[@]}"; do
        for task in ${LANGUAGE_TASKS[$language]}; do
            echo "=== Aggregating ${task} (${language}) results for ${model} ==="

            uv run python src/finetuning/aggregate.py \
                --results-dir ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/finetuning \
                --task-config configs/evaluation/${task}.yaml \
                --output ${SCRATCH}/cpt-learning-dynamics/results/${model}-large/${language}/aggregated/${task}_aggregated.json
        done
    done
done

# Every figure here is a grid spanning all models and languages at once, so
# there is nothing to loop over; any subset is selected with the flags below.
echo "=== Plotting downstream result grids ==="

uv run python src/visualisation/plot_downstream.py \
    --results-dir ${SCRATCH}/cpt-learning-dynamics/results \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/plots/downstream \
    "$@"
