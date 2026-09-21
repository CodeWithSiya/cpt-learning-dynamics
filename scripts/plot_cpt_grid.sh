#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-plot-cpt-grid"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/plot_cpt_grid_%j.log
#SBATCH --error=logs/plot_cpt_grid_%j.log

git pull
git log -1

export UV_LINK_MODE=copy

set -a
source /home/mdnsiy014/cpt-learning-dynamics/.env
set +a

export SCRATCH=/home/mdnsiy014/scratch

module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

echo "=== Plotting CPT loss grid ==="

uv run python src/visualisation/plot_cpt_grid.py \
    --results-dir ${SCRATCH}/cpt-learning-dynamics/results \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/plots/cpt_grid \
    "$@"
