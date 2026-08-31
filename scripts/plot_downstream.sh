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

# Every figure here is a grid spanning all models and languages at once, so
# there is nothing to loop over; any subset is selected with the flags below.
echo "=== Plotting downstream result grids ==="

uv run python src/visualisation/plot_downstream.py \
    --results-dir ${SCRATCH}/cpt-learning-dynamics/results \
    --output-dir ${SCRATCH}/cpt-learning-dynamics/results/plots/downstream \
    "$@"
