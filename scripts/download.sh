#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name="download-data"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Redirect HuggingFace cache to scratch
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME=${SCRATCH}/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load Python and synchronise uv environment 
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# Download WURA isiXhosa corpus
uv run python src/data/download_corpus.py \
    --language xho \
    --output_dir datasets/corpus

# Download all evaluation datasets
uv run python src/data/download_eval_data.py \
    --language xho \
    --output_dir datasets/eval