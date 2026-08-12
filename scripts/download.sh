#!/bin/bash

#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name="cpt-download-data"
#SBATCH --mail-user=mdnsiy014@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.log

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
export DATA_DIR=${SCRATCH}/cpt-learning-dynamics/datasets
mkdir -p "${HF_HOME}" "${DATA_DIR}"

# Load Python and synchronise uv environment 
module load python/miniconda3-py3.12
cd /home/mdnsiy014/cpt-learning-dynamics
uv sync --frozen

# Download WURA isiXhosa corpus
uv run python src/data/download_wura.py \
    --language xho \
    --output-dir ${DATA_DIR}/raw/corpus

# Download WURA isiZulu corpus
uv run python src/data/download_wura.py \
    --language zul \
    --output-dir ${DATA_DIR}/raw/corpus

# Download all isiXhosa evaluation datasets
uv run python src/data/download_eval.py \
    --language xho \
    --output-dir ${DATA_DIR}/raw/evaluation

# Download all isiZulu evaluation datasets
uv run python src/data/download_eval.py \
    --language zul \
    --output-dir ${DATA_DIR}/raw/evaluation

# Download FLORES-200 isiXhosa dataset
uv run python src/data/download_flores.py \
    --language xho_Latn \
    --output-dir ${DATA_DIR}/raw/flores

# Download FLORES-200 isiZulu dataset
uv run python src/data/download_flores.py \
    --language zul_Latn \
    --output-dir ${DATA_DIR}/raw/flores

# Download FLORES-200 English-isiXhosa parallel dataset
uv run python src/data/download_flores.py \
    --language eng_Latn-xho_Latn \
    --output-dir ${DATA_DIR}/raw/flores

# Download FLORES-200 English-isiZulu parallel dataset
uv run python src/data/download_flores.py \
    --language eng_Latn-zul_Latn \
    --output-dir ${DATA_DIR}/raw/flores