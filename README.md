# Continued Pretraining Learning Dynamics of Encoder-Only Models on South African Languages

This project studies how encoder-only language models change during continued pretraining (CPT) on isiXhosa and isiZulu. The repository contains the code for analysing downstream task performance, cross-lingual alignment, and representation geometry throughout training.

The pipeline trains `roberta`, `xlmr`, `nguni-xlmr`, and `afriberta` on each language. It saves checkpoints frequently early in training and less often later, then evaluates the checkpoints on named entity recognition, part-of-speech tagging, news topic classification, alignment with English, and representation isotropy using FLORES-200.

## Structure

| Path                  | Contents                                                                               |
| --------------------- | -------------------------------------------------------------------------------------- |
| [`src/`](src)         | Pipeline code and Python entry points                                                  |
| [`scripts/`](scripts) | SLURM batch scripts for each pipeline stage                                            |
| [`configs/`](configs) | YAML files for models, tasks, and fine-tuning                                          |
| `results/`            | Result JSON, figures, and logs generated during runs. This directory is not committed. |

## Setup

Clone the repository:

```bash
git clone https://github.com/CodeWithSiya/cpt-learning-dynamics.git
cd cpt-learning-dynamics
```

On the cluster, clone into your home directory and note the resulting path. The batch scripts `cd` into a hardcoded repository path and run `git pull` when a job starts, so that path must match wherever you cloned.

The project requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/), a fast Python package and environment manager. Follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then run this from the repository root:

```bash
uv sync --frozen
```

This creates or updates the project environment from `uv.lock`. Use `uv run ...` for project commands so they use that environment. The scripts also use `uv sync --frozen` when a job starts on the cluster.

Create a `.env` file in the repo root with:

```
HF_TOKEN=<huggingface token, for gated datasets and higher rate limits>
WANDB_API_KEY=<Weights & Biases API key, for run tracking>
```

The Hugging Face token is needed for gated datasets and higher download limits. The [Weights & Biases API key](https://docs.wandb.ai/models/quickstart) enables experiment tracking.

## Dependencies

Declared in [`pyproject.toml`](pyproject.toml) and pinned in [`uv.lock`](uv.lock), which resolves to 134 packages in total. The direct dependencies are:

| Package           | Pinned       | Used for                                                                  |
| ----------------- | ------------ | ------------------------------------------------------------------------- |
| `torch`           | 2.11.0+cu128 | Tensor and GPU backend for every training and evaluation stage            |
| `transformers`    | 5.12.1       | Model loading, tokenisers, and the `Trainer` used for CPT and fine-tuning |
| `datasets`        | 3.0.0        | Loading, tokenising, and saving corpora and evaluation datasets to disk   |
| `accelerate`      | 1.14.0       | Distributed launch for CPT across GPUs                                    |
| `evaluate`        | 0.4.6        | `seqeval` metric wrapper for span-level NER scoring                       |
| `seqeval`         | 1.2.2        | Span-level F1 for NER                                                     |
| `scikit-learn`    | 1.9.0        | Token and sequence-level F1, accuracy, and per-class reports              |
| `isoscore`        | 2.0.1        | Representation isotropy in the geometry stage                             |
| `numpy`           | 2.4.6        | Embedding arrays and metric aggregation across seeds                      |
| `scienceplots`    | 2.2.2        | Publication figure styling; also supplies `matplotlib`                    |
| `pyyaml`          | 6.0.3        | Reading the YAML configs in [`configs/`](configs)                         |
| `wandb`           | 0.27.2       | Experiment tracking for CPT and fine-tuning runs                          |
| `huggingface-hub` | 1.19.0       | Authenticated model and dataset downloads                                 |
| `sentencepiece`   | 0.2.2        | Tokeniser backend for XLM-R and AfriBERTa                                 |
| `tiktoken`        | 0.13.0       | Tokeniser backend required by `transformers`                              |
| `python-dotenv`   | 1.2.2        | Loading `HF_TOKEN` and `WANDB_API_KEY` from `.env`                        |
| `requests`        | 2.34.2       | Downloading the ANTC dataset, which is fetched as raw TSV                 |

## Running on an HPC cluster

The full pipeline is designed for a [SLURM](https://slurm.schedmd.com/overview.html)-based HPC system. GPU memory and run time make the training and evaluation stages unsuitable for a laptop. Before submitting a job, update the cluster-specific values in [`scripts/`](scripts): repository paths, your `.env` path, email address, account, partition, and resource requests.

Keep datasets, Hugging Face caches, checkpoints, logs, and generated results on the cluster's scratch filesystem. The scripts use `$SCRATCH`, `$HF_HOME`, `$HF_DATASETS_CACHE`, and `$DATA_DIR` for this purpose. Scratch paths are usually faster and have more capacity than home directories, but they may be cleaned periodically, so copy important final results elsewhere.

Run the stages in this order:

```
download → preprocess → pretrain → {alignment, layerwise, finetune} → aggregate → plot
```

Each stage is submitted with `sbatch`. Most scripts accept optional model and language arguments, which is useful for testing one part of the sweep:

```bash
sbatch scripts/download.sh
sbatch scripts/pretrain.sh roberta xho
```

See [`scripts/README.md`](scripts/README.md) for the dependency graph and argument patterns. See [`src/README.md`](src/README.md) for the Python entry points and their command-line interfaces.

## Main stages in detail

The main result-producing stages are continued pretraining, fine-tuning, and cross-lingual geometry. Each batch script wraps a Python entry point and sets its flags from the model, language, and task it is processing. The examples below show a complete command for one combination, with every flag supplied. Run a command directly to reproduce one cell of the sweep, or use it as a reference when editing the corresponding script.

Two variables appear throughout. `$DATA_DIR` holds downloaded and preprocessed datasets, and `$SCRATCH` holds checkpoints and results. Both are set near the top of each script.

### Continued pretraining

Trains one model on one language and writes checkpoints on the two-phase schedule.

```bash
uv run accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    src/pretraining/pretrain.py \
    --model-config configs/models/roberta.yaml \
    --train-corpus $DATA_DIR/processed/corpus/roberta/xho/train \
    --validation-corpus $DATA_DIR/processed/corpus/roberta/xho/validation \
    --output-dir $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho \
    --wandb-run-id roberta-large-cpt-200k-xho
```

| Flag                                     | Purpose                                                                                                                                            |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model-config`                         | Which base model to train and with what hyperparameters. Swap the file to change model.                                                            |
| `--train-corpus` / `--validation-corpus` | Preprocessed corpus from `preprocess_wura.sh`. The path encodes model and language because each model tokenises the corpus with its own tokeniser. |
| `--output-dir`                           | Where checkpoints and `log_history.json` are written. Checkpoints land in a `checkpoints/` subdirectory.                                           |
| `--wandb-run-id`                         | Resumes an existing Weights & Biases run so a requeued job continues the same chart. Omit it to start a fresh run.                                 |

Training runs under `accelerate` because CPT is distributed across GPUs. `--num_processes` should match the GPUs on the node; `pretrain.sh` reads it from `$SLURM_GPUS_ON_NODE`. Step count, learning rate, batch size, and the checkpoint schedule live in the model config, not on the command line — see [`configs/README.md`](configs/README.md).

### Fine-tuning

Fine-tunes every checkpoint from one CPT run on one task, at one seed.

```bash
uv run python src/finetuning/finetune.py \
    --checkpoint-dir $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/checkpoints \
    --task-config configs/evaluation/ner.yaml \
    --finetune-config configs/finetuning/ner.yaml \
    --preprocessed-dir $DATA_DIR/processed/evaluation/roberta/xho/ner \
    --output-dir $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/finetuning \
    --seed 42
```

| Flag                 | Purpose                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `--checkpoint-dir`   | The `checkpoints/` directory from the CPT run. Every `step-N` inside it is fine-tuned in turn. |
| `--task-config`      | Task definition: dataset columns, label set, and F1 variant.                                   |
| `--finetune-config`  | Fine-tuning hyperparameters for that task.                                                     |
| `--preprocessed-dir` | Tokenised evaluation dataset from `preprocess_eval.sh`, again per model and language.          |
| `--output-dir`       | Results are written under `step-N/<task>/seed-N/results.json`.                                 |
| `--seed`             | One run per seed. The five seeds are averaged later by `aggregate.sh`.                         |

To evaluate a different task, change `--task-config`, `--finetune-config`, and `--preprocessed-dir` together; all three must refer to the same task. After all seeds have finished, aggregate the results:

```bash
uv run python src/finetuning/aggregate.py \
    --results-dir $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/finetuning \
    --task-config configs/evaluation/ner.yaml \
    --output $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/aggregated/ner_aggregated.json
```

### Cross-Lingual Representation Geometry

Measures cross-lingual alignment and isotropy against English using FLORES-200. `layerwise.py` reports each hidden layer for a few checkpoints, and produces the data behind the geometry figure:

```bash
uv run python src/geometry/layerwise.py \
    --checkpoint-dir $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/checkpoints \
    --flores-dir $DATA_DIR/raw/flores \
    --language xho_Latn \
    --checkpoints 0,40000,100000,140000,200000 \
    --output $SCRATCH/cpt-learning-dynamics/results/roberta-large/xho/alignment/xho_Latn_layerwise.json \
    --batch-size 64
```

| Flag               | Purpose                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `--checkpoint-dir` | Same CPT checkpoints directory used for fine-tuning.                                                         |
| `--flores-dir`     | FLORES-200 root from `download.sh`. The script appends `eng_Latn-<language>` itself.                         |
| `--language`       | FLORES-200 code for the target language: `xho_Latn` or `zul_Latn`.                                           |
| `--checkpoints`    | Comma-separated steps to analyse. Every layer of every checkpoint is expensive, so this stage uses a subset. |
| `--batch-size`     | Sentences per forward pass. Lower it if a GPU runs out of memory.                                            |

`alignment.py` takes the same flags without `--checkpoints`, and measures every checkpoint at the final layer only.

## Reproducibility

CPT uses seed `42`. Fine-tuning uses five seeds (`42`, `123`, `456`, `789`, and `1738`), with results reported as means and standard deviations. [`uv.lock`](uv.lock) pins dependency versions. GPU determinism is enabled where supported, but bitwise results can still differ across GPU types.
