# Source code

The `src/` directory contains six packages. The YAML files in [`configs/`](../configs) provide their settings, and the batch scripts in [`scripts/`](../scripts) call their entry points.

| Package          | Role                                                                      |
| ---------------- | ------------------------------------------------------------------------- |
| `data/`          | Download and preprocess the CPT corpus and downstream evaluation datasets |
| `pretraining/`   | Continued pretraining (CPT) and the two-phase checkpoint schedule         |
| `geometry/`      | Cross-lingual alignment and representation isotropy from FLORES-200       |
| `finetuning/`    | Fine-tune CPT checkpoints on downstream tasks and aggregate across seeds  |
| `visualisation/` | Panel-grid figures shared across every result type                        |
| `utils/`         | Small helpers shared across the packages above                            |

## `data/`

Downloads the WURA corpus and downstream evaluation datasets, then prepares them for training and evaluation. `preprocess_eval.py` checks that each task config's `label_names` follow the dataset's label order before preprocessing.

## `pretraining/`

`pretrain.py` runs masked language modelling from a `ModelConfig`. It saves `step-N` checkpoints using the two-phase schedule in `schedule.py`: frequent checkpoints during the first 10% of training and less frequent checkpoints afterward. Hugging Face Trainer resumption saves use separate `checkpoint-N` directories.

## `geometry/`

This package computes cross-lingual alignment using mean cosine similarity between matched English and target-language sentence embeddings from FLORES-200, as well as representation isotropy with IsoScore. `alignment.py` covers every checkpoint. `layerwise.py` evaluates a selected subset at every hidden layer, while `embeddings.py` contains shared extraction and metric functions.

## `finetuning/`

`finetune.py` fine-tunes a task-specific head from a CPT checkpoint and evaluates it on the test split. Results are stored in seed-specific folders. `aggregate.py` calculates metric means and standard deviations across seeds.

## `visualisation/`

The plotting scripts produce panel-grid figures for CPT loss (`plot_cpt_grid.py`), downstream F1 (`plot_downstream.py`), alignment and isotropy (`plot_geometry.py`), and per-checkpoint fine-tuning loss (`plot_finetune.py`). They share typography, colours, and panel sizing through `style.py`.

## `utils/`

`extract.py` parses checkpoint steps and discovers `step-N` directories for the pretraining, geometry, and fine-tuning scripts. `reproducibility.py` contains the determinism settings shared by `pretrain.py` and `finetune.py`.

## Usage

Scripts under `data/`, `pretraining/`, `geometry/`, and `finetuning/` are standalone `argparse` entry points. Use `--help` to see the available flags. Plotting scripts need `src/visualisation` on `PYTHONPATH` because they import `style` directly:

```bash
uv run python src/pretraining/pretrain.py --help
PYTHONPATH=src/visualisation uv run python src/visualisation/plot_geometry.py --help
```

Run these commands from the repository root with `uv run`; the HPC scripts use the same pattern. If a local command cannot import `src`, rebuild the environment with `uv sync --reinstall` or set `PYTHONPATH=.` for that command.
