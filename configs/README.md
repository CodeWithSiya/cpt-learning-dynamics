# Configuration

YAML files define the models, datasets, and fine-tuning settings used by the pipeline. The code loads and validates each file when a run starts.

| Directory     | Loaded into                                   | Drives                                 |
| ------------- | --------------------------------------------- | -------------------------------------- |
| `models/`     | `ModelConfig` (`src/pretraining/config.py`)   | Continued pretraining (CPT)            |
| `evaluation/` | `TaskConfig` (`src/finetuning/config.py`)     | Dataset preprocessing and task metrics |
| `finetuning/` | `FinetuneConfig` (`src/finetuning/config.py`) | Downstream fine-tuning hyperparameters |

## `models/`

The directory contains one file for each base model: `roberta`, `xlmr`, `nguni-xlmr`, and `afriberta`. Each file specifies the Hugging Face checkpoint, masked-language-model training settings, and Weights & Biases run identity.

All four models train for 200,000 steps with an effective batch size of 80 (`per_device_batch_size: 10` and `gradient_accumulation_steps: 8`).

The `checkpoint_schedule` block controls the two-phase schedule in [`src/pretraining/schedule.py`](../src/pretraining/schedule.py). It saves checkpoints frequently during the first 10% of training and less frequently afterward. Missing fields use the defaults in `CheckpointScheduleConfig`.

Checkpoint output paths are passed per run via `--output-dir`, so the same model config is reused across both languages.

## `evaluation/`

Each downstream task has its own file: `ner`, `pos`, `ntc_xho`, and `ntc_zul`. The file names the dataset, input and label columns, label order, and F1 variant (`span`, `token`, or `sequence`).

`label_names` must follow the source dataset's `ClassLabel` order because that order defines the integer label IDs. [`validate_label_names`](../src/data/preprocess_eval.py) checks the order during preprocessing and raises an error for a mismatch.

## `finetuning/`

Each file contains the fine-tuning hyperparameters for one task. The current values are shared across tasks, but separate files make task-specific changes easy later.

## Usage

Pass a config path on the command line to select a model or task:

```bash
uv run python src/pretraining/pretrain.py --model-config configs/models/roberta.yaml ...

uv run python src/finetuning/finetune.py \
    --task-config configs/evaluation/ner.yaml \
    --finetune-config configs/finetuning/ner.yaml \
    ...
```

`TaskConfig` and `FinetuneConfig` validate their fields in `__post_init__`. Invalid values such as a negative learning rate or empty `label_names` fail when the config is loaded.
