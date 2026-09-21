# Configuration

Each experiment uses a YAML configuration file, which is loaded and validated when the run starts.

| Directory     | Loaded into                                   | Drives                                 |
| ------------- | --------------------------------------------- | -------------------------------------- |
| `models/`     | `ModelConfig` (`src/pretraining/config.py`)   | Continued pretraining (CPT)            |
| `evaluation/` | `TaskConfig` (`src/finetuning/config.py`)     | Dataset preprocessing and task metrics |
| `finetuning/` | `FinetuneConfig` (`src/finetuning/config.py`) | Downstream fine-tuning hyperparameters |

## `models/`

One file per base model: `roberta`, `xlmr`, `nguni-xlmr`, and `afriberta`. Each specifies the Hugging Face checkpoint to continue pretraining from, the MLM optimisation settings, and the W&B run identity.

All four models train for 200,000 steps at an effective batch size of 80 (`per_device_batch_size: 10` × `gradient_accumulation_steps: 8`).

The `checkpoint_schedule` block configures the two-phase checkpointing schedule implemented in `src/pretraining/schedule.py`: dense checkpoints across the first 10% of training and sparse checkpoints thereafter. Unset fields fall back to the defaults in `CheckpointScheduleConfig`.

Checkpoint output paths are passed per run via `--output-dir`, so the same model config is reused across both languages.

## `evaluation/`

One file per downstream task: `ner`, `pos`, `ntc_xho`, and `ntc_zul`. Each describes the dataset for that task, which columns hold the inputs and labels, the full label set in dataset order, and which F1 variant is used for evaluation (`span`, `token`, or `sequence`).

`label_names` must match the order of the source dataset's `ClassLabel` feature, since that order defines the integer label IDs. `validate_label_names` in `src/data/preprocess_eval.py` checks this against the dataset at preprocessing time and raises an error if they do not match.

## `finetuning/`

One file per task, holding the fine-tuning hyperparameters. Values are identical across all four tasks, so differences in downstream performance can be attributed to the CPT checkpoint being evaluated. Each task keeps its own file so that its hyperparameters can be adjusted independently later.
