# Pipeline scripts

These [SLURM](https://slurm.schedmd.com/overview.html) scripts run the pipeline on an HPC cluster. Each script covers one stage and loops over the configured models, languages, tasks, or seeds. Pass positional arguments when you want to run a smaller part of the sweep, such as `sbatch scripts/pretrain.sh roberta xho`.

Before submitting jobs, update the cluster-specific settings at the top of each script: repository and `.env` paths, email address, account, partition, and resource requests. The scripts place Hugging Face caches, datasets, checkpoints, logs, and results under `$SCRATCH`; make sure it points to a suitable scratch filesystem and copy important outputs elsewhere before the scratch space is cleaned.

When a job starts, the script loads `.env`, creates the scratch directories, loads Python 3.12, and runs `uv sync --frozen`.

## Pipeline order

Run the stages in this order:

1. `download.sh` downloads the training, evaluation, and FLORES-200 data.
2. `preprocess_wura.sh` prepares the continued-pretraining corpus.
3. `preprocess_eval.sh` prepares the downstream evaluation datasets. This can run alongside `preprocess_wura.sh`.
4. `pretrain.sh` trains each model and saves its checkpoints. It requires the processed WURA corpus.
5. `alignment.sh`, `layerwise.sh`, and `finetune.sh` analyse or evaluate the checkpoints. These stages can run independently after `pretrain.sh`.
6. `aggregate.sh` combines the fine-tuning results across seeds. It requires `finetune.sh`.
7. `plot_cpt_grid.sh` plots CPT loss and can run after `pretrain.sh`.
8. `plot_geometry.sh` plots alignment and isotropy from `layerwise.sh`.
9. `plot_downstream.sh` plots downstream results and performs aggregation as part of the plotting step.
10. `plot_finetune.sh` plots per-checkpoint fine-tuning loss for debugging.

| Script               | Loops over                          | Produces                                                  |
| -------------------- | ----------------------------------- | --------------------------------------------------------- |
| `download.sh`        | model / language                    | Raw WURA corpus, evaluation datasets, FLORES-200          |
| `preprocess_wura.sh` | model × language                    | Tokenised, chunked CPT corpus                             |
| `preprocess_eval.sh` | model × language × task             | Tokenised, label-aligned evaluation datasets              |
| `pretrain.sh`        | model × language                    | CPT checkpoints on the two-phase schedule                 |
| `alignment.sh`       | model × language                    | Cross-lingual alignment, every checkpoint                 |
| `layerwise.sh`       | model × language                    | Layer-wise alignment, a chosen checkpoint subset          |
| `finetune.sh`        | model × language × task × seed      | Fine-tuning results per checkpoint, per seed              |
| `aggregate.sh`       | model × language × task             | Fine-tuning results aggregated across seeds               |
| `plot_cpt_grid.sh`   | (single grid, all models/languages) | CPT loss grid                                             |
| `plot_geometry.sh`   | (single grid, all models/languages) | Alignment and isotropy grid, from `layerwise.sh`'s output |
| `plot_downstream.sh` | (aggregates, then a single grid)    | Overall and per-class F1 grids                            |
| `plot_finetune.sh`   | model × language × task × seed      | Per-checkpoint fine-tuning loss grid (debug only)         |

`plot_downstream.sh` runs the aggregation loop before plotting, so it can be submitted without a separate `aggregate.sh` job.

`alignment.sh` writes alignment for every checkpoint. The current geometry plot reads the per-layer output from `layerwise.sh`, so use `alignment.sh` when the checkpoint-level values are needed for a table or a new figure.

## Usage

Submit a stage with `sbatch`:

```bash
sbatch scripts/pretrain.sh
```

Most scripts accept optional positional arguments for a single model and language:

```bash
sbatch scripts/pretrain.sh roberta xho
sbatch scripts/finetune.sh roberta xho 42      # finetune.sh and plot_finetune.sh also take a seed
```

Omit an argument to keep the full loop at that position. For example, `sbatch scripts/pretrain.sh roberta` runs `roberta` on both languages. The grid scripts (`plot_cpt_grid.sh`, `plot_downstream.sh`, and `plot_geometry.sh`) pass command-line options to the Python plotting modules:

```bash
sbatch scripts/plot_geometry.sh --models roberta xlmr --languages xho
```

To test Python logic without a SLURM allocation, run the underlying module directly as described in [`src/README.md`](../src/README.md). Keep full training and large evaluation jobs on a compute node.

## Task and language coverage

isiXhosa (`xho`) and isiZulu (`zul`) use NER and POS. News topic classification uses `ntc_xho` for isiXhosa and `ntc_zul` for isiZulu because the available datasets differ. Each language uses `roberta`, `xlmr`, `nguni-xlmr`, and `afriberta`. Fine-tuning uses five seeds (`42`, `123`, `456`, `789`, and `1738`) for the mean and standard deviation calculated by `aggregate.sh`.
