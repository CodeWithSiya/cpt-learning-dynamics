"""
Plot fine-tuning training and validation loss curves across all CPT
checkpoints, for a given task and seed, as a grid of subplots.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import style

from src.utils.extract import checkpoint_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GRID_COLUMNS = 5

EPOCH_LABEL = "Epoch"
LOSS_LABEL = "Loss"

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot fine-tuning loss curves across all checkpoints for a task/seed."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing finetuning results."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to plot."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to plot results for."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(style.MODEL_DISPLAY_NAMES),
        help="Model to plot, used in the figure title and output filename."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the figure to."
    )
    return parser.parse_args()

def load_checkpoint_results(results_dir: Path, task: str, seed: int) -> dict[int, list[dict]]:
    """
    Load fine-tuning results for a given task and seed, across all checkpoints.

    :param results_dir: Root directory containing finetuning results.
    :param task: Task name.
    :param seed: Seed to load results for.
    :return: Mapping from checkpoint step to that checkpoint's fine-tuning log history.
    """
    results_by_step = {}

    for step_dir in sorted(results_dir.glob("step-*"), key=checkpoint_step):
        results_path = step_dir / task / f"seed-{seed}" / "results.json"

        if not results_path.exists():
            logger.warning(f"No results found for {step_dir.name}/{task}/seed-{seed}, skipping.")
            continue

        with open(results_path) as f:
            results = json.load(f)
            results_by_step[checkpoint_step(step_dir)] = results["log_history"]

    logger.info(f"Loaded fine-tuning history for {len(results_by_step)} checkpoints.")
    return results_by_step

def plot_finetune_grid(results_by_step: dict[int, list[dict]], output_path: Path) -> None:
    """
    Plot a grid of fine-tuning loss curves, one panel per checkpoint.

    :param results_by_step: Mapping from checkpoint step to fine-tuning log history.
    :param output_path: File path to save the figure to.
    """
    steps = sorted(results_by_step.keys())
    num_rows = (len(steps) + GRID_COLUMNS - 1) // GRID_COLUMNS

    fig, axes = style.grid_figure(num_rows, GRID_COLUMNS, sharex=True, sharey=False)

    for index, step in enumerate(steps):
        row, col = divmod(index, GRID_COLUMNS)
        ax = axes[row][col]

        log_history = results_by_step[step]

        train_points = [(entry["epoch"], entry["loss"]) for entry in log_history if "loss" in entry]
        eval_points = [(entry["epoch"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]

        if train_points:
            train_epochs, train_values = zip(*train_points)
            ax.plot(train_epochs, train_values, label="Train", color=style.PALETTE[0])

        if eval_points:
            eval_epochs, eval_values = zip(*eval_points)
            ax.plot(eval_epochs, eval_values, label="Validation", color=style.PALETTE[4])

        ax.set_title(f"step-{step:,}")

    # Blank out the unused panels in the final row
    for index in range(len(steps), num_rows * GRID_COLUMNS):
        row, col = divmod(index, GRID_COLUMNS)
        axes[row][col].axis("off")

    style.configure_value_axis(axes.flat)

    style.label_grid(
        axes,
        xlabel=EPOCH_LABEL,
        ylabels=LOSS_LABEL,
        fig=fig,
    )

    style.finalise_grid(fig, axes, output_path)
    logger.info(f"Saved fine-tuning grid plot to {output_path}")

def main() -> None:
    """Main entry point for plotting fine-tuning loss curves."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_step = load_checkpoint_results(results_dir, args.task, args.seed)
    if not results_by_step:
        logger.error(f"No results found for task '{args.task}', seed {args.seed} in {results_dir}")
        return

    filename = args.model.replace("-", "_")
    output_path = output_dir / f"{filename}_{args.task}_seed{args.seed}_finetune_grid.pdf"

    plot_finetune_grid(results_by_step, output_path)

if __name__ == "__main__":
    main()
