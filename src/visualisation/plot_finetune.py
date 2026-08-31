"""
Plot fine-tuning training and validation loss curves across all CPT
checkpoints, for a given task and seed, as a grid of subplots.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

import style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GRID_COLUMNS = 5

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
        "--model-name",
        type=str,
        required=True,
        help="Model name, used in the plot title and output filename."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the plot image to."
    )
    return parser.parse_args()

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a checkpoint directory name."""
    return int(path.name.split("-")[1])

def load_checkpoint_results(results_dir: Path, task: str, seed: int):
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

def plot_finetune_grid(results_by_step: dict[int, list[dict]], task: str, model_name: str, output_path: Path) -> None:
    """
    Plot a grid of fine-tuning loss curves, one subplot per checkpoint.

    :param results_by_step: Mapping from checkpoint step to fine-tuning log history.
    :param task: Task name, used in the plot title.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    steps = sorted(results_by_step.keys())
    num_checkpoints = len(steps)

    num_cols = GRID_COLUMNS
    num_rows = (num_checkpoints + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)

    for index, step in enumerate(steps):
        row, col = divmod(index, num_cols)
        ax = axes[row][col]

        log_history = results_by_step[step]

        train_points = [(entry["epoch"], entry["loss"]) for entry in log_history if "loss" in entry]
        eval_points = [(entry["epoch"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]

        if train_points:
            train_epochs, train_values = zip(*train_points)
            ax.plot(train_epochs, train_values, color=style.PALETTE[0], label="Train")

        if eval_points:
            eval_epochs, eval_values = zip(*eval_points)
            ax.plot(eval_epochs, eval_values, color=style.PALETTE[1], label="Val")

        ax.set_title(f"step-{step}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(num_checkpoints, num_rows * num_cols):
        row, col = divmod(idx, num_cols)
        axes[row][col].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.supxlabel("Epoch")
    fig.supylabel("Loss")
    fig.suptitle(f"Fine-tuning Loss per Checkpoint: {model_name} ({task.upper()})", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

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

    filename = args.model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = output_dir / f"{filename}_{args.task}_seed{args.seed}_finetune_grid.png"

    plot_finetune_grid(results_by_step, args.task, args.model_name, output_path)

if __name__ == "__main__":
    main()
