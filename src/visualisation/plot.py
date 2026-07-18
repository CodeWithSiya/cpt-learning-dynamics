"""
Plot learning dynamics curves from fine-tuning results across CPT checkpoints.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Evaluation metric for each task
TASK_METRICS = {
    "ner": "f1",
    "pos": "accuracy",
    "ntc": "f1"
}

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot learning dynamics curves from fine-tuning results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing step-N/<task>/results.json subfolders."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_METRICS.keys()),
        help="Task to plot (ner, pos, ntc)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save plot images to."
    )
    return parser.parse_args()

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a checkpoint directory name."""
    return int(path.name.split("-")[1])

def load_results_for_task(results_dir: Path, task: str) -> dict[int, dict]:
    """
    Load fine-tuning results for a given task across all available checkpoints.

    :param results_dir: Root directory containing evaluation results.
    :param task: Name of the evaluation task.
    :return: Mapping from checkpoint step to the contents of that checkpoint's results file.
    """
    results_by_step = {}

    # Iterate through checkpoints in step order
    for step_dir in sorted(results_dir.glob("step-*"), key=checkpoint_step):
        step = checkpoint_step(step_dir)
        results_path = step_dir / task / "results.json"

        # Skip checkpoints that do not contain results for this task
        if not results_path.exists():
            logger.warning(f"No results found for {step_dir.name}/{task}, skipping.")
            continue

        # Load the evaluation results for this checkpoint
        with open(results_path) as f:
            results_by_step[step] = json.load(f)

    logger.info(f"Loaded {len(results_by_step)} checkpoints for task '{task}'")
    return results_by_step

def extract_per_class_series(results_by_step: dict[int, dict]) -> dict[str, list[tuple[int, float]]]:
    """
    Extract per-class F1 scores across checkpoints.

    :param results_by_step: Mapping from checkpoint step to the contents of that checkpoint's results file.
    :return: Mapping from class name to a list of (step, F1 score) pairs.
    """
    series = {}

    # Extract each class's F1 scores across checkpoints
    for step, results in sorted(results_by_step.items()):
        step_per_class_f1 = results["test_metrics"].get("eval_per_class_f1", {})
        for class_name, value in step_per_class_f1.items():
            series.setdefault(class_name, []).append((step, value))

    return series

def extract_overall_series(results_by_step: dict[int, dict], metric_key: str) -> list[tuple[int, float]]:
    """
    Extract the overall metric across checkpoints.

    :param results_by_step: Mapping from checkpoint step to the contents of that checkpoint's results file.
    :param metric_key: Key of the overall metric in test_metrics (e.g. "f1", "accuracy").
    :return: List of (step, value) pairs, sorted by step.
    """
    series = []
    metric_name = f"eval_{metric_key}"

    # Extract each overall metric  cross checkpoints
    for step, results in sorted(results_by_step.items()):
        test_metrics = results["test_metrics"]

        # Skip checkpoints that do not contain the required metric
        if metric_name not in test_metrics:
            continue

        # Store the checkpoint step and metric value
        series.append((step, test_metrics[metric_name]))

    return series

def plot_per_class_dynamics(series: dict[str, list[tuple[int, float]]], task: str, output_path: Path) -> None:
    """
    Plot per-class F1 scores across continued pretraining checkpoints.

    :param series: Mapping from class name to a list of (step, F1 score) pairs.
    :param task: Task name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot one learning dynamics curve for each class
    for class_name, points in sorted(series.items()):
        steps, values = zip(*points)
        ax.plot(steps, values, marker="o", markersize=3, label=class_name)

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{task.upper()} Per-Class Learning Dynamics")
    ax.set_xscale("symlog")
    ax.legend(title="Class", loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Save the figure 
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved per-class learning dynamics plot to {output_path}")

def plot_overall_dynamics(series: list[tuple[int, float]], metric_name: str, task: str, output_path: Path):
    """
    Plot the overall metric vs. step.

    :param series: List of (step, value) pairs.
    :param metric_name: Name of the metric, used for the y-axis label.
    :param task: Task name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate the checkpoint steps and metric values
    steps, values = zip(*series)

    # Plot the metric across CPT checkpoints
    ax.plot(steps, values, marker="o", markersize=4, color="tab:blue")

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{task.upper()} Overall Learning Dynamics")
    ax.set_xscale("symlog")
    ax.grid(True, alpha=0.3)

    # Save the figure 
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved overall plot to {output_path}")

def main() -> None:
    """Main entry point for plotting learning dynamics."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_step = load_results_for_task(results_dir, args.task)
    if not results_by_step:
        logger.error(f"No results found for task '{args.task}' in {results_dir}")
        return

    # Per-class plot
    per_class_series = extract_per_class_series(results_by_step)
    if per_class_series:
        plot_per_class_dynamics(
            per_class_series,
            task=args.task,
            output_path=output_dir / f"{args.task}_per_class.png"
        )
    else:
        logger.warning(f"No per-class F1 data found for task '{args.task}'")

    # Overall plot
    metric_key = TASK_METRICS[args.task]
    overall_series = extract_overall_series(results_by_step, metric_key)
    if overall_series:
        plot_overall_dynamics(
            overall_series,
            metric_name=metric_key,
            task=args.task,
            output_path=output_dir / f"{args.task}_overall.png"
        )
    else:
        logger.warning(f"No overall metric data found for task '{args.task}'")

if __name__ == "__main__":
    main()