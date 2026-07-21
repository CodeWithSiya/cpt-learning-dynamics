"""
Plot learning dynamics curves from aggregated fine-tuning results across
CPT checkpoints, with shaded standard deviation bands across seeds.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
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
        description="Plot learning dynamics curves from aggregated fine-tuning results."
    )
    parser.add_argument(
        "--aggregated-results",
        type=str,
        required=True,
        help="Path to aggregated results JSON."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_METRICS.keys()),
        help="Task to plot."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save plot images to."
    )
    return parser.parse_args()

def load_aggregated_results(path: Path) -> dict[int, dict]:
    """
    Load aggregated results keyed by checkpoint step.

    :param path: Path to the aggregated results JSON file.
    :return: Mapping from checkpoint step to aggregated results.
    """
    with open(path) as f:
        raw = json.load(f)

    # Load aggregated results and convert JSON keys to integers
    aggregated = {int(step): data for step, data in raw.items()}
    logger.info(f"Loaded aggregated results for {len(aggregated)} checkpoints.")
    return aggregated

def plot_per_class_dynamics(aggregated: dict[int, dict], task: str, output_path: Path) -> None:
    """
    Plot per-class F1 scores across continued pretraining checkpoints.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param task: Task name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = sorted(aggregated.keys())
    plot_steps = [max(s, 1) for s in steps]

    # Collect all classes across checkpoints
    class_names = set()
    for step_data in aggregated.values():
        class_names.update(step_data["per_class_mean"].keys())

    # Plot the mean F1 and standard deviation for each class
    for class_name in sorted(class_names):
        means = np.array([aggregated[s]["per_class_mean"].get(class_name, np.nan) for s in steps])
        stds = np.array([aggregated[s]["per_class_std"].get(class_name, 0.0) for s in steps])

        line, = ax.plot(plot_steps, means, label=class_name, linewidth=1.3)
        ax.fill_between(plot_steps, means - stds, means + stds, alpha=0.15, color=line.get_color())

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("F1")
    ax.set_title(f"{task.upper()} Per-Class Learning Dynamics")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Save the figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved class-wise F1 plot to {output_path}")

def plot_overall_dynamics(aggregated: dict[int, dict], metric_name: str, task: str, output_path: Path) -> None:
    """
    Plot the overall metric vs. step.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param metric_name: Name of the metric, used for the y-axis label.
    :param task: Task name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = sorted(aggregated.keys())
    plot_steps = [max(s, 1) for s in steps]

    means = np.array([aggregated[s]["overall_mean"] for s in steps])
    stds = np.array([aggregated[s]["overall_std"] for s in steps])

    # Plot the mean metric and standard deviation
    ax.plot(plot_steps, means, color="tab:blue", linewidth=1.5)
    ax.fill_between(plot_steps, means - stds, means + stds, alpha=0.2, color="tab:blue")

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{task.upper()} Overall {metric_name.capitalize()}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Save the figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved overall plot to {output_path}")

def main() -> None:
    """Main entry point for plotting learning dynamics."""
    args = parse_args()

    aggregated_path = Path(args.aggregated_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = load_aggregated_results(aggregated_path)
    if not aggregated:
        logger.error(f"No aggregated results found in {aggregated_path}")
        return

    # Plot per-class learning dynamics
    plot_per_class_dynamics(
        aggregated,
        task=args.task,
        output_path=output_dir / f"{args.task}_per_class.png"
    )

    # Plot overall learning dynamics
    metric_key = TASK_METRICS[args.task]
    plot_overall_dynamics(
        aggregated,
        metric_name=metric_key,
        task=args.task,
        output_path=output_dir / f"{args.task}_overall.png"
    )

if __name__ == "__main__":
    main()