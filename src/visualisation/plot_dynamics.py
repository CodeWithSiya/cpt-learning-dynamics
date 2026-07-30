"""
Plot learning dynamics curves from aggregated fine-tuning results across CPT checkpoints.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Figure styling
FIGURE_SIZE = (10, 6)
LINE_WIDTH = 1.4

# Evaluation metric for each task
TASK_METRICS = {
    "ner": "f1",
    "pos": "f1",
    "ntc": "f1"
}

# Colour palette, one colour per task
PALETTE = ["lightcoral", "cornflowerblue", "lightgreen"]

# Width of the symlog linear region
SYMLOG_LINTHRESH = 1
SYMLOG_LINSCALE = 1.0

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot learning dynamics curves from aggregated fine-tuning results."
    )
    parser.add_argument(
        "--aggregated-dir",
        type=str,
        required=True,
        help="Directory containing the per-task aggregated results JSON files."
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
    logger.info(f"Loaded aggregated results for {len(aggregated)} checkpoints from {path.name}.")
    return aggregated

def metric_series(values: list) -> np.ndarray:
    """
    Convert a per-step metric series to a float array.

    :param values: Metric values, one per checkpoint step.
    :return: Float array of the same length.
    """
    return np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=float
    )

def clamp_for_log(steps: list[int]) -> list[int]:
    """
    Replace step 0 with 1 so it can be plotted on a log-style axis.

    :param steps: Checkpoint steps, possibly including 0.
    :return: Steps with 0 replaced by 1.
    """
    return [max(step, 1) for step in steps]

def format_power_of_ten(value: float, _pos: int) -> str:
    """
    Format a tick value as clean power-of-ten notation.

    :param value: Tick value to format.
    :param _pos: Tick position.
    :return: Formatted tick label.
    """
    exponent = int(round(np.log10(value)))
    return f"$10^{{{exponent}}}$"

def configure_step_axis(ax, steps: list[int]) -> None:
    """
    Configure the checkpoint step axis, shared by every learning dynamics plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted (already clamped, all >= 1).
    """
    max_step = max(steps)
    highest = int(np.floor(np.log10(max_step)))
    ticks = [float(10 ** exponent) for exponent in range(0, highest + 1)]

    ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=SYMLOG_LINSCALE)
    ax.set_xlim(min(steps), max_step)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_power_of_ten))

def configure_axes(ax, steps: list[int], title: str, scale: bool = True) -> None:
    """
    Apply the shared labels, scales and grid styling to a learning dynamics plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    :param scale: Whether to scale the y-axis or not.
    :param title: Plot title.
    """
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("F1")
    ax.set_title(title)
    configure_step_axis(ax, steps)
    if scale: ax.set_ylim(0, 1) 
    ax.grid(True, alpha=0.3, linestyle=":")

def save_figure(fig, output_path: Path, description: str) -> None:
    """
    Write a figure to disk and close it.

    :param fig: Figure to save.
    :param output_path: File path to save the plot image to.
    :param description: Short description of the plot, used in the log message.
    """
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved {description} plot to {output_path}")

def plot_per_class_dynamics(aggregated: dict[int, dict], task: str, model_name: str, output_path: Path) -> None:
    """
    Plot per-class F1 for a single task across continued pretraining checkpoints.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param task: Task name, used in the plot title.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    steps = sorted(aggregated.keys())
    plot_steps = clamp_for_log(steps)

    # Collect all classes across checkpoints
    class_names = set()
    for step_data in aggregated.values():
        class_names.update(step_data["per_class_mean"].keys())

    # Plot the mean F1 for each class
    for class_name in sorted(class_names):
        means = metric_series([aggregated[s]["per_class_mean"].get(class_name) for s in steps])

        ax.plot(plot_steps, means, label=class_name, linewidth=LINE_WIDTH)

    # Add labels and formatting
    configure_axes(ax, plot_steps, f"{task.upper()} Per-Class Learning Dynamics ({model_name})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

    save_figure(fig, output_path, f"{task} per-class")

def plot_model_dynamics(aggregated_by_task: dict[str, dict[int, dict]], model_name: str, output_path: Path) -> None:
    """
    Plot the overall F1 for all tasks on a single figure.

    :param aggregated_by_task: Mapping from task name to aggregated results.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    all_steps = []

    # Plot the mean F1 for each task
    for i, (task, aggregated) in enumerate(aggregated_by_task.items()):
        steps = sorted(aggregated.keys())
        all_steps.extend(steps)
        plot_steps = clamp_for_log(steps)

        means = metric_series([aggregated[s]["overall_mean"] for s in steps])
        color = PALETTE[i % len(PALETTE)]

        ax.plot(plot_steps, means, label=task.upper(), linewidth=LINE_WIDTH, color=color)

    # Add labels and formatting
    configure_axes(ax, clamp_for_log(sorted(set(all_steps))), f"Learning Dynamics ({model_name})", scale=False)
    ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, output_path, "all-tasks overview")

def main() -> None:
    """Main entry point for plotting learning dynamics."""
    args = parse_args()

    aggregated_dir = Path(args.aggregated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the aggregated results for every task that has been aggregated
    aggregated_by_task = {}

    for task in TASK_METRICS:
        aggregated_path = aggregated_dir / f"{task}_aggregated.json"

        if not aggregated_path.exists():
            logger.warning(f"No aggregated results for '{task}' in {aggregated_dir}, skipping.")
            continue

        aggregated = load_aggregated_results(aggregated_path)
        if aggregated:
            aggregated_by_task[task] = aggregated
        else:
            logger.warning(f"Aggregated results for '{task}' are empty, skipping.")

    if not aggregated_by_task:
        logger.error(f"No aggregated results found in {aggregated_dir}")
        return

    filename = args.model_name.lower().replace(" ", "_").replace("-", "_")

    # Plot per-class learning dynamics, one figure per task
    for task, aggregated in aggregated_by_task.items():
        plot_per_class_dynamics(
            aggregated,
            task=task,
            model_name=args.model_name,
            output_path=output_dir / f"{filename}_{task}_per_class.png"
        )

    # Plot overall learning dynamics for all tasks on a single figure
    plot_model_dynamics(
        aggregated_by_task,
        model_name=args.model_name,
        output_path=output_dir / f"{filename}_overall.png"
    )

if __name__ == "__main__":
    main()
    