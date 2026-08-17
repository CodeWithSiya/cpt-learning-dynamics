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

import style

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Figure styling
FIGURE_SIZE = style.figure_size(style.COLUMN_WIDTH)

# Evaluation metric for each task
TASK_METRICS = {
    "ner": "f1",
    "pos": "f1",
    "ntc_xho": "f1",
    "ntc_zul": "f1"
}

# Display name for each task, used in plot legends
TASK_DISPLAY_NAMES = {
    "ner": "NER",
    "pos": "POS",
    "ntc_xho": "NTC",
    "ntc_zul": "NTC"
}

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

def configure_step_axis(ax, steps: list[int]) -> None:
    """
    Configure the checkpoint step axis, shared by every learning dynamics plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    """
    ax.set_xlim(min(steps), max(steps))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{int(x)}")
    )

def configure_value_axis(ax) -> None:
    """
    Scale the F1 axis from zero and end it a tick clear of the data.

    :param ax: Axes to configure.
    """
    locator = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(locator)

    high = ax.dataLim.intervaly[1]
    ticks = locator.tick_values(0.0, high)
    step = ticks[1] - ticks[0]

    # Keep a tick of headroom so the topmost label never crowds the title
    top = min(t for t in ticks if t >= high)
    if top - high < 0.25 * step:
        top += step

    # F1 cannot exceed one, so never leave headroom past it
    ax.set_ylim(0.0, min(top, 1.0))

def configure_axes(ax, steps: list[int]) -> None:
    """
    Apply the shared labels, scales and grid styling to a learning dynamics plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    """
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("Macro F1")
    configure_step_axis(ax, steps)
    configure_value_axis(ax)
    ax.grid(True)

def save_figure(fig, output_path: Path, description: str) -> None:
    """
    Write a figure to disk and close it.

    :param fig: Figure to save.
    :param output_path: File path to save the plot image to.
    :param description: Short description of the plot, used in the log message.
    """
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved {description} plot to {output_path}")

def plot_per_class_dynamics(aggregated: dict[int, dict], task: str, output_path: Path) -> None:
    """
    Plot per-class F1 for a single task across continued pretraining checkpoints.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param task: Task name, used in the log message.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    steps = sorted(aggregated.keys())

    # Collect all classes across checkpoints
    class_names = set()
    for step_data in aggregated.values():
        class_names.update(step_data["per_class_mean"].keys())

    # Plot the mean F1 for each class
    for class_name in sorted(class_names):
        means = metric_series([aggregated[s]["per_class_mean"].get(class_name) for s in steps])

        ax.plot(steps, means, label=class_name)

    # Add labels and formatting
    configure_axes(ax, steps)

    style.legend_below(fig, ax, ncol=min(len(class_names), 4))

    save_figure(fig, output_path, f"{task} per-class")

def plot_model_dynamics(aggregated_by_task: dict[str, dict[int, dict]], output_path: Path) -> None:
    """
    Plot the overall F1 for all tasks on a single figure.

    :param aggregated_by_task: Mapping from task name to aggregated results.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    all_steps = []

    # Plot the mean F1 for each task
    for task, aggregated in aggregated_by_task.items():
        steps = sorted(aggregated.keys())
        all_steps.extend(steps)

        means = metric_series([aggregated[s]["overall_mean"] for s in steps])

        ax.plot(steps, means, label=TASK_DISPLAY_NAMES[task])

    # Add labels and formatting
    configure_axes(ax, sorted(set(all_steps)))

    style.legend_below(fig, ax, ncol=len(aggregated_by_task))

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
            output_path=output_dir / f"{filename}_{task}_per_class.pdf"
        )

    # Plot overall learning dynamics for all tasks on a single figure
    plot_model_dynamics(
        aggregated_by_task,
        output_path=output_dir / f"{filename}_overall.pdf"
    )

if __name__ == "__main__":
    main()
