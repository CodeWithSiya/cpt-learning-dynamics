"""
Plot CPT training and validation loss curves across models.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

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

# Display name for each model, used in plot titles and legends
MODEL_DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "xlmr": "XLMR",
    "nguni-xlmr": "Nguni-XLMR",
    "afriberta": "AfriBERTa"
}

# Display name for each supported language, used in plot titles
LANGUAGE_DISPLAY_NAMES = {
    "xho": "isiXhosa",
    "zul": "isiZulu"
}

# Axis label and plot title for every loss the trainer logs
LOSS_LABELS = {
    "loss": ("Training Loss", "Continued Pretraining Training Loss"),
    "eval_loss": ("Validation Loss", "Continued Pretraining Validation Loss")
}

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot CPT training and validation loss curves across models."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root results directory containing the per-model result folders."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODEL_DISPLAY_NAMES),
        choices=list(MODEL_DISPLAY_NAMES),
        help="Models to plot. Defaults to every model."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        choices=list(LANGUAGE_DISPLAY_NAMES),
        help="Language subset the models were continually pretrained on."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the combined all-models plot to."
    )
    return parser.parse_args()

def load_log_history(path: Path) -> list[dict]:
    """
    Load the CPT log history.

    :param path: Path to the CPT log history JSON file.
    :return: List of log entries.
    """
    with open(path) as f:
        log_history = json.load(f)

    logger.info(f"Loaded {len(log_history)} log entries from {path.name}.")
    return log_history

def loss_series(log_history: list[dict], loss: str) -> tuple[list[int], list[float]]:
    """
    Extract a loss curve from a log history.

    :param log_history: Full log history for one model.
    :param loss: Which loss field to read (e.g. "eval_loss").
    :return: Checkpoint steps and their loss values.
    """
    points = [(entry["step"], entry[loss]) for entry in log_history if loss in entry]

    if not points:
        return [], []

    steps, values = zip(*sorted(points))
    return list(steps), list(values)

def configure_step_axis(ax, steps: list[int]) -> None:
    """
    Configure the checkpoint step axis.

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
    Scale the loss axis to the data and end it exactly on a tick.

    :param ax: Axes to configure.
    """
    locator = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(locator)

    low, high = ax.dataLim.intervaly
    ticks = locator.tick_values(low, high)
    ax.set_ylim(max(t for t in ticks if t <= low), min(t for t in ticks if t >= high))

def configure_axes(ax, steps: list[int], ylabel: str, title: str) -> None:
    """
    Apply labels, scales and grid styling to a loss plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    :param ylabel: Y-axis label.
    :param title: Plot title.
    """
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

def plot_model_loss(log_history_by_model: dict[str, list[dict]], loss: str, language: str,
                    output_path: Path) -> None:
    """
    Plot one loss across CPT steps for all models on a single figure.

    :param log_history_by_model: Mapping from model name to log history.
    :param loss: Which loss field to plot.
    :param language: Language the models were continually pretrained on.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ylabel, title = LOSS_LABELS[loss]
    title = f"{title} ({LANGUAGE_DISPLAY_NAMES[language]})"
    all_steps = []

    # Plot the loss for each model
    for model, log_history in log_history_by_model.items():
        steps, values = loss_series(log_history, loss)

        if not steps:
            logger.warning(f"No '{loss}' entries for '{model}', skipping.")
            continue

        all_steps.extend(steps)
        ax.plot(steps, values, label=MODEL_DISPLAY_NAMES[model])

    if not all_steps:
        logger.error(f"No '{loss}' entries found for any model, skipping {output_path.name}.")
        plt.close(fig)
        return

    # Add labels and formatting
    configure_axes(ax, sorted(set(all_steps)), ylabel, title)

    style.legend_below(fig, ax, ncol=len(log_history_by_model))

    save_figure(fig, output_path, f"all-models {loss}")

def main() -> None:
    """Main entry point for plotting CPT loss curves."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the log history for every model that has been continually pretrained
    log_history_by_model = {}

    for model in args.models:
        log_history_path = results_dir / f"{model}-large" / args.language / "log_history.json"

        if not log_history_path.exists():
            logger.warning(f"No log history for '{model}' in {results_dir}, skipping.")
            continue

        log_history = load_log_history(log_history_path)
        if log_history:
            log_history_by_model[model] = log_history
        else:
            logger.warning(f"Log history for '{model}' is empty, skipping.")

    if not log_history_by_model:
        logger.error(f"No log histories found in {results_dir}")
        return

    # Plot every loss for all models, one figure per loss
    for loss in LOSS_LABELS:
        plot_model_loss(
            log_history_by_model,
            loss,
            args.language,
            output_path=output_dir / f"all_models_{args.language}_{loss}.pdf"
        )

if __name__ == "__main__":
    main()
