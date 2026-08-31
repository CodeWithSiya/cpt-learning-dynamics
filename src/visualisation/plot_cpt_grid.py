"""
Plot CPT training and validation loss as a panel grid spanning languages and loss types.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["xho", "zul"]

LOSS_TYPES = ["loss", "eval_loss"]

LOSS_LABELS = {
    "loss": "Training Loss",
    "eval_loss": "Validation Loss"
}

STEP_LABEL = "CPT Step"

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot CPT loss curves as a panel grid."
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
        default=list(style.MODEL_DISPLAY_NAMES),
        choices=list(style.MODEL_DISPLAY_NAMES),
        help="Models to plot, in legend order. Defaults to every model."
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=SUPPORTED_LANGUAGES,
        choices=SUPPORTED_LANGUAGES,
        help="Languages to plot, in row order. Defaults to every language."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the figure to."
    )
    return parser.parse_args()

def load_log_history(path: Path) -> list[dict]:
    """
    Load the CPT log history.

    :param path: Path to the CPT log history JSON file.
    :return: List of log entries.
    """
    if not path.exists():
        return []

    with open(path) as f:
        log_history = json.load(f)

    logger.info(f"Loaded {len(log_history)} log entries from {path.name}.")
    return log_history

def loss_series(log_history: list[dict], loss_type: str) -> tuple[list[int], list[float]]:
    """
    Extract a loss curve from a log history.

    :param log_history: Full log history for one model.
    :param loss_type: Which loss field to read (e.g. "eval_loss").
    :return: Checkpoint steps and their loss values.
    """
    points = [(entry["step"], entry[loss_type]) for entry in log_history if loss_type in entry]

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
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{int(x)}")
    )

def plot_grid(log_histories, models, languages, output_path: Path) -> None:
    """
    Plot CPT loss with rows for languages and columns for loss types, one line per model.

    :param log_histories: Mapping from (model, language) to log history.
    :param models: Models to draw as lines.
    :param languages: Languages to draw as rows.
    :param output_path: File path to save the figure to.
    """
    fig, axes = style.grid_figure(
        len(languages), len(LOSS_TYPES), row_labels=True, sharey="col"
    )

    for row, language in enumerate(languages):
        for col, loss_type in enumerate(LOSS_TYPES):
            ax = axes[row][col]

            all_steps = []

            for model in models:
                log_history = log_histories.get((model, language))
                if not log_history:
                    continue

                steps, values = loss_series(log_history, loss_type)
                if not steps:
                    continue

                all_steps.extend(steps)
                ax.plot(
                    steps,
                    values,
                    label=style.MODEL_DISPLAY_NAMES[model],
                    color=style.MODEL_COLOURS[model],
                )

            if all_steps:
                configure_step_axis(ax, sorted(set(all_steps)))

    for col in range(len(LOSS_TYPES)):
        style.configure_value_axis(axes[:, col], cap_at_one=False)

    style.label_grid(
        axes,
        column_titles=[LOSS_LABELS[loss_type] for loss_type in LOSS_TYPES],
        xlabel=STEP_LABEL,
        ylabels="Loss",
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
    )
    logger.info(f"Saved CPT loss grid to {output_path}")

def main() -> None:
    """Main entry point for plotting CPT loss grid."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_histories = {}

    for model in args.models:
        for language in args.languages:
            log_history_path = results_dir / f"{model}-large" / language / "log_history.json"
            log_history = load_log_history(log_history_path)

            if log_history:
                log_histories[(model, language)] = log_history

    if not log_histories:
        logger.error(f"No log histories found in {results_dir}")
        return

    plot_grid(
        log_histories, args.models, args.languages,
        output_dir / "cpt_grid.pdf"
    )

if __name__ == "__main__":
    main()
