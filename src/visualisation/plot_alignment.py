"""
Plot cross-lingual alignment (cosine gap) across CPT checkpoints.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FIGURE_SIZE = style.figure_size(style.COLUMN_WIDTH)

MODEL_DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "xlmr": "XLMR",
    "nguni-xlmr": "Nguni-XLMR",
    "afriberta": "AfriBERTa"
}

SUPPORTED_LANGUAGES = ["xho_Latn", "zul_Latn"]

METRIC_LABELS = {
    "matched_cosine_similarity": "Cosine Similarity",
    "baseline_cosine_similarity": "Cosine Similarity",
    "cosine_gap": "Cosine Gap",
    "p_at_1_english_to_target": "P@1",
    "p_at_1_target_to_english": "P@1",
    "iso_score_shared": "IsoScore"
}

DEFAULT_METRICS = ["baseline_cosine_similarity", "cosine_gap", "iso_score_shared"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot cross-lingual alignment (cosine gap) across CPT checkpoints."
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
        default="xho_Latn",
        choices=SUPPORTED_LANGUAGES,
        help="FLORES-200 language subset the alignment results were computed on."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS,
        choices=list(METRIC_LABELS),
        help="Which metrics to plot."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the combined all-models plot to."
    )
    return parser.parse_args()

def load_alignment_results(path: Path) -> dict[int, dict]:
    """
    Load alignment results, keyed by checkpoint step.

    :param path: Path to alignment results JSON.
    :return: Mapping from checkpoint step to results dict.
    """
    with open(path) as f:
        raw = json.load(f)

    results = {int(step): data for step, data in raw.items()}
    logger.info(f"Loaded alignment results for {len(results)} checkpoints from {path.name}.")
    return results

def metric_series(alignment_by_step: dict[int, dict], steps: list[int], metric: str) -> np.ndarray:
    """
    Convert a per-step series for one metric to a float array.

    :param alignment_by_step: Mapping from checkpoint step to alignment results.
    :param steps: Checkpoint steps to read, in plotting order.
    :param metric: Which metric field to read (e.g. "cosine_gap").
    :return: Float array of the same length, with missing values as NaN.
    """
    values = [alignment_by_step[step].get(metric) for step in steps]
    return np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=float
    )

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
    Scale the metric axis from zero and end it a tick clear of the data.

    :param ax: Axes to configure.
    """
    locator = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(locator)

    low, high = ax.dataLim.intervaly
    low = min(low, 0.0)

    ticks = locator.tick_values(low, high)
    step = ticks[1] - ticks[0]

    # Keep a tick of headroom, unless the metric is already capped at one
    bottom = max(t for t in ticks if t <= low)
    top = min(t for t in ticks if t >= high)
    if top - high < 0.25 * step:
        top = 1.0 if high <= 1.0 else top + step

    ax.set_ylim(bottom, top)

def configure_axes(ax, steps: list[int], ylabel: str) -> None:
    """
    Apply labels, scales and grid styling to an alignment plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    :param ylabel: Y-axis label.
    """
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel(ylabel)
    configure_step_axis(ax, steps)
    configure_value_axis(ax)

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

def plot_model_metric(alignment_by_model: dict[str, dict[int, dict]], metric: str,
                      output_path: Path) -> None:
    """
    Plot one metric across CPT steps for all models on a single figure.

    :param alignment_by_model: Mapping from model name to alignment results.
    :param metric: Which metric field to plot.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    all_steps = []

    for model, alignment_by_step in alignment_by_model.items():
        steps = sorted(alignment_by_step.keys())
        all_steps.extend(steps)

        values = metric_series(alignment_by_step, steps, metric)

        ax.plot(steps, values, label=MODEL_DISPLAY_NAMES[model])

    configure_axes(ax, sorted(set(all_steps)), METRIC_LABELS[metric])

    style.legend_below(fig, ax, ncol=min(len(alignment_by_model), 4))

    save_figure(fig, output_path, f"all-models {metric}")

def main() -> None:
    """Main entry point for plotting cross-lingual alignment dynamics."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    language = args.language.split("_")[0]

    alignment_by_model = {}

    for model in args.models:
        alignment_path = results_dir / f"{model}-large" / language / "alignment" / f"{args.language}_alignment.json"

        if not alignment_path.exists():
            logger.warning(f"No alignment results for '{model}' in {results_dir}, skipping.")
            continue

        alignment_by_step = load_alignment_results(alignment_path)
        if alignment_by_step:
            alignment_by_model[model] = alignment_by_step
        else:
            logger.warning(f"Alignment results for '{model}' are empty, skipping.")

    if not alignment_by_model:
        logger.error(f"No alignment results found in {results_dir}")
        return

    for metric in args.metrics:
        plot_model_metric(
            alignment_by_model,
            metric,
            output_path=output_dir / f"all_models_{args.language}_{metric}.pdf"
        )

if __name__ == "__main__":
    main()
