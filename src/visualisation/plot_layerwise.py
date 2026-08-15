"""
Plot layer-wise cross-lingual alignment profiles for selected CPT checkpoints.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Figure styling
FIGURE_SIZE = (10, 6)
LINE_WIDTH = 1.4

# Display name for each model, used in plot titles and legends
MODEL_DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "xlmr": "XLMR",
    "nguni-xlmr": "Nguni-XLMR",
    "afriberta": "AfriBERTa"
}

# Colour palette, one colour per model
PALETTE = ["lightcoral", "cornflowerblue", "lightgreen", "sandybrown"]

# Axis label and title fragment for every metric layerwise.py reports
METRIC_LABELS = {
    "matched_cosine_similarity": ("Cosine Similarity", "Matched Pair Cosine Similarity"),
    "baseline_cosine_similarity": ("Cosine Similarity", "Non-Matched Baseline Cosine Similarity"),
    "cosine_gap": ("Cosine Gap", "Cross-Lingual Alignment (Cosine Gap)"),
    "p_at_1_english_to_target": ("P@1", "Top-1 Retrieval Accuracy (English to Target)"),
    "p_at_1_target_to_english": ("P@1", "Top-1 Retrieval Accuracy (Target to English)"),
    "iso_score_shared": ("IsoScore", "Isotropy of the Shared Bilingual Space")
}

# Supported languages
SUPPORTED_LANGUAGES = ["xho_Latn", "zul_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot layer-wise cross-lingual alignment profiles across CPT checkpoints."
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
        help="FLORES-200 language subset the layer-wise results were computed on."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(METRIC_LABELS),
        choices=list(METRIC_LABELS),
        help="Which layer-wise metrics to plot. Defaults to every metric."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the plots to."
    )
    return parser.parse_args()

def load_layerwise_results(path: Path) -> dict[int, dict]:
    """
    Load layer-wise alignment results, keyed by checkpoint step.

    :param path: Path to layer-wise results JSON.
    :return: Mapping from checkpoint step to results dict.
    """
    with open(path) as f:
        raw = json.load(f)

    # Load results and convert JSON keys to integers
    results = {int(step): data for step, data in raw.items()}
    logger.info(f"Loaded layer-wise results for {len(results)} checkpoints from {path.name}.")
    return results

def metric_series(layerwise_scores: dict, metric: str) -> np.ndarray:
    """
    Read one metric's per-layer values as a float array.

    :param layerwise_scores: Layer-wise results for a single checkpoint.
    :param metric: Which metric field to read.
    :return: Float array with one value per layer.
    """
    values = layerwise_scores.get(metric, [])
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)

def layer_positions(num_values: int, relative: bool) -> np.ndarray:
    """
    Build the x-axis positions for a layer profile.

    :param num_values: Number of layers in the profile, including the embedding layer.
    :param relative: Whether to return relative depth or absolute layer indices.
    :return: Float array of x positions.
    """
    layers = np.arange(num_values, dtype=float)

    if relative and num_values > 1:
        return layers / (num_values - 1)

    return layers

def configure_axes(ax, metric: str, title: str) -> None:
    """
    Apply labels, limits and grid styling to a layer-wise plot.

    :param ax: Axes to configure.
    :param metric: Metric being plotted, used for the y-axis label.
    :param title: Plot title.
    """
    ylabel, _ = METRIC_LABELS[metric]

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.grid(True, alpha=0.3, linestyle=":", which="both")

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

def plot_model_layerwise(model: str, layerwise_by_step: dict[int, dict], metric: str,
                         output_path: Path) -> None:
    """
    Plot the layer-wise profile for a single model, with one line per checkpoint.

    :param model: Model name, used in the plot title.
    :param layerwise_by_step: Mapping from checkpoint step to layer-wise results.
    :param metric: Which metric to plot.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    steps = sorted(layerwise_by_step.keys())

    for step in steps:
        values = metric_series(layerwise_by_step[step], metric)

        if values.size == 0:
            logger.warning(f"No '{metric}' values for '{model}' at step {step}, skipping.")
            continue

        ax.plot(
            layer_positions(values.size, relative=False),
            values,
            label=f"step {step:,}",
            linewidth=LINE_WIDTH,
        )

    _, title = METRIC_LABELS[metric]
    configure_axes(
        ax,
        metric,
        f"{title} by Layer ({MODEL_DISPLAY_NAMES[model]})",
    )
    ax.legend(loc="best", fontsize=9, title="Checkpoint")

    save_figure(fig, output_path, f"{model} layer-wise {metric}")

def main() -> None:
    """Main entry point for plotting layer-wise alignment profiles."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    language = args.language.split("_")[0]

    # Load the layer-wise results for every model that has been evaluated
    layerwise_by_model = {}

    for model in args.models:
        layerwise_path = results_dir / f"{model}-large" / language / "alignment" / f"{args.language}_layerwise.json"

        if not layerwise_path.exists():
            logger.warning(f"No layer-wise results for '{model}' in {results_dir}, skipping.")
            continue

        layerwise_by_step = load_layerwise_results(layerwise_path)
        if layerwise_by_step:
            layerwise_by_model[model] = layerwise_by_step
        else:
            logger.warning(f"Layer-wise results for '{model}' are empty, skipping.")

    if not layerwise_by_model:
        logger.error(f"No layer-wise results found in {results_dir}")
        return

    for metric in args.metrics:
        for model, layerwise_by_step in layerwise_by_model.items():
            plot_model_layerwise(
                model,
                layerwise_by_step,
                metric,
                output_path=output_dir / f"{model}_{args.language}_layerwise_{metric}.png"
            )

if __name__ == "__main__":
    main()
