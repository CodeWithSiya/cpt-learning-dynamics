"""
Plot representation geometry as panel grids spanning every model, language and metric.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

import style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["xho", "zul"]

LANGUAGE_SUBSETS = {
    "xho": "xho_Latn",
    "zul": "zul_Latn",
}

METRIC_LABELS = {
    "matched_cosine_similarity": "Cosine similarity",
    "baseline_cosine_similarity": "Cosine similarity",
    "cosine_gap": "Cosine gap",
    "p_at_1_english_to_target": "P@1",
    "p_at_1_target_to_english": "P@1",
    "iso_score_shared": "IsoScore",
}

DEFAULT_METRICS = ["baseline_cosine_similarity", "iso_score_shared"]

LAYER_LABEL = "Layer"
STEP_LABEL = "CPT Step"

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot representation geometry as panel grids."
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
        help="Models to plot, in column order. Defaults to every model."
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=SUPPORTED_LANGUAGES,
        choices=SUPPORTED_LANGUAGES,
        help="Languages to plot. Defaults to every language."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS,
        choices=list(METRIC_LABELS),
        help="Metrics to plot, in row order. Defaults to cosine similarity and IsoScore."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the figures to."
    )
    return parser.parse_args()

def load_results(results_dir: Path, model: str, language: str, kind: str) -> dict[int, dict]:
    """
    Load one model's geometry results for a language, keyed by checkpoint step.

    :param results_dir: Root results directory.
    :param model: Model name, e.g. "xlmr".
    :param language: Language code, e.g. "zul".
    :param kind: Which analysis to load, either "alignment" or "layerwise".
    :return: Mapping from checkpoint step to results, empty when missing.
    """
    subset = LANGUAGE_SUBSETS[language]
    path = results_dir / f"{model}-large" / language / "alignment" / f"{subset}_{kind}.json"

    if not path.exists():
        logger.warning(f"No {kind} results at {path}, leaving that panel empty.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    return {int(step): data for step, data in raw.items()}

def load_all(results_dir: Path, models: list[str], languages: list[str],
             kind: str) -> dict[tuple[str, str], dict[int, dict]]:
    """
    Load every result the requested grid needs.

    :param results_dir: Root results directory.
    :param models: Models to load.
    :param languages: Languages to load.
    :param kind: Which analysis to load, either "alignment" or "layerwise".
    :return: Mapping from (model, language) to results by checkpoint step.
    """
    loaded = {}

    for model in models:
        for language in languages:
            results = load_results(results_dir, model, language, kind)
            if results:
                loaded[(model, language)] = results

    logger.info(f"Loaded {kind} results for {len(loaded)} model/language cells.")
    return loaded

def layer_series(step_results: dict, metric: str) -> np.ndarray:
    """
    Read one metric's per-layer profile as a float array.

    :param step_results: Layer-wise results for a single checkpoint.
    :param metric: Which metric field to read.
    :return: Float array with one value per layer.
    """
    values = step_results.get(metric, [])
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)

def step_series(results: dict[int, dict], steps: list[int], metric: str) -> np.ndarray:
    """
    Read one metric across checkpoint steps as a float array.

    :param results: Mapping from checkpoint step to alignment results.
    :param steps: Checkpoint steps to read, in plotting order.
    :param metric: Which metric field to read.
    :return: Float array of the same length as steps.
    """
    values = [results[step].get(metric) for step in steps]
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)

def step_label(step: int) -> str:
    """
    Format a checkpoint step for a legend entry.

    :param step: Checkpoint training step.
    :return: Compact label, e.g. "4k".
    """
    return f"{step / 1000:g}k" if step >= 1000 else f"{step}"

def plot_layerwise(layerwise, models, languages, metrics, output_path: Path) -> None:
    """
    Plot layer-wise profiles with one row per metric and language pairing and one
    column per model, drawing a line for each continued pretraining checkpoint.

    :param layerwise: Mapping from (model, language) to layer-wise results.
    :param models: Models to draw as columns.
    :param languages: Languages nested inside each metric down the rows.
    :param metrics: Metrics to draw as row groups.
    :param output_path: File path to save the figure to.
    """
    rows = [(metric, language) for language in languages for metric in metrics]

    all_steps = sorted({step for results in layerwise.values() for step in results})
    colours = dict(zip(all_steps, style.categorical_colours(len(all_steps))))

    fig, axes = style.grid_figure(
        len(rows), len(models), row_labels=True, sharex="col", sharey="row"
    )

    for row, (metric, language) in enumerate(rows):
        for col, model in enumerate(models):
            ax = axes[row][col]

            results = layerwise.get((model, language))
            if not results:
                continue

            for step in sorted(results):
                values = layer_series(results[step], metric)
                if values.size == 0:
                    continue

                ax.plot(
                    np.arange(values.size),
                    values,
                    label=step_label(step),
                    color=colours[step],
                )

            ax.set_xlim(0, max(ax.get_xlim()[1], 1))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))

    for metric in metrics:
        style.configure_value_axis([
            ax
            for row, (row_metric, _) in enumerate(rows) if row_metric == metric
            for ax in axes[row]
        ])

    style.label_grid(
        axes,
        column_titles=[style.MODEL_DISPLAY_NAMES[model] for model in models],
        xlabel=LAYER_LABEL,
        ylabels=[METRIC_LABELS[metric] for metric, _ in rows],
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for _, language in rows],
        max_legend_cols=len(all_steps),
    )
    logger.info(f"Saved layer-wise geometry grid to {output_path}")

def plot_checkpoints(alignment, models, languages, metrics, output_path: Path) -> None:
    """
    Plot geometry against continued pretraining step, with one row per metric and
    one column per language, drawing a line for each model.

    :param alignment: Mapping from (model, language) to alignment results.
    :param models: Models to draw as lines.
    :param languages: Languages to draw as columns.
    :param metrics: Metrics to draw as rows.
    :param output_path: File path to save the figure to.
    """
    fig, axes = style.grid_figure(len(metrics), len(languages))

    for row, metric in enumerate(metrics):
        for col, language in enumerate(languages):
            ax = axes[row][col]

            for model in models:
                results = alignment.get((model, language))
                if not results:
                    continue

                steps = sorted(results)
                ax.plot(
                    steps,
                    step_series(results, steps, metric),
                    label=style.MODEL_DISPLAY_NAMES[model],
                    color=style.MODEL_COLOURS[model],
                )
                style.configure_step_axis(ax, steps)

        style.configure_value_axis(axes[row])

    style.label_grid(
        axes,
        column_titles=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
        xlabel=STEP_LABEL,
        ylabels=[METRIC_LABELS[metric] for metric in metrics],
        fig=fig,
    )

    style.finalise_grid(fig, axes, output_path)
    logger.info(f"Saved checkpoint-wise geometry grid to {output_path}")

def main() -> None:
    """Main entry point for plotting representation geometry."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layerwise = load_all(results_dir, args.models, args.languages, "layerwise")
    alignment = load_all(results_dir, args.models, args.languages, "alignment")

    if not layerwise and not alignment:
        logger.error(f"No geometry results found under {results_dir}")
        return

    if layerwise:
        plot_layerwise(
            layerwise, args.models, args.languages, args.metrics,
            output_dir / "geometry_layerwise.pdf"
        )

    if alignment:
        plot_checkpoints(
            alignment, args.models, args.languages, args.metrics,
            output_dir / "geometry_checkpoints.pdf"
        )

if __name__ == "__main__":
    main()
