"""
Plot pseudo-perplexity across CPT checkpoints for all models.
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
    "nguni-xlmr": "Nguni-XLMR"
}

# Colour palette, one colour per model
PALETTE = ["lightcoral", "cornflowerblue", "lightgreen"]

# Supported languages
SUPPORTED_LANGUAGES = ["xho_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot pseudo-perplexity across CPT checkpoints."
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
        help="FLORES-200 language subset the perplexity results were computed on."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the combined all-models plot to."
    )
    return parser.parse_args()

def load_perplexity_results(path: Path) -> dict[int, dict]:
    """
    Load pseudo-perplexity results, keyed by checkpoint step.

    :param path: Path to perplexity results JSON.
    :return: Mapping from checkpoint step to results directory.
    """
    with open(path) as f:
        raw = json.load(f)

     # Load results and convert JSON keys to integers
    results = {int(step): data for step, data in raw.items()}
    logger.info(f"Loaded perplexity results for {len(results)} checkpoints from {path.name}.")
    return results

def perplexity_series(perplexity_by_step: dict[int, dict], steps: list[int]) -> np.ndarray:
    """
    Convert a per-step pseudo-perplexity series to a float array.

    :param perplexity_by_step: Mapping from checkpoint step to perplexity results.
    :param steps: Checkpoint steps to read, in plotting order.
    :return: Float array of the same length.
    """
    values = [perplexity_by_step[step].get("pseudo_perplexity") for step in steps]
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
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

def configure_axes(ax, steps: list[int], title: str) -> None:
    """
    Apply labels, scales and grid styling to a perplexity plot.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    :param title: Plot title.
    """
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("Pseudo-Perplexity (PPPL)")
    ax.set_title(title)
    configure_step_axis(ax, steps)
    ax.set_yscale("log")
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

def plot_perplexity_dynamics(perplexity_by_step: dict[int, dict], model_name: str, output_path: Path) -> None:
    """
    Plot pseudo-perplexity for a single model across checkpoints.

    :param perplexity_by_step: Mapping from checkpoint step to perplexity results.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    steps = sorted(perplexity_by_step.keys())
    pppl = perplexity_series(perplexity_by_step, steps)

    ax.plot(steps, pppl, linewidth=LINE_WIDTH, color="cornflowerblue")

    # Add labels and formatting
    configure_axes(ax, steps, f"Pseudo-Perplexity Dynamics ({model_name})")

    save_figure(fig, output_path, f"{model_name} pseudo-perplexity")

def plot_model_perplexity(perplexity_by_model: dict[str, dict[int, dict]], output_path: Path) -> None:
    """
    Plot the pseudo-perplexity for all models on a single figure.

    :param perplexity_by_model: Mapping from model name to perplexity results.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    all_steps = []

    # Plot the pseudo-perplexity for each model
    for i, (model, perplexity_by_step) in enumerate(perplexity_by_model.items()):
        steps = sorted(perplexity_by_step.keys())
        all_steps.extend(steps)

        pppl = perplexity_series(perplexity_by_step, steps)
        color = PALETTE[i % len(PALETTE)]

        ax.plot(steps, pppl, label=MODEL_DISPLAY_NAMES[model], linewidth=LINE_WIDTH, color=color)

    # Add labels and formatting
    configure_axes(ax, sorted(set(all_steps)), "Pseudo-Perplexity Dynamics")
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, output_path, "all-models overview")

def main() -> None:
    """Main entry point for plotting pseudo-perplexity dynamics."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the perplexity results for every model that has been evaluated
    perplexity_by_model = {}

    for model in args.models:
        perplexity_path = results_dir / f"{model}-large" / "perplexity" / f"{args.language}_perplexity.json"

        if not perplexity_path.exists():
            logger.warning(f"No perplexity results for '{model}' in {results_dir}, skipping.")
            continue

        perplexity_by_step = load_perplexity_results(perplexity_path)
        if perplexity_by_step:
            perplexity_by_model[model] = perplexity_by_step
        else:
            logger.warning(f"Perplexity results for '{model}' are empty, skipping.")

    if not perplexity_by_model:
        logger.error(f"No perplexity results found in {results_dir}")
        return

    # Plot pseudo-perplexity dynamics
    for model, perplexity_by_step in perplexity_by_model.items():
        model_name = MODEL_DISPLAY_NAMES[model]
        filename = model_name.lower().replace(" ", "_").replace("-", "_")

        # Save each model's plot alongside its other plots
        model_plots_dir = results_dir / f"{model}-large" / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        plot_perplexity_dynamics(
            perplexity_by_step,
            model_name=model_name,
            output_path=model_plots_dir / f"{filename}_{args.language}_perplexity.png"
        )

    # Plot pseudo-perplexity for all models on a single figure
    if len(perplexity_by_model) < 2:
        logger.info("Fewer than two models plotted, skipping the all-models overview.")
        return

    plot_model_perplexity(
        perplexity_by_model,
        output_path=output_dir / f"all_models_{args.language}_perplexity.png"
    )

if __name__ == "__main__":
    main()
