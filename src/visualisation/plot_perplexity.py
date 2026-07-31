"""
Plot pseudo-perplexity across CPT checkpoints for a single model.
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

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot pseudo-perplexity across CPT checkpoints."
    )
    parser.add_argument(
        "--perplexity-path", 
        type=str, 
        required=True, 
        help="Path to perplexity results JSON."
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        required=True, 
        help="Model name, used in the plot title and output filename."
    )
    parser.add_argument("--output", type=str, required=True, help="File path to save the plot image to.")
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
    logger.info(f"Loaded perplexity results for {len(results)} checkpoints.")
    return results

def configure_step_axis(ax, steps: list[int]) -> None:
    """
    Configure the checkpoint step axis.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    """
    ax.set_xlim(min(steps), max(steps))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

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
    Plot pseudo-perplexity across continued pretraining checkpoints.

    :param perplexity_by_step: Mapping from checkpoint step to perplexity results.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    steps = sorted(perplexity_by_step.keys())
    pppl = np.array([perplexity_by_step[s]["pseudo_perplexity"] for s in steps], dtype=float)

    ax.plot(steps, pppl, linewidth=LINE_WIDTH, color="cornflowerblue")

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("Pseudo-Perplexity (PPPL)")
    ax.set_title(f"Pseudo-Perplexity Dynamics ({model_name})")
    configure_step_axis(ax, steps)
    ax.grid(True, alpha=0.3, linestyle=":")

    save_figure(fig, output_path, "pseudo-perplexity")

def main() -> None:
    """Main entry point for plotting pseudo-perplexity dynamics."""
    args = parse_args()

    perplexity_path = Path(args.perplexity_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    perplexity_by_step = load_perplexity_results(perplexity_path)
    if not perplexity_by_step:
        logger.error(f"No perplexity results found in {perplexity_path}")
        return

    plot_perplexity_dynamics(perplexity_by_step, model_name=args.model_name, output_path=output_path)

if __name__ == "__main__":
    main()
