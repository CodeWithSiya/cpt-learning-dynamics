"""
Plot CPT training and validation loss curves from a completed run's history.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot CPT training and validation loss curves."
    )
    parser.add_argument(
        "--log-history",
        type=str,
        required=True,
        help="Path to log_history.json."
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
        help="Directory to save the plot images to."
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

    logger.info(f"Loaded {len(log_history)} log entries from {path}")
    return log_history

def plot_train_loss_curve(log_history: list[dict], model_name: str, output_path: Path) -> None:
    """
    Plot the training loss curve across CPT steps.

    :param log_history: Full log history.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    # Collect training losses
    train_points = []

    for entry in log_history:
        step = entry["step"]

        if "loss" in entry:
            train_points.append((step, entry["loss"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the training loss curve
    if train_points:
        train_steps, train_values = zip(*train_points)
        ax.plot(train_steps, train_values, color="lightcoral")
    else:
        logger.warning("No training loss entries found in log history.")

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Continued Pretraining Training Loss ({model_name})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save the figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved training loss curve to {output_path}")

def plot_eval_loss_curve(log_history: list[dict], model_name: str, output_path: Path) -> None:
    """
    Plot the validation loss curve across CPT steps.

    :param log_history: Full log history.
    :param model_name: Model name, used in the plot title.
    :param output_path: File path to save the plot image to.
    """
    # Collect validation losses
    eval_points = []

    for entry in log_history:
        step = entry["step"]

        if "eval_loss" in entry:
            eval_points.append((step, entry["eval_loss"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the validation loss curve
    if eval_points:
        eval_steps, eval_values = zip(*eval_points)
        ax.plot(eval_steps, eval_values, color="cornflowerblue")
    else:
        logger.warning("No validation loss entries found in log history.")

    # Add labels and formatting
    ax.set_xlabel("Continued Pretraining Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"Continued Pretraining Validation Loss ({model_name})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save the figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved validation loss curve to {output_path}")

def main() -> None:
    """Main entry point for plotting CPT loss curves."""
    args = parse_args()

    log_history_path = Path(args.log_history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_history = load_log_history(log_history_path)
    if not log_history:
        logger.error(f"No log history found in {log_history_path}")
        return
    
    # Build the output filenames from the model name
    filename = args.model_name.lower().replace(" ", "_").replace("-", "_")
    train_output_path = output_dir / f"{filename}_train_loss.png"
    eval_output_path = output_dir / f"{filename}_eval_loss.png"

    plot_train_loss_curve(log_history, args.model_name, train_output_path)
    plot_eval_loss_curve(log_history, args.model_name, eval_output_path)

if __name__ == "__main__":
    main()