"""
Plot t-SNE visualisations of sentence embeddings across CPT checkpoints.
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import (
    AutoModel,
    AutoTokenizer,
    logging as hf_logging
)

import style
from src.evaluation.embeddings import (
    DEFAULT_BATCH_SIZE,
    PIVOT_LANGUAGE,
    load_flores_pairs,
    embed_sentences
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IS_GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IS_GPU_AVAILABLE else "cpu")
RANDOM_SEED = 42
GRID_COLUMNS = 5

SUPPORTED_LANGUAGES = ["xho_Latn", "zul_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot a grid of t-SNE visualisations across CPT checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to directory containing CPT checkpoints."
    )
    parser.add_argument(
        "--flores-dir",
        type=str,
        required=True,
        help="Path to the root FLORES-200 directory, as saved by download_flores.py."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho_Latn",
        choices=SUPPORTED_LANGUAGES,
        help="FLORES-200 target language."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name, used in the plot title and output filename."
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=100,
        help="Max number of sentence pairs to include per subplot, for readability. Default: 100."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the t-SNE grid image to."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sentences to embed per forward pass. Default: {DEFAULT_BATCH_SIZE}."
    )
    return parser.parse_args()

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a checkpoint directory name."""
    return int(path.name.split("-")[1])

def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """
    Discover all CPT checkpoints in a directory, sorted by step number.

    :param checkpoint_dir: Path to directory containing checkpoint subfolders.
    :return: Sorted list of checkpoint paths.
    """
    checkpoints = []

    for path in checkpoint_dir.iterdir():
        if path.is_dir() and path.name.startswith("step-"):
            checkpoints.append(path)

    checkpoints.sort(key=checkpoint_step)

    return checkpoints

def compute_tsne_projection(english_embeddings: np.ndarray, target_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D t-SNE projection for a checkpoint's English and
    target-language sentence embeddings.

    :param english_embeddings: Array of English sentence embeddings.
    :param target_embeddings: Array of target-language sentence embeddings.
    :return: Tuple of projected English and target-language coordinates.
    """
    # Combine both languages so t-SNE projects them into a shared 2D space
    combined = np.vstack([english_embeddings, target_embeddings])

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    projected = tsne.fit_transform(combined)

    n = len(english_embeddings)
    return projected[:n], projected[n:]

def plot_tsne_grid(projections_by_step: dict[int, tuple[np.ndarray, np.ndarray]], model_name: str, language: str, output_path: Path):
    """
    Plot a grid of t-SNE projections, one subplot per checkpoint.

    :param projections_by_step: Mapping from checkpoint step to projected coordinates.
    :param model_name: Model name, used in the plot title.
    :param language: Target language name, used in the plot labels.
    :param output_path: File path to save the plot image to.
    """
    steps = sorted(projections_by_step.keys())
    num_checkpoints = len(steps)

    num_cols = GRID_COLUMNS
    num_rows = (num_checkpoints + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)

    for index, step in enumerate(steps):
        row, col = divmod(index, num_cols)
        ax = axes[row][col]

        english_2d, target_2d = projections_by_step[step]

        ax.scatter(english_2d[:, 0], english_2d[:, 1], color=style.PALETTE[0], alpha=0.6, s=8, label="English")
        ax.scatter(target_2d[:, 0], target_2d[:, 1], color=style.PALETTE[1], alpha=0.6, s=8, label=f"{language}")

        ax.set_title(f"step-{step}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(num_checkpoints, num_rows * num_cols):
        row, col = divmod(idx, num_cols)
        axes[row][col].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved t-SNE grid to {output_path}")

def main() -> None:
    """Main entry point for plotting t-SNE grid plots."""
    args = parse_args()

    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    flores_pairs_dir = Path(args.flores_dir) / f"{PIVOT_LANGUAGE}-{args.language}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")

    if not checkpoints:
        raise ValueError(f"No 'step-' checkpoint directories found in {checkpoint_dir}.")

    english_sentences, target_sentences = load_flores_pairs(flores_pairs_dir, args.language)

    # Limit to a subset of sentences for readability
    subset_size = min(args.max_sentences, len(english_sentences))
    english_subset = english_sentences[:subset_size]
    target_subset = target_sentences[:subset_size]

    # CPT leaves the tokenizer unchanged, so load it once from the first checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
    hf_logging.set_verbosity_warning()

    projections_by_step = {}

    for checkpoint_path in checkpoints:
        step = checkpoint_step(checkpoint_path)
        logger.info(f"Computing t-SNE embeddings for {checkpoint_path.name}...")

        hf_logging.set_verbosity_error()
        model = AutoModel.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()
        hf_logging.set_verbosity_warning()

        english_embeddings = embed_sentences(english_subset, model, tokenizer, DEVICE, args.batch_size)
        target_embeddings = embed_sentences(target_subset, model, tokenizer, DEVICE, args.batch_size)

        projections_by_step[step] = compute_tsne_projection(english_embeddings, target_embeddings)

        # Free GPU memory before loading the next checkpoint
        del model
        if IS_GPU_AVAILABLE:
            torch.cuda.empty_cache()

    filename = args.model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = output_dir / f"{filename}_{args.language}_tsne_grid.png"

    plot_tsne_grid(projections_by_step, args.model_name, args.language, output_path)

if __name__ == "__main__":
    main()
