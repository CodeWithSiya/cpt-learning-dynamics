"""
Compute cross-lingual alignment across CPT checkpoints using FLORES-200, 
following Idris et al. (2026).
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    logging as hf_logging
)

from src.evaluation.embeddings import (
    DEFAULT_BATCH_SIZE,
    PIVOT_LANGUAGE,
    load_flores_pairs,
    embed_sentences,
    compute_similarity_matrix
)

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constant Values
IS_GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IS_GPU_AVAILABLE else "cpu")

# Supported target languages, matching the paired subsets saved by download_flores.py
SUPPORTED_LANGUAGES = ["xho_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute cross-lingual alignment for CPT checkpoints."
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
        help="FLORES-200 target language to align against English."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File path to save the cosine similarity results JSON to."
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

    # Search for checkpoint directories that begin with 'step-'
    for path in checkpoint_dir.iterdir():
        if path.is_dir() and path.name.startswith("step-"):
            checkpoints.append(path)

    # Sort the checkpoints by step numbers
    checkpoints.sort(key=checkpoint_step)

    return checkpoints

def compute_alignment_score(english_sentences: list[str], target_sentences: list[str], 
                            model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: torch.device,
                            batch_size: int) -> dict:
    """
    Compute the cosine gap and top-1 retrieval accuracy (P@1) between
    aligned translation pairs and a baseline, for a single checkpoint.

    :param english_sentences: List of English sentences.
    :param target_sentences: List of parallel target-language sentences.
    :param model: Model to extract embeddings from.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to run the embedding lookup on.
    :param batch_size: Number of sentences to embed per forward pass.
    :return: Dictionary with matched, baseline and cosine gap scores.
    """
    # Compute static embeddings for each example sentence
    english_embeddings = embed_sentences(english_sentences, model, tokenizer, device, batch_size)
    target_embeddings = embed_sentences(target_sentences, model, tokenizer, device, batch_size)

    # Compute a cosine similarity matrix between the parallel sentence embeddings
    similarity_matrix = compute_similarity_matrix(english_embeddings, target_embeddings)

    # Compute the mean cosine similarity of matched translation pairs
    matched_mean = float(np.mean(np.diag(similarity_matrix)))

    # Compute the baseline average over all non-matched pairs
    n = similarity_matrix.shape[0]
    baseline_scores = similarity_matrix[~np.eye(n, dtype=bool)]
    baseline_mean = float(np.mean(baseline_scores))

    # Compute top-1 retrieval accuracy (P@1) in both directions
    target_to_english_predictions = similarity_matrix.argmax(axis=1)
    p_at_1_target_to_english = float(np.mean(target_to_english_predictions == np.arange(n)))

    english_to_target_predictions = similarity_matrix.argmax(axis=0)
    p_at_1_english_to_target = float(np.mean(english_to_target_predictions == np.arange(n)))

    return {
        "matched_cosine_similarity": matched_mean,
        "baseline_cosine_similarity": baseline_mean,
        "cosine_gap": matched_mean - baseline_mean,
        "p_at_1_target_to_english": p_at_1_target_to_english,
        "p_at_1_english_to_target": p_at_1_english_to_target,
    }

def main():
    """Parse CLI arguments and compute cross-lingual alignment across checkpoints."""
    args = parse_args()

    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    flores_pairs_dir = Path(args.flores_dir) / f"{PIVOT_LANGUAGE}-{args.language}"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all CPT checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")

    if not checkpoints:
        raise ValueError(f"No 'step-' checkpoint directories found in {checkpoint_dir}.")

    # Load FLORES-200 parallel sentences
    english_sentences, target_sentences = load_flores_pairs(flores_pairs_dir, args.language)

    # CPT leaves the tokenizer unchanged, so load it once from the first checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
    hf_logging.set_verbosity_warning()

    results = {}

    for checkpoint_path in checkpoints:
        step = checkpoint_step(checkpoint_path)
        logger.info(f"Computing alignment for {checkpoint_path.name}...")

        # Load the base encoder for this checkpoint
        hf_logging.set_verbosity_error()
        model = AutoModel.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()
        hf_logging.set_verbosity_warning()

        # Compute alignment scores for this checkpoint
        alignment_scores = compute_alignment_score(
            english_sentences, target_sentences, model, tokenizer, DEVICE, args.batch_size
        )
        results[step] = alignment_scores

        logger.info(f"{checkpoint_path.name}: {alignment_scores}")

        # Free GPU memory before loading the next checkpoint
        del model
        if IS_GPU_AVAILABLE:
            torch.cuda.empty_cache()

        # Save results after each checkpoint
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info(f"Saved alignment results for {len(results)} checkpoints to {output_path}")

if __name__ == "__main__":
    main()