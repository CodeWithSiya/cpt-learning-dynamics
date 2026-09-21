"""
Compute layer-wise cross-lingual alignment for a subset of CPT checkpoints using FLORES-200.
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

from src.geometry.embeddings import (
    DEFAULT_BATCH_SIZE,
    PIVOT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    compute_iso_score,
    load_flores_pairs,
    embed_sentences_by_layer,
    matched_cosine_similarities
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

# Metrics computed at every layer
LAYERWISE_METRICS = [
    "matched_cosine_similarity",
    "iso_score_shared"
]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute layer-wise cross-lingual alignment for a subset of CPT checkpoints."
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
        "--checkpoints",
        type=str,
        required=True,
        help="Comma-separated list of checkpoint steps to analyse."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File path to save the layer-wise alignment results JSON to."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sentences to embed per forward pass. Default: {DEFAULT_BATCH_SIZE}."
    )
    return parser.parse_args()

def compute_layerwise_alignment(english_sentences: list[str], target_sentences: list[str], model: PreTrainedModel, 
                                tokenizer: PreTrainedTokenizerBase, device: torch.device, batch_size: int) -> dict:
    """
    Compute alignment metrics for every hidden layer.

    :param english_sentences: List of English sentences.
    :param target_sentences: List of parallel target-language sentences.
    :param model: Model to extract embeddings from.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to run the embedding lookup on.
    :param batch_size: Number of sentences to embed per forward pass.
    :return: Dictionary with the number of hidden states (transformer layers plus the
             embedding layer) and one list of per-layer values per metric.
    """
    # Compute mean-pooled embeddings at every layer, for each language
    english_layers = embed_sentences_by_layer(english_sentences, model, tokenizer, device, batch_size)
    target_layers = embed_sentences_by_layer(target_sentences, model, tokenizer, device, batch_size)

    # One list of per-layer values per metric
    scores_by_layer = {metric: [] for metric in LAYERWISE_METRICS}

    for english_embeddings, target_embeddings in zip(english_layers, target_layers):
        # Mean cosine similarity of matched translation pairs
        scores_by_layer["matched_cosine_similarity"].append(
            float(np.mean(matched_cosine_similarities(english_embeddings, target_embeddings)))
        )

        # Isotropy of this layer's shared bilingual representation space
        scores_by_layer["iso_score_shared"].append(
            compute_iso_score(np.concatenate([english_embeddings, target_embeddings], axis=0))
        )

    return {"num_layers": len(scores_by_layer["matched_cosine_similarity"]), **scores_by_layer}

def main() -> None:
    """Parse CLI arguments and compute layer-wise alignment for selected checkpoints."""
    args = parse_args()

    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    flores_pairs_dir = Path(args.flores_dir) / f"{PIVOT_LANGUAGE}-{args.language}"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_steps = [int(step.strip()) for step in args.checkpoints.split(",")]

    # Keep only the requested steps that are present on disk
    checkpoints = []
    for step in requested_steps:
        checkpoint_path = checkpoint_dir / f"step-{step}"
        if checkpoint_path.exists():
            checkpoints.append((step, checkpoint_path))
        else:
            logger.warning(f"Checkpoint step-{step} not found in {checkpoint_dir}, skipping.")

    if not checkpoints:
        raise ValueError(f"None of the requested checkpoints were found in {checkpoint_dir}.")

    # Load FLORES-200 parallel sentences
    english_sentences, target_sentences = load_flores_pairs(flores_pairs_dir, args.language)

    # CPT leaves the tokenizer unchanged, so load it once from the first checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])
    hf_logging.set_verbosity_warning()

    results = {}

    for step, checkpoint_path in checkpoints:
        logger.info(f"Computing layer-wise alignment for step-{step}...")

        # Load the base encoder for this checkpoint
        hf_logging.set_verbosity_error()
        model = AutoModel.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()
        hf_logging.set_verbosity_warning()

        # Compute layer-wise alignment scores for this checkpoint
        layerwise_scores = compute_layerwise_alignment(
            english_sentences, target_sentences, model, tokenizer, DEVICE, args.batch_size
        )
        results[step] = layerwise_scores

        logger.info(
            f"step-{step}: {layerwise_scores['num_layers']} layers, "
            f"matched cosine similarity range "
            f"[{min(layerwise_scores['matched_cosine_similarity']):.4f}, "
            f"{max(layerwise_scores['matched_cosine_similarity']):.4f}]"
        )

        # Free GPU memory before loading the next checkpoint
        del model
        if IS_GPU_AVAILABLE:
            torch.cuda.empty_cache()

        # Save results after each checkpoint
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info(f"Saved layer-wise alignment results for {len(results)} checkpoints to {output_path}")

if __name__ == "__main__":
    main()
