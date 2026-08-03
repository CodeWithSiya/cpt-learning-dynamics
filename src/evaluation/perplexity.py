"""
Compute pseudo-perplexity (PPPL) for CPT checkpoints on the FLORES-200 devtest split,
following Salazar et al. (2020), "Masked Language Model Scoring".
"""

import argparse
import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    logging as hf_logging
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
MAX_SEQ_LENGTH = 512

# Supported languages, matching the subsets saved by download_flores.py
SUPPORTED_LANGUAGES = ["xho_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute pseudo-perplexity for CPT checkpoints on FLORES-200."
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
        help="FLORES-200 devtest language subset to evaluate on."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File path to save the pseudo-perplexity results JSON to."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of masked copies to process per forward pass. Default: 32."
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

def load_flores_sentences(flores_dir: Path) -> list[str]:
    """
    Load FLORES-200 devtest sentences from disk.

    :param flores_dir: Path to the downloaded FLORES-200 dataset.
    :return: List of isiXhosa sentences.
    """
    dataset = cast(Dataset, load_from_disk(str(flores_dir)))
    sentences = [str(sentence) for sentence in dataset["sentence"]]
    logger.info(f"Loaded {len(sentences):,} FLORES-200 devtest sentences.")
    return sentences

def compute_sentence_pll(sentence: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, batch_size: int) -> tuple[float, int]:
    """
    Compute the pseudo-log-likelihood (PLL) of a single sentence by masking
    each non-special token in turn and summing the log-probability assigned
    to the original token at the masked position.

    :param sentence: Raw sentence text.
    :param model: Model to evaluate.
    :param tokenizer: Tokenizer corresponding to the model being evaluated.
    :param batch_size: Number of masked variants to process per forward pass.
    :return: A tuple containing the sentence PLL and its whitespace word count.
    """
    # Tokenise the sentence and add the model's special tokens
    tokenized_sentence = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_special_tokens_mask=True
    )

    input_ids = tokenized_sentence["input_ids"][0]
    special_tokens_mask = tokenized_sentence["special_tokens_mask"][0]

    # The PLL would cover only the retained prefix while the word count below covers
    # the whole sentence, silently deflating PPPL, so flag any sentence that truncates.
    if input_ids.shape[0] >= MAX_SEQ_LENGTH:
        logger.warning(
            f"Sentence truncated at {MAX_SEQ_LENGTH} tokens; its PLL will be "
            f"normalised by the untruncated word count, deflating PPPL."
        )

    # Only mask content tokens, leaving special tokens unchanged
    mask_positions = (special_tokens_mask == 0).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions:
        return 0.0, 0

    # Cache the mask token id and initialise the running PLL total
    mask_id = tokenizer.mask_token_id
    total_log_prob = 0.0

    # Process masked sentence variants in batches
    for batch_start in range(0, len(mask_positions), batch_size):
        batch_positions = mask_positions[batch_start:batch_start + batch_size]

        # Create one masked copy of the sentence for each target position
        batch_input_ids = input_ids.unsqueeze(0).repeat(len(batch_positions), 1)
        for i, pos in enumerate(batch_positions):
            batch_input_ids[i, pos] = mask_id

        batch_input_ids = batch_input_ids.to(DEVICE)

        # Compute the logits for the batch
        with torch.no_grad():
            logits = model(batch_input_ids).logits

        # Keep only the logits at the masked positions before normalising
        positions = torch.tensor(batch_positions, device=DEVICE)
        masked_logits = logits[torch.arange(len(batch_positions), device=DEVICE), positions]
        log_probs = torch.log_softmax(masked_logits, dim=-1)

        # Add the log-probability assigned to each original token
        true_token_ids = input_ids[positions.cpu()].to(DEVICE)
        total_log_prob += log_probs.gather(1, true_token_ids.unsqueeze(1)).sum().item()

    # Return the PLL together with the word count for downstream normalisation
    word_count = len(sentence.split())
    return total_log_prob, word_count

def compute_pseudo_perplexity(sentences: list[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, batch_size: int) -> float:
    """
    Compute word-normalised pseudo-perplexity (PPPL) over a corpus of sentences.

    :param sentences: List of raw sentence strings.
    :param model: Model to evaluate.
    :param tokenizer: Tokenizer matching the model.
    :param batch_size: Number of masked copies to process per forward pass.
    :return: Word-normalised pseudo-perplexity (PPPL) over the corpus.
    """
    # Accumulate the corpus pseudo-log-likelihood and word count
    total_log_prob = 0.0
    total_words = 0

    # Compute the sentence PLL for each sentence in the corpus
    for sentence in sentences:
        log_prob, word_count = compute_sentence_pll(sentence, model, tokenizer, batch_size)
        total_log_prob += log_prob
        total_words += word_count

    if total_words == 0:
        raise ValueError("Cannot compute pseudo-perplexity for an empty corpus.")

    # Convert the corpus PLL to word-normalised pseudo-perplexity
    return math.exp(-total_log_prob / total_words)

def main() -> None:
    """Parse command-line arguments and compute pseudo-perplexity for each checkpoint."""
    args = parse_args()

    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    flores_dir = Path(args.flores_dir) / args.language
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all CPT checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")

    if not checkpoints:
        raise ValueError(f"No 'step-' checkpoint directories found in {checkpoint_dir}.")

    # Load the FLORES-200 devtest sentences
    sentences = load_flores_sentences(flores_dir)

    # CPT leaves the tokenizer unchanged, so load it once from the first checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
    hf_logging.set_verbosity_warning()

    results = {}

    for checkpoint_path in checkpoints:
        step = checkpoint_step(checkpoint_path)
        logger.info(f"Computing pseudo-perplexity for {checkpoint_path.name}...")

        # Load the model for this checkpoint
        hf_logging.set_verbosity_error()
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()
        hf_logging.set_verbosity_warning()

        pppl = compute_pseudo_perplexity(sentences, model, tokenizer, args.batch_size)
        results[step] = {"pseudo_perplexity": pppl}

        logger.info(f"{checkpoint_path.name}: PPPL = {pppl:.4f}")

        # Free GPU memory before loading the next checkpoint
        del model
        if IS_GPU_AVAILABLE:
            torch.cuda.empty_cache()

        # Save results after each checkpoint
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info(f"Saved pseudo-perplexity results for {len(results)} checkpoints to {output_path}")

if __name__ == "__main__":
    main()