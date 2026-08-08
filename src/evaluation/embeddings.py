"""
Helpers for extracting static (non-contextual) embedding-layer representations from a model.

Based on Conneau et al. (2020) for analysing the static embedding layer, and 
Reimers and Gurevych (2019) for mean pooling token representations.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Constant Values
MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 64
PIVOT_LANGUAGE = "eng_Latn"

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_flores_pairs(flores_pairs_dir: Path, language: str) -> tuple[list[str], list[str]]:
    """
    Load parallel English and target-language sentences from FLORES-200.

    :param flores_pairs_dir: Path to the downloaded FLORES-200 paired dataset.
    :param language: Target language.
    :return: Tuple of (English sentences, target-language sentences).
    """
    dataset = cast(Dataset, load_from_disk(str(flores_pairs_dir)))
    english_sentences = list(dataset[f"sentence_{PIVOT_LANGUAGE}"])
    target_sentences = list(dataset[f"sentence_{language}"])

    logger.info(f"Loaded {len(english_sentences):,} parallel English-{language} pairs.")
    return english_sentences, target_sentences

def embed_sentences(sentences: list[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                    device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Compute mean-pooled static embeddings for a list of sentences, in batches.

    :param sentences: List of raw sentence texts.
    :param model: Model to extract the embedding layer from.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to run the embedding lookup on.
    :param batch_size: Number of sentences to embed per batch.
    :return: Array of shape (N, D), one mean-pooled embedding per sentence.
    """
    embedding_layer = model.get_input_embeddings()
    batch_embeddings = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]

        tokenized_batch = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_special_tokens_mask=True
        )

        input_ids = tokenized_batch["input_ids"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)
        special_tokens_mask = tokenized_batch["special_tokens_mask"].to(device)

        # Pool over real tokens only, excluding padding and special tokens
        content_mask = attention_mask * (1 - special_tokens_mask)

        # Fall back to all non-padding tokens for sentences with no content tokens
        empty_rows = content_mask.sum(dim=1) == 0
        if empty_rows.any():
            content_mask[empty_rows] = attention_mask[empty_rows]

        # Extract the token embeddings from the embedding layer
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids)

        # Compute the masked mean across token embeddings
        mask = content_mask.unsqueeze(-1).to(token_embeddings.dtype)
        mean_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

        batch_embeddings.append(mean_embeddings.float().cpu().numpy())

    return np.concatenate(batch_embeddings, axis=0)

def compute_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """
    Compute a cosine similarity matrix between two sets of embeddings.
    
    Each embedding is L2-normalised before computing pairwise cosine 
    similarities, following Idris et al. (2026)

    :param embeddings_a: Array of shape (N, D), one embedding per row.
    :param embeddings_b: Array of shape (M, D), one embedding per row.
    :return: Similarity matrix of shape (N, M), where entry [i, j] is the
             cosine similarity between embeddings_a[i] and embeddings_b[j].
    """
    normed_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    normed_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
    
    return normed_a @ normed_b.T