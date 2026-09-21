"""
Helpers for extracting the (contextual) final hidden state representations from a model.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from IsoScore import IsoScore
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Constant Values
MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 64
PIVOT_LANGUAGE = "eng_Latn"
SUPPORTED_LANGUAGES = ["xho_Latn", "zul_Latn"]

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

def tokenize_batch(batch: list[str], tokenizer: PreTrainedTokenizerBase,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenise a batch of sentences and build the content pooling mask.

    :param batch: List of raw sentence texts.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to place the tensors on.
    :return: Tuple of (input ids, attention mask, content mask), where the content
             mask covers real tokens only, excluding padding and special tokens.
    """
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

    return input_ids, attention_mask, content_mask

def mean_pool(token_embeddings: torch.Tensor, content_mask: torch.Tensor) -> np.ndarray:
    """
    Compute the masked mean across token embeddings, following Reimers and Gurevych (2019).

    :param token_embeddings: Tensor of shape (B, T, D) of per-token representations.
    :param content_mask: Tensor of shape (B, T) marking the tokens to pool over.
    :return: Array of shape (B, D), one mean-pooled embedding per sentence.
    """
    mask = content_mask.unsqueeze(-1).to(token_embeddings.dtype)
    mean_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    return mean_embeddings.float().cpu().numpy()

def embed_sentences(sentences: list[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                    device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Compute mean-pooled final-layer (contextualised) embeddings for a list of sentences, in batches.

    :param sentences: List of raw sentence texts.
    :param model: Model to run the forward pass through.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to run the embedding lookup on.
    :param batch_size: Number of sentences to embed per batch.
    :return: Array of shape (N, D), one mean-pooled embedding per sentence.
    """
    batch_embeddings = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        input_ids, attention_mask, content_mask = tokenize_batch(batch, tokenizer, device)

        # Extract the token embeddings from the final hidden state
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state

        batch_embeddings.append(mean_pool(token_embeddings, content_mask))

    return np.concatenate(batch_embeddings, axis=0)

def embed_sentences_by_layer(sentences: list[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                             device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> list[np.ndarray]:
    """
    Compute mean-pooled embeddings at every hidden layer for a list of sentences, in batches.

    :param sentences: List of raw sentence texts.
    :param model: Model to run the forward pass through.
    :param tokenizer: Tokenizer matching the model.
    :param device: Device to run the embedding lookup on.
    :param batch_size: Number of sentences to embed per batch.
    :return: List of arrays, one per layer (embedding layer first), each of shape (N, D).
    """
    # One list of batch embeddings per layer, sized on the first forward pass
    batch_embeddings_by_layer = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        input_ids, attention_mask, content_mask = tokenize_batch(batch, tokenizer, device)

        # Extract the token embeddings at every layer, including the embedding layer
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

        if not batch_embeddings_by_layer:
            batch_embeddings_by_layer = [[] for _ in hidden_states]

        # Pool each layer's token embeddings separately
        for layer, token_embeddings in enumerate(hidden_states):
            batch_embeddings_by_layer[layer].append(mean_pool(token_embeddings, content_mask))

    return [np.concatenate(batches, axis=0) for batches in batch_embeddings_by_layer]

def matched_cosine_similarities(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity of each aligned embedding pair.

    :param embeddings_a: Array of shape (N, D), one embedding per row.
    :param embeddings_b: Array of shape (N, D), aligned row-wise with embeddings_a.
    :return: Array of shape (N,), where entry [i] is the cosine similarity
             between embeddings_a[i] and embeddings_b[i].
    """
    normed_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    normed_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    return np.sum(normed_a * normed_b, axis=1)

def compute_iso_score(embeddings: np.ndarray) -> float:
    """
    Compute the IsoScore of a point cloud, following Rudman et al. (2022).

    :param embeddings: Array of shape (N, D), one embedding per row.
    :return: IsoScore in [0, 1], where 1 indicates a perfectly isotropic cloud.
    """
    return float(IsoScore.IsoScore(embeddings.astype(np.float64)))