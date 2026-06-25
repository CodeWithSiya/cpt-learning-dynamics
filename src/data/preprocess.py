"""
Tokenise and chunk the WURA corpus for continued pretraining (CPT) for a model configuration.
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase, logging as hf_logging

from src.pretraining.config import ModelConfig

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize and chunk the WURA corpus for a given model config."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw corpus on disk, as saved by download_corpus.py.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model YAML config (provides tokenizer and max_seq_length).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the processed dataset to."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split of the raw DatasetDict to preprocess. Default: train.",
    )
    parser.add_argument(
        "--nproc", 
        type=int,
        default=1,
        help="Number of processes for dataset preprocessing. Default: 1."
    )
    return parser.parse_args()

def tokenize_and_chunk(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, block_size: int, num_proc: int = 1) -> Dataset:
    """
    Tokenise raw text and chunk into fixed-length blocks for masked language modelling (MLM).

    :param dataset: HuggingFace Dataset with "headline" and "content" columns.
    :param tokenizer: Tokenizer matching the model being pretrained.
    :param block_size: Fixed chunk length.
    :return: Dataset of fixed-length token id chunks.
    """
    def _tokenize(examples: dict[str, list[str]]):
        """Join headline and content, then tokenize for MLM pretraining."""
        texts = [
            f"{headline}\n\n{content}"
            for headline, content in zip(examples["headline"], examples["content"])
        ]
        return tokenizer(texts, truncation=False, return_special_tokens_mask=True)
    
    def _chunk(examples: dict[str, list[list[int]]]):
        """Concatenate and chunk tokenized sequences into fixed-length blocks."""
        # Concatenate examples into one sequence
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        
        # Ensure each block is exactly block_size tokens long
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        # Split concatenated sequence into chunks of block_size tokens
        result = {k: [] for k in concatenated.keys()}
        for k, v in concatenated.items():
            for i in range(0, total_length, block_size):
                result[k].append(v[i: i + block_size])
        return result
    
    hf_logging.set_verbosity_error()

    logger.info(f"Tokenizing {len(dataset):,} documents...")
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenising documents"
    )

    hf_logging.set_verbosity_warning()

    logger.info(f"Chunking into blocks of {block_size} tokens...")
    chunked = tokenized.map(
        _chunk,
        batched=True,
        num_proc=num_proc,
        desc="Chunking sequences"
    )
    logger.info(f"Produced {len(chunked):,} chunks.")

    return chunked

def main() -> None:
    """Main entry point for preprocessing the corpus."""
    args = parse_args()

    # Load the model configuration settings
    config = ModelConfig.from_yaml(args.config)
    logger.info(f"Loaded config for {config.model_name_or_path} (max_seq_length={config.max_seq_length})")

    # Initialise the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Load the corpus split from disk
    logger.info(f"Loading raw corpus from {args.input} (split={args.split})...")
    corpus = load_from_disk(args.input)[args.split]

    # Preprocess the corpus
    chunked_corpus = tokenize_and_chunk(
        dataset=corpus, 
        tokenizer=tokenizer, 
        block_size=config.max_seq_length,
        num_proc=args.nproc
    )

    # Save the preprocessed corpus
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    chunked_corpus.save_to_disk(str(output_path))
    logger.info(f"Saved preprocessed dataset to {output_path}")

if __name__ == '__main__':
    main()