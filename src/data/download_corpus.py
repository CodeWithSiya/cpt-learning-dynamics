"""
Download the WURA isiXhosa corpus from HuggingFace for CPT.

Dataset: https://huggingface.co/datasets/castorini/wura
"""

import argparse
import logging
import os
from argparse import Namespace
from typing import cast, Optional

from datasets import DatasetDict, load_dataset

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# WURA dataset constants
DATASET_NAME = "castorini/wura"
SUPPORTED_LANGUAGES = ["xho"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the WURA isiXhosa corpus from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/corpus",
        help="Directory to save the downloaded dataset to disk."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded files. "
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        choices=SUPPORTED_LANGUAGES,
        help="WURA language subset to download. Default: xho (isiXhosa)."
    )
    return parser.parse_args()

def load_wura(language: str, cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load the given WURA language dataset from HuggingFace.
    
    :param language: WURA language subset to download.
    :param cache_dir: Path to the HuggingFace cache directory.
    :returns DatasetDict: Dataset with 'train' and 'validation' splits.
    """
    logger.info(f"Loading {DATASET_NAME} ({language})...")

    dataset = cast(DatasetDict, load_dataset(
        path=DATASET_NAME,
        name=language,
        trust_remote_code=True,
        cache_dir=cache_dir,
        verification_mode="no_checks"
    ))

    return dataset

def log_dataset_info(dataset: DatasetDict) -> None:
    """
    Log basic statistics about the loaded dataset.
    
    :param dataset: Loaded WURA DatasetDict.
    """
    logger.info(f"Dataset structure: {dataset}")
    logger.info(f"Train samples: {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")

    # Log the first sample to confirm field names and content
    sample = dataset["train"][0]
    logger.info(f"Sample fields: {list(sample.keys())}")
    logger.info(f"Sample headline: {sample['headline']}")
    logger.info(f"Sample content (first 200 chars): {sample['content'][:200]}")

def save_dataset(dataset: DatasetDict, output_dir: str) -> None:
    """
    Save the WURA dataset to disk.
    
    :param dataset: Loaded WURA DatasetDict to save.
    :param output_dir: Directory path to save the dataset to.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")

def main() -> None:
    """Main entry point for downloading the WURA corpus."""
    args = parse_args()

    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")

    # Include language in output path for multi-language support
    output_dir = os.path.join(args.output_dir, args.language)

    dataset = load_wura(language=args.language, cache_dir=cache_dir)
    log_dataset_info(dataset)
    save_dataset(dataset, output_dir)

    logger.info("Download complete.")

if __name__ == "__main__":
    main()