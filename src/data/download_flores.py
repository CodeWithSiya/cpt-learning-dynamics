"""
Download the FLORES-200 devtest split for pseudo-perplexity and cross-lingual
alignment evaluation.

Dataset: https://huggingface.co/datasets/facebook/flores
"""

import argparse
import logging
import os
from dotenv import load_dotenv
from argparse import Namespace
from typing import cast, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import get_token
  
# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FLORES-200 dataset constants
DATASET_NAME = "facebook/flores"
SPLIT = "devtest"

# Single-language configurations (for pseudo-perplexity)
SUPPORTED_LANGUAGES = ["xho_Latn"]

# English-paired configs (for cross-lingual alignment)
SUPPORTED_LANGUAGE_PAIRS = ["eng_Latn-xho_Latn"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the FLORES-200 devtest split from HuggingFace."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("DATA_DIR", "datasets"), "raw/flores"),
        help="Directory to save the downloaded dataset to disk."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded files. "
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho_Latn",
        choices=SUPPORTED_LANGUAGES + SUPPORTED_LANGUAGE_PAIRS,
        help="FLORES-200 devtest subset to download."
    )
    return parser.parse_args()

def load_flores(language: str, cache_dir: Optional[str] = None) -> Dataset:
    """
    Load the given FLORES-200 language devtest split from HuggingFace.
    
    :param language: FLORES-200 devtest language subset to download.
    :param cache_dir: Path to the HuggingFace cache directory.
    :return: FLORES-200 Dataset for the devtest split.
    """
    logger.info(f"Loading {DATASET_NAME} ({language})...")

    # Try to get HF token
    token = get_token()
    if token:
        logger.info("Using HuggingFace authentication")
    else:
        logger.info("No HuggingFace token found, proceeding without authentication")

    dataset = cast(Dataset, load_dataset(
        path=DATASET_NAME,
        name=language,
        split=SPLIT,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=token,
        verification_mode="no_checks"
    ))

    return dataset

def save_dataset(dataset: Dataset, output_dir: str) -> None:
    """
    Save the FLORES-200 devtest dataset to disk in HuggingFace Arrow format.

    :param dataset: Loaded FLORES-200 devtest Dataset to save.
    :param output_dir: Directory path to save the dataset to.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")

def log_dataset_info(dataset: Dataset) -> None:
    """
    Log basic statistics about the loaded dataset.

    :param dataset: Loaded FLORES-200 devtest Dataset.
    """
    logger.info(f"Dataset structure: {dataset}")
    logger.info(f"Devtest samples: {len(dataset):,}")

def main() -> None:
    """Main entry point for downloading the FLORES-200 isiXhosa devtest split."""
    args = parse_args()

    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")

    # Include language in output path for multi-language support
    output_dir = os.path.join(args.output_dir, args.language)

    dataset = load_flores(language=args.language, cache_dir=cache_dir)
    log_dataset_info(dataset)
    save_dataset(dataset, output_dir)

    logger.info("Download complete.")

if __name__ == "__main__":
    main()