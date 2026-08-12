"""
Download evaluation datasets from HuggingFace for downstream task evaluation.

Datasets:
    - https://huggingface.co/datasets/masakhane/masakhaner2
    - https://huggingface.co/datasets/masakhane/masakhapos
    - https://huggingface.co/datasets/masakhane/masakhanews
    - https://github.com/uds-lsv/afro-maft/tree/main/dataset/ANTC
"""

import argparse
import logging
import os
from dotenv import load_dotenv
from argparse import Namespace
from dataclasses import dataclass
from typing import cast, Optional

from datasets import DatasetDict, load_dataset
from huggingface_hub import get_token

# Load environment variables
load_dotenv()

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class EvalTaskConfig:
    """Configuration for a single evaluation task dataset."""
    dataset_name: Optional[str]
    trust_remote_code: bool
    splits: list[str]
    languages: list[str]
    description: str
    tsv_url: Optional[str] = None

# ANTC is distributed as TSV files on GitHub rather than as a HuggingFace dataset
ANTC_URL = "https://raw.githubusercontent.com/uds-lsv/afro-maft/main/dataset/ANTC/{language}/{file}.tsv"

# Remote ANTC split filenames differ from the canonical split names used downstream
ANTC_SPLIT_FILES = {"train": "train", "validation": "dev", "test": "test"}

# Evaluation tasks and datasets
EVAL_TASKS = {
    "ner": EvalTaskConfig(
        dataset_name="masakhane/masakhaner2",
        trust_remote_code=True,
        splits=["train", "validation", "test"],
        languages=["xho", "zul"],
        description="MasakhaNER 2.0: Named Entity Recognition"
    ),
    "pos": EvalTaskConfig(
        dataset_name="masakhane/masakhapos",
        trust_remote_code=True,
        splits=["train", "validation", "test"],
        languages=["xho", "zul"],
        description="MasakhaPOS: Part-of-Speech Tagging"
    ),
    "ntc_xho": EvalTaskConfig(
        dataset_name="masakhane/masakhanews",
        trust_remote_code=False,
        splits=["train", "validation", "test"],
        languages=["xho"],
        description="MasakhaNEWS: News Topic Classification"
    ),
    "ntc_zul": EvalTaskConfig(
        dataset_name=None,
        trust_remote_code=False,
        splits=["train", "validation", "test"],
        languages=["zul"],
        description="ANTC: African News Topic Classification",
        tsv_url=ANTC_URL
    ),
}

# Supported languages
SUPPORTED_LANGUAGES = ["xho", "zul"]

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets from HuggingFace."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(EVAL_TASKS.keys()),
        choices=list(EVAL_TASKS.keys()),
        help="Evaluation tasks to download. Defaults to all tasks."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        choices=SUPPORTED_LANGUAGES,
        help="Language subset to download. Default: xho (isiXhosa)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("DATA_DIR", "datasets"), "raw/evaluation"),
        help="Directory to save evaluation datasets to disk."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory."
    )
    return parser.parse_args()

def load_eval_dataset(task_name: str, config: EvalTaskConfig, language: str, cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load a single evaluation dataset from HuggingFace.

    :param task_name: Name of the evaluation task.
    :param config: EvalTaskConfig for this task.
    :param language: Language subset to download.
    :param cache_dir: HuggingFace cache directory.
    :return: DatasetDict with train, validation and test splits.
    """
    logger.info(f"Loading {config.description} ({language})...")

    # ANTC is not hosted on HuggingFace, so read its TSV files directly. Categories are left
    # as strings, so preprocessing maps them to label IDs via the task config.
    if config.tsv_url is not None:
        return cast(DatasetDict, load_dataset(
            path="csv",
            data_files={
                split: config.tsv_url.format(language=language, file=file)
                for split, file in ANTC_SPLIT_FILES.items()
            },
            delimiter="\t",
            cache_dir=cache_dir
        ))

    if config.dataset_name is None:
        raise ValueError(f"[{task_name}] has neither a HuggingFace dataset name nor a TSV URL")

    # Try to get HF token
    token = get_token()
    if token:
        logger.info("Using HuggingFace authentication")
    else:
        logger.info("No HuggingFace token found, proceeding without authentication")

    dataset = cast(DatasetDict, load_dataset(
        path=config.dataset_name,
        name=language,
        trust_remote_code=config.trust_remote_code,
        cache_dir=cache_dir,
        token=token,
        verification_mode="no_checks"
    ))

    return dataset

def log_dataset_info(task_name: str, dataset: DatasetDict) -> None:
    """
    Log basic statistics about a loaded evaluation task.
    
    :param task_name: Name of the evaluation task.
    :param dataset: Loaded DatasetDict.
    """
    logger.info(f"[{task_name}] splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        logger.info(f"[{task_name}] {split}: {len(data):,} samples")

    # Log the first sample to confirm field names
    sample = dataset["train"][0]
    logger.info(f"[{task_name}] fields: {list(sample.keys())}")

def save_eval_dataset(dataset: DatasetDict, task_name: str, language: str, output_dir: str) -> None:
    """
    Save an evaluation dataset to disk.

    :param dataset: Loaded DatasetDict to save.
    :param task_name: Name of the evaluation task.
    :param language: Language subset.
    :param output_dir: Root directory for evaluation datasets.
    """ 
    task_dir = os.path.join(output_dir, task_name, language)
    os.makedirs(task_dir, exist_ok=True)
    dataset.save_to_disk(task_dir)
    logger.info(f"[{task_name}] saved to {task_dir}")

def main() -> None:
    """Main entry point for downloading evaluation datasets."""
    args = parse_args()

    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")

    logger.info(f"Downloading tasks: {args.tasks}")
    logger.info(f"Language: {args.language}")

    for task_name in args.tasks:
        config = EVAL_TASKS[task_name]

        # Task coverage is uneven: MasakhaNEWS has no isiZulu, ANTC has no isiXhosa
        if args.language not in config.languages:
            logger.info(
                f"[{task_name}] skipping: no {args.language} subset "
                f"(available: {config.languages})"
            )
            continue

        dataset = load_eval_dataset(
            task_name=task_name,
            config=config,
            language=args.language,
            cache_dir=cache_dir
        )
        log_dataset_info(task_name, dataset)
        save_eval_dataset(
            dataset=dataset,
            task_name=task_name,
            language=args.language,
            output_dir=args.output_dir
        )

    logger.info("All downloads complete.")

if __name__ == "__main__":
    main()