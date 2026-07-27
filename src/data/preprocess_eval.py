"""
Tokenise and prepare evaluation datasets for downstream fine-tuning.

Handles two preprocessing methods:
    1. Token Classification: tokenise pre-tokenised word sequence
       and align word-level labels.
    2. Sequence Classification: tokenise raw text and map string category
       labels to integer ids.

Sources:
    1. Token Classification: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb
    2. Sequence Classification: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb
"""

import argparse
import logging
from argparse import Namespace
from typing import cast
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, logging as hf_logging

from src.finetuning.config import TaskConfig, TaskType
from src.pretraining.config import ModelConfig

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constant Values
IGNORE_INDEX = -100

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess evaluation datasets for downstream fine-tuning."
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model YAML config (provides model_name_or_path and max_seq_length)."
    )
    parser.add_argument(
        "--task-config",
        type=str,
        required=True,
        help="Path to evaluation task YAML config."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        help="Language subset to preprocess. Default: xho (isiXhosa)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the preprocessed dataset to."
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes for dataset preprocessing. Default: 1."
    )

    return parser.parse_args()
    
def validate_label_names(dataset: DatasetDict, task_config: TaskConfig) -> None:
    """
    Check that the label names in the task config match the labels in the dataset.

    :param dataset: DatasetDict loaded from disk.
    :param task_config: TaskConfig for the task.
    :raises ValueError: If the config labels disagree with the dataset labels.
    """
    feature = dataset["train"].features[task_config.label_field]

    # Token-level labels are a sequence of class labels, so unwrap the inner feature
    class_label = getattr(feature, "feature", feature)
    dataset_label_names = getattr(class_label, "names", None)

    # Integer label ids: the config ordering must match the dataset ordering exactly
    if dataset_label_names is not None:
        if list(dataset_label_names) != list(task_config.label_names):
            raise ValueError(
                f"label_names in the '{task_config.task_name}' config do not match the dataset.\n"
                f"  config:  {list(task_config.label_names)}\n"
                f"  dataset: {list(dataset_label_names)}"
            )
        logger.info(f"Validated {len(dataset_label_names)} label names against the dataset.")
        return

    # String labels: ordering is set by the config, so only check that every label is covered
    observed = set()
    for split in dataset.values():
        observed.update(split.unique(task_config.label_field))

    unknown = observed - set(task_config.label_names)
    if unknown:
        raise ValueError(
            f"Dataset contains labels missing from the '{task_config.task_name}' config: "
            f"{sorted(unknown)}"
        )
    logger.info(f"Validated {len(observed)} label values against the dataset.")

def token_classification_preprocessor(tokenizer: PreTrainedTokenizerBase, task_config: TaskConfig, max_length: int):
    """
    Create a preprocessing function for token classification tasks.

    Tokenises input sequences and aligns word-level labels with the resulting
    subword tokens. The first subword of each word inherits the word-level label. 
    Subsequent subwords are assigned -100 and ignored by the loss function.

    :param tokenizer: Tokenizer for the model being fine-tuned.
    :param task_config: TaskConfig for the task.
    :param max_length: Maximum sequence length.
    :return: Preprocessing function for use with dataset.map().
    """
    def tokenize_and_align_labels(examples: dict[str, list]) -> BatchEncoding:
        """Tokenize example sequences using the model's pretrained tokenizer and align labels."""
        # Tokenize input examples
        tokenized_inputs = tokenizer(
            examples[task_config.input_field],
            truncation=True,
            max_length=max_length,
            is_split_into_words=True
        )

        # Perform label alignment on the tokenized inputs
        labels = []
        for i, label in enumerate(examples[task_config.label_field]):
            # Initialise word and label ID lists
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word, label_ids = None, []

            for word_id in word_ids:
                # Ignore special tokens with word_id = None
                if word_id is None:
                    label_ids.append(IGNORE_INDEX)
                # Set the label for the first token of each word
                elif word_id != prev_word:
                    label_ids.append(label[word_id])
                # For the other tokens in a word, set their label ID to -100
                else:   
                    label_ids.append(IGNORE_INDEX)
                prev_word = word_id
            
            # Append the label to the list of label IDs
            labels.append(label_ids)

        # Return the tokenized and aligned inputs
        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    
    return tokenize_and_align_labels

def sequence_classification_preprocessor(tokenizer: PreTrainedTokenizerBase, task_config: TaskConfig, max_length: int):
    """
    Create a tokenization function for sequence classification tasks.

    Tokenises input sequences and maps string category labels to integer ids 
    using the label2id mapping defined in TaskConfig.

    :param tokenizer: Tokenizer for the model being fine-tuned.
    :param task_config: TaskConfig for the task.
    :param max_length: Maximum sequence length.
    :return: Preprocessing function for use with dataset.map().
    """
    def tokenize_and_map_labels(examples: dict[str, list]) -> BatchEncoding:
        """Tokenize and map example sequences using the model's pretrained tokenizer."""
        # Tokenize input examples
        tokenized_inputs = tokenizer(
            examples[task_config.input_field],
            truncation=True,
            max_length=max_length,
            padding=False
        )

        # Map string labels to integer ids
        tokenized_inputs["labels"] = [
            task_config.label2id[category]
            for category in examples[task_config.label_field]
        ]

        return tokenized_inputs
    
    return tokenize_and_map_labels

def preprocess_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, task_config: TaskConfig, max_length: int, num_proc: int):
    """
    Tokenize and prepare an evaluation dataset for fine-tuning.
    
    :param dataset: DatasetDict loaded from disk.
    :param tokenizer: Tokenizer for the model being fine-tuned.
    :param task_config: TaskConfig for the task.
    :param max_length: Maximum sequence length.
    :param num_proc: Number of processes used for preprocessing.
    :return: Tokenized DatasetDict with all splits provided.
    """
    # Select the appropriate preprocessor function
    if task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
        preprocessor = token_classification_preprocessor(
            tokenizer, 
            task_config, 
            max_length
        )
        description = f"Tokenising and aligning labels for {task_config.task_name}"
    else:
        preprocessor = sequence_classification_preprocessor(
            tokenizer, 
            task_config, 
            max_length
        )
        description = f"Tokenising and mapping labels for {task_config.task_name}"

    # Remove column names from the dataset
    column_names = dataset["train"].column_names

    logger.info(f"{description}...")
    processed = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc,
        desc=description
    )

    logger.info(f"Preprocessed splits: { {k: len(v) for k, v in processed.items()} }")
    return processed

def main() -> None:
    """Main entry point for preprocessing the evaluation datasets."""
    args = parse_args()

    # Load model and evaluation configs
    model_config = ModelConfig.from_yaml(args.model_config)
    task_config = TaskConfig.from_yaml(args.task_config)
    logger.info(f"Model: {model_config.model_name_or_path}")
    logger.info(f"Task: {task_config.task_name} ({task_config.task_type.value})")

    # Initialise the model's tokenizer
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    hf_logging.set_verbosity_warning()

    # Load evaluation dataset from disk
    dataset_path = Path(task_config.dataset_path) / args.language
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = cast(DatasetDict, load_from_disk(str(dataset_path)))
    logger.info(f"Loaded splits: { {k: len(v) for k, v in dataset.items()} }")

    # Validate label names
    validate_label_names(dataset, task_config)

    # Preprocess the dataset
    processed_dataset = preprocess_dataset(
        dataset=dataset, 
        tokenizer=tokenizer, 
        task_config=task_config,
        max_length=model_config.max_seq_length,
        num_proc=args.nproc
    )

    # Save the preprocessed dataset to disk
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(output_path))
    logger.info(f"Saved preprocessed dataset to {output_path}")

if __name__ == '__main__':
    main()