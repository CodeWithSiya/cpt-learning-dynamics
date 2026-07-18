"""
Fine-tune and evaluate encoder-only CPT checkpoints on downstream tasks.

For each CPT checkpoint, this script performs the following:
1. Loads the preprocessed evaluation dataset from disk.
2. Fine-tunes a task-specific head on a downstream dataset.
3. Evaluates on the test split.
4. Saves results alongside the checkpoint.
"""

import os
import json
import logging
import argparse
import warnings
from dotenv import load_dotenv
from argparse import Namespace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from evaluate import load as load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    logging as hf_logging
)

from sklearn.metrics import classification_report

from src.evaluation.config import EvalConfig, EvalMetric, EvalTaskType

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constant Values
RANDOM_SEED = 42
IGNORE_INDEX = -100
IS_GPU_AVAILABLE = torch.cuda.is_available()

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate CPT checkpoints on a downstream task."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to directory containing CPT checkpoints."
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Path to evaluation task YAML config."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        required=True,
        help="Path to preprocessed evaluation dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuning results to disk."
    )
    return parser.parse_args()

def set_reproducibility() -> None:
    """Configure reproducibility settings for training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message="Flash Attention defaults to a non-deterministic algorithm")

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

def unpack_predictions(prediction: EvalPrediction):
    """Extract predicted and reference label IDs."""
    logits = prediction.predictions
    labels = prediction.label_ids
    predictions = np.argmax(logits, axis=-1)
    return predictions, labels

def create_compute_metrics(eval_config: EvalConfig):
    """
    Create a compute_metrics function for the HuggingFace Trainer.

    :param eval_config: EvalConfig for the task.
    :return: compute_metrics function compatible with HuggingFace Trainer.
    """
    if eval_config.metric == EvalMetric.SPAN_F1:
        seqeval = load_metric("seqeval")
        
        def compute_metrics(prediction: EvalPrediction):
            """Compute Span-level F1 for NER."""
            # Extract logits and labels, then compute predicted label IDs
            predictions, labels = unpack_predictions(prediction)

            true_labels = []
            true_predictions = []

            # Convert each sequence of token indices to label strings
            for label_seq, prediction_seq in zip(labels, predictions):
                true_label_seq = []
                true_prediction_seq = []

                for label_id, prediction_id in zip(label_seq, prediction_seq):
                    # Skip padding tokens with label -100
                    if label_id != IGNORE_INDEX:
                        true_label_seq.append(eval_config.label_names[label_id])
                        true_prediction_seq.append(eval_config.label_names[prediction_id])

                true_labels.append(true_label_seq)
                true_predictions.append(true_prediction_seq)

            # Compute evaluation results
            results = seqeval.compute(
                predictions=true_predictions,
                references=true_labels,
            ) or {}

            # Extract per-entity-type F1 for learning dynamics plotting
            per_class_f1 = {
                entity_type: metrics["f1"]
                for entity_type, metrics in results.items()
                if isinstance(metrics, dict) and "f1" in metrics
            }

            return {
                "precision": results.get("overall_precision", 0.0),
                "recall": results.get("overall_recall", 0.0),
                "f1": results.get("overall_f1", 0.0),
                "accuracy": results.get("overall_accuracy", 0.0),
                "per_class_f1": per_class_f1
            }
        
    elif eval_config.metric == EvalMetric.ACCURACY:
        accuracy_metric = load_metric("accuracy")

        def compute_metrics(prediction: EvalPrediction):
            """Compute accuracy for POS Tagging."""
            # Extract logits and labels, then compute predicted label IDs
            predictions, labels = unpack_predictions(prediction)

            true_labels = []
            true_predictions = []

            # Flatten and filter out padding tokens with label -100
            for label_seq, prediction_seq in zip(labels, predictions):
                for label_id, prediction_id in zip(label_seq, prediction_seq):
                     # Skip padding tokens with label -100
                    if label_id != IGNORE_INDEX:
                        true_labels.append(label_id)
                        true_predictions.append(prediction_id)

            # Compute accuracy result
            result = accuracy_metric.compute(
                predictions=true_predictions,
                references=true_labels
            ) or {}

            # Per-tag F1 breakdown for learning dynamics plotting
            report = cast(dict, classification_report(
                true_labels,
                true_predictions,
                labels=list(range(len(eval_config.label_names))),
                target_names=eval_config.label_names,
                output_dict=True,
                zero_division=0
            ))
            per_class_f1 = {
                tag: report[tag]["f1-score"]
                for tag in eval_config.label_names
                if tag in report
            }

            return {
                "accuracy": result.get("accuracy", 0.0),
                "per_class_f1": per_class_f1
            }
        
    elif eval_config.metric == EvalMetric.WEIGHTED_F1:
        # Extract logits and labels, then compute predicted label IDs
        f1_metric = load_metric("f1")

        def compute_metrics(prediction: EvalPrediction):
            """Compute weighted F1 for NTC."""
            predictions, labels = unpack_predictions(prediction)

            # Compute and return weighted F1 result
            result = f1_metric.compute(
                predictions=predictions,
                references=labels,
                average="weighted",
            ) or {}

            # Per-category F1 breakdown for learning dynamics plotting
            report = cast(dict, classification_report(
                labels,
                predictions,
                labels=list(range(len(eval_config.label_names))),
                target_names=eval_config.label_names,
                output_dict=True,
                zero_division=0
            ))
            per_class_f1 = {
                category: report[category]["f1-score"]
                for category in eval_config.label_names
                if category in report
            }

            return {
                "f1": result.get("f1", 0.0),
                "per_class_f1": per_class_f1
            }

    else:
        raise ValueError(
            f"Unsupported metric: {eval_config.metric}. "
            f"Expected one of: seqeval, accuracy, f1_weighted."
        )
    
    return compute_metrics

def finetune_and_evaluate(checkpoint_path: Path, eval_config: EvalConfig, dataset: DatasetDict, output_dir: Path) -> dict[str, object]:
    """
    Fine-tune a CPT checkpoint on a single downstream task and evaluate it on the test split.
    
    :param checkpoint_path: Path to the CPT checkpoint.
    :param eval_config: EvalConfig for the task.
    :param dataset: Preprocessed DatasetDict loaded from disk.
    :param output_dir: Directory to save fine-tuning results to.
    :return: Dict containing test metrics and full training log history.
    """
    step = checkpoint_path.name
    logger.info(f"Fine-tuning {step} on {eval_config.task_name}")

    # Load tokenizer from the CPT checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    hf_logging.set_verbosity_warning()

    def model_init():
        """Load a CPT checkpoint with a newly initialised task-specific head."""
        kwargs = dict(
            num_labels=eval_config.num_labels,
            id2label=eval_config.id2label,
            label2id=eval_config.label2id,
            ignore_mismatched_sizes=True  # Randomly initialise the task-specific head and reuse the CPT encoder weights
        )

        # Load the checkpoint with the appropriate task-specific classification head
        if eval_config.task_type == EvalTaskType.TOKEN_CLASSIFICATION:
            return AutoModelForTokenClassification.from_pretrained(checkpoint_path, **kwargs)
        else:
            return AutoModelForSequenceClassification.from_pretrained(checkpoint_path, **kwargs)
        
    # Select the data collator for the downstream task
    if eval_config.task_type == EvalTaskType.TOKEN_CLASSIFICATION:
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    # Output directory for this (checkpoint, task) pair
    run_dir = output_dir / step / eval_config.task_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure W&B run name for this fine-tuning run
    if eval_config.wandb_project:
        os.environ["WANDB_PROJECT"] = eval_config.wandb_project
        
    wandb_project = os.environ.get("WANDB_PROJECT")
    run_name = f"{step}-{eval_config.task_name}" if wandb_project else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=eval_config.epochs,
        learning_rate=eval_config.learning_rate,
        per_device_train_batch_size=eval_config.batch_size,
        per_device_eval_batch_size=eval_config.batch_size,
        warmup_steps=eval_config.warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=RANDOM_SEED,
        report_to="wandb" if wandb_project else "none",
        run_name=run_name,
        bf16=IS_GPU_AVAILABLE,
        use_cpu=not IS_GPU_AVAILABLE,
        dataloader_num_workers=4,
        dataloader_pin_memory=IS_GPU_AVAILABLE,
        load_best_model_at_end=False,
    )

    # Model training
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(eval_config)
    )

    trainer.train()

    # Evaluate on test split
    logger.info(f"Evaluating {step} / {eval_config.task_name} on test split...")
    test_metrics = trainer.evaluate(cast(DatasetDict, dataset["test"]))

    results = {
        "checkpoint": step,
        "task": eval_config.task_name,
        "test_metrics": test_metrics,
        "log_history": trainer.state.log_history
    }

    # Save results alongside the checkpoint
    results_path = run_dir / "results.json"
    with open(results_path,"w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    return results

def main() -> None:
    """Parse CLI arguments and run fine-tuning across all checkpoints."""
    load_dotenv()
    args = parse_args()
    set_reproducibility()

    # Log device information
    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Configure checkpoint and output paths
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all CPT checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")

    # Load evaluation configuration
    eval_config = EvalConfig.from_yaml(args.eval_config)
    logger.info(f"Task: {eval_config.task_name} ({eval_config.task_type.value})")

    # Load preprocessed evaluation dataset from disk
    dataset = cast(DatasetDict, load_from_disk(args.preprocessed_dir))
    logger.info(f"Loaded preprocessed dataset: { {k: len(v) for k, v in dataset.items()} }")

    # Loop over all checkpoints, fine-tune and evaluate the checkpoint on the downstream task
    for checkpoint_path in checkpoints:
        finetune_and_evaluate(
            checkpoint_path=checkpoint_path,
            eval_config=eval_config,
            dataset=dataset,
            output_dir=output_dir
        )

    logger.info(f"Fine-tuning and evaluation complete for {eval_config.task_name}.")

if __name__ == '__main__':
    main()