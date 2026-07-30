"""
Fine-tune and evaluate encoder-only CPT checkpoints on downstream tasks.

For each CPT checkpoint, this script performs the following:
1. Loads the preprocessed evaluation dataset from disk.
2. Fine-tunes a task-specific head on a downstream dataset.
3. Evaluates on the test split.
4. Saves results alongside the checkpoint, under a seed-specific subfolder.
"""

import os
import json
import logging
import argparse
import shutil
import warnings
from copy import deepcopy
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

from sklearn.metrics import accuracy_score, classification_report

from src.finetuning.config import FinetuneConfig, TaskConfig, TaskMetric, TaskType

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constant Values
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
        "--task-config",
        type=str,
        required=True,
        help="Path to evaluation task YAML config."
    )
    parser.add_argument(
        "--finetune-config",
        type=str,
        required=True,
        help="Path to fine-tuning YAML config."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for this fine-tuning run."
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

def create_compute_metrics(task_config: TaskConfig):
    """
    Create a compute_metrics function for the HuggingFace Trainer.

    :param task_config: TaskConfig for the task.
    :return: compute_metrics function compatible with HuggingFace Trainer.
    """
    if task_config.metric == TaskMetric.SPAN_F1:
        seqeval = load_metric("seqeval")

        # Fixed set of entity types, so the macro F1 denominator is constant across checkpoints
        entity_types = list(dict.fromkeys(
            label.split("-", 1)[1]
            for label in task_config.label_names
            if "-" in label
        ))

        def compute_metrics(prediction: EvalPrediction):
            """Compute span-level per-class F1 for NER, plus macro and micro overall F1."""
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
                        true_label_seq.append(task_config.label_names[label_id])
                        true_prediction_seq.append(task_config.label_names[prediction_id])

                true_labels.append(true_label_seq)
                true_predictions.append(true_prediction_seq)

            # Compute evaluation results
            results = seqeval.compute(
                predictions=true_predictions,
                references=true_labels,
            ) or {}

            # Extract per-entity-type F1 for learning dynamics plotting
            per_class_f1 = {}
            for entity_type in entity_types:
                metrics = results.get(entity_type)
                per_class_f1[entity_type] = (
                    float(metrics["f1"])
                    if isinstance(metrics, dict) and "f1" in metrics else 0.0
                )

            # Compute Macro F1 score
            macro_f1 = (
                sum(per_class_f1.values()) / len(per_class_f1)
                if per_class_f1 else 0.0
            )

            return {
                "f1": macro_f1,                                    
                "precision": results.get("overall_precision", 0.0),
                "recall": results.get("overall_recall", 0.0),
                "micro_f1": results.get("overall_f1", 0.0),
                "per_class_f1": per_class_f1
            }

    elif task_config.metric == TaskMetric.TOKEN_F1:

        def compute_metrics(prediction: EvalPrediction):
            """Compute token-level per-class F1 for POS, plus macro F1 and accuracy."""
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

            # Restrict scoring to tags that actually occur in the references
            present_ids = sorted(set(true_labels))
            present_tags = [task_config.label_names[label_id] for label_id in present_ids]

            # Per-tag F1 breakdown for learning dynamics plotting
            report = cast(dict, classification_report(
                true_labels,
                true_predictions,
                labels=present_ids,
                target_names=present_tags,
                output_dict=True,
                zero_division=0
            ))
            per_class_f1 = {
                tag: report[tag]["f1-score"]
                for tag in present_tags
                if tag in report
            }

            return {
                "f1": report["macro avg"]["f1-score"],
                "accuracy": accuracy_score(true_labels, true_predictions),
                "weighted_f1": report["weighted avg"]["f1-score"],
                "per_class_f1": per_class_f1
            }

    elif task_config.metric == TaskMetric.SEQUENCE_F1:

        def compute_metrics(prediction: EvalPrediction):
            """Compute sequence-level per-class F1 for NTC, plus macro F1 and accuracy."""
            # Extract logits and labels, then compute predicted label IDs
            predictions, labels = unpack_predictions(prediction)

            # Per-category F1 breakdown for learning dynamics plotting
            report = cast(dict, classification_report(
                labels,
                predictions,
                labels=list(range(len(task_config.label_names))),
                target_names=task_config.label_names,
                output_dict=True,
                zero_division=0
            ))
            per_class_f1 = {
                category: report[category]["f1-score"]
                for category in task_config.label_names
                if category in report
            }

            return {
                "f1": report["macro avg"]["f1-score"], 
                "accuracy": report["accuracy"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "per_class_f1": per_class_f1
            }

    else:
        raise ValueError(
            f"Unsupported metric: {task_config.metric}. "
            f"Expected one of: span, token, sequence."
        )

    return compute_metrics

def finetune_and_evaluate(checkpoint_path: Path, task_config: TaskConfig, finetune_config: FinetuneConfig, dataset: DatasetDict, output_dir: Path, seed: int) -> dict[str, object]:
    """
    Fine-tune a CPT checkpoint on a single downstream task and evaluate it on the test split.
    
    :param checkpoint_path: Path to the CPT checkpoint.
    :param task_config: TaskConfig for the task.
    :param finetune_config: FinetuneConfig with fine-tuning hyperparameters.
    :param dataset: Preprocessed DatasetDict loaded from disk.
    :param output_dir: Directory to save fine-tuning results to.
    :param seed: Random seed for this fine-tuning run.
    :return: Dict containing test metrics and full training log history.
    """
    step = checkpoint_path.name
    logger.info(f"Fine-tuning {step} on {task_config.task_name} (seed={seed})")

    # Load tokenizer from the CPT checkpoint
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    hf_logging.set_verbosity_warning()

    def model_init():
        """Load a CPT checkpoint with a newly initialised task-specific head."""
        kwargs = dict(
            num_labels=task_config.num_labels,
            id2label=task_config.id2label,
            label2id=task_config.label2id,
            ignore_mismatched_sizes=True  # Randomly initialise the task-specific head and reuse the CPT encoder weights
        )

        # Load the checkpoint with the appropriate task-specific classification head
        if task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
            return AutoModelForTokenClassification.from_pretrained(checkpoint_path, **kwargs)
        else:
            return AutoModelForSequenceClassification.from_pretrained(checkpoint_path, **kwargs)
        
    # Select the data collator for the downstream task
    if task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    # Output directory for this (checkpoint, task) pair
    run_dir = output_dir / step / task_config.task_name / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure W&B run name for this fine-tuning run
    if finetune_config.wandb_project:
        os.environ["WANDB_PROJECT"] = finetune_config.wandb_project
        
    wandb_project = os.environ.get("WANDB_PROJECT")
    run_name = f"{step}-{task_config.task_name}-seed-{seed}" if wandb_project else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=finetune_config.epochs,
        learning_rate=finetune_config.learning_rate,
        per_device_train_batch_size=finetune_config.batch_size,
        per_device_eval_batch_size=finetune_config.batch_size,
        warmup_ratio=finetune_config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        seed=seed,
        report_to="wandb" if wandb_project else "none",
        run_name=run_name,
        bf16=IS_GPU_AVAILABLE,
        use_cpu=not IS_GPU_AVAILABLE,
        dataloader_num_workers=4,
        dataloader_pin_memory=IS_GPU_AVAILABLE,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{task_config.best_model_metric}",
        greater_is_better=True,
    )

    # Model training
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(task_config)
    )

    trainer.train()

    # Snapshot the training log history before evaluating on the test split
    log_history = deepcopy(trainer.state.log_history)

    # Evaluate on test split
    logger.info(f"Evaluating {step} / {task_config.task_name} on test split...")
    test_metrics = trainer.evaluate(cast(DatasetDict, dataset["test"]))

    results = {
        "checkpoint": step,
        "task": task_config.task_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "log_history": log_history,
    }

    # Save results alongside the checkpoint
    results_path = run_dir / "results.json"
    with open(results_path,"w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Discard the fine-tuned weights and keep only the metrics for this (checkpoint, task, seed)
    for checkpoint in run_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint, ignore_errors=True)

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

    # Load task and fine-tuning configurations
    task_config = TaskConfig.from_yaml(args.task_config)
    finetune_config = FinetuneConfig.from_yaml(args.finetune_config)
    logger.info(f"Task: {task_config.task_name} ({task_config.task_type.value}), seed={args.seed}")

    # Load preprocessed evaluation dataset from disk
    dataset = cast(DatasetDict, load_from_disk(args.preprocessed_dir))
    logger.info(f"Loaded preprocessed dataset: { {k: len(v) for k, v in dataset.items()} }")

    # Loop over all checkpoints, fine-tune and evaluate the checkpoint on the downstream task
    for checkpoint_path in checkpoints:
        finetune_and_evaluate(
            checkpoint_path=checkpoint_path,
            finetune_config=finetune_config,
            task_config=task_config,
            dataset=dataset,
            output_dir=output_dir,
            seed=args.seed
        )

    logger.info(f"Fine-tuning and evaluation complete for {task_config.task_name}.")

if __name__ == '__main__':
    main()