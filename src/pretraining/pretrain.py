""" 
Continued Pretraining (CPT) script for encoder-only models on a given language.

The script performs the following tasks:
1. Loads a ModelConfig from YAML and a pre-tokenized corpus (produced by src/data/preprocess_corpus.py).
2. Runs dynamic masked language modelling using the HuggingFace Trainer.
3. Saves model checkpoints using the schedule defined in schedule.py.
""" 

import os
import argparse
import json
import warnings

from dotenv import load_dotenv
from argparse import Namespace
from typing import cast, Optional
from pathlib import Path

import torch

from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerControl,
    TrainerCallback,
    TrainerState,
    logging,
)
from transformers.trainer_utils import get_last_checkpoint

from src.pretraining.config import ModelConfig
from src.pretraining.schedule import compute_checkpoint_steps

# Constant Values (MLM probability from Devlin et al. [2018])
RANDOM_SEED = 42
MLM_PROBABILITY = 0.15
IS_GPU_AVAILABLE = torch.cuda.is_available()

class ProgressCallback(TrainerCallback):
    """Log training loss at each logging step."""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs) -> None:
        """
        Print step and loss whenever the trainer logs.

        :param args: TrainingArguments for the current run.
        :param state: TrainerState tracking progress.
        :param control: TrainerControl flags.
        :param logs: Dict of logged values, expected to contain "loss".
        """
        if logs and "loss" in logs:
            print(f"Step {state.global_step}/{args.max_steps} | Loss: {logs['loss']:.4f}", flush=True)

class CheckpointScheduleCallback(TrainerCallback):
    """Save a checkpoint directly on a steps dictated by the two-phase schedule."""

    def __init__(self, checkpoint_steps: list[int], save_dir: Path, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Store the precomputed checkpoint and target directory.

        :param checkpoint_steps: Steps at which to save, from compute_schedule_step().
        :param save_dir: Directory under which step subfolders are created.
        :param tokenizer: Tokenizer to save alongside each model checkpoint.
        """
        self.checkpoint_steps = set(checkpoint_steps)
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """
        Save the initial baseline checkpoint before training starts, if scheduled.

        :param args: TrainingArguments for the current run.
        :param state: TrainerState tracking process.
        :param control: TrainerControl flags.
        """
        if state.global_step > 0:
            print("Resuming training.", flush=True)
            return

        if 0 in self.checkpoint_steps and state.is_world_process_zero:
            model = kwargs.get("model")
            if model is not None:
                model.save_pretrained(self.save_dir / "step-0")
                self.tokenizer.save_pretrained(self.save_dir / "step-0")
                print("Saved checkpoint: step-0", flush=True)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Save directly when the current step is in the schedule.

        :param args: TrainingArguments for the current run.
        :param state: TrainerState tracking progress.
        :param control: TrainerControl flags.
        """
        if state.global_step in self.checkpoint_steps and state.is_world_process_zero: 
            kwargs["model"].save_pretrained(self.save_dir / f"step-{state.global_step}")
            self.tokenizer.save_pretrained(self.save_dir / f"step-{state.global_step}")
            print(f"Saved checkpoint: step-{state.global_step}", flush=True)
            
def set_reproducibility() -> None:
    """Configure determinism settings for training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message="Flash Attention defaults to a non-deterministic algorithm")

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run continued pretraining for a single model config."
    )
    parser.add_argument(
        "--model-config", 
        type=str, 
        required=True, 
        help="Path to model YAML config."
    )
    parser.add_argument(
        "--train-corpus",
        type=str,
        required=True,
        help="Path to the pre-tokenized training corpus on disk (output of src/data/preprocess_corpus.py)."
    )
    parser.add_argument(
        "--validation-corpus",
        type=str,
        required=True,
        help="Path to the pre-tokenized validation corpus on disk (output of src/data/preprocess_corpus.py)."
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional W&B run ID for resuming a specific run. If omitted, a new run is created."
    )
    return parser.parse_args() 

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a checkpoint directory name."""
    return int(path.name.split("-")[1])

def get_full_log_history(checkpoint_dir: Path) -> list[dict]:
    """
    Retrieve the complete log history from the most recent resumption checkpoint.

    :param checkpoint_dir: Directory containing checkpoint-N subfolders.
    :return: Full log history across resumptions. 
    """   
    checkpoints = []     

    # Search for all checkpoint directories and sort them
    for path in checkpoint_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            checkpoints.append(path)

    if not checkpoints:
        return []

    checkpoints.sort(key=checkpoint_step)  

    # Get the training history fron the latest checkpoint
    latest_checkpoint = checkpoints[-1]
    with open(latest_checkpoint / "trainer_state.json") as f:
        state = json.load(f)

    return state.get("log_history", [])

def run_pretraining(config: ModelConfig, train_corpus_path: str, validation_corpus_path: str, wandb_run_id: Optional[str] = None) -> None:
    """
    Run continued MLM pretraining for a single model config.

    :param config: ModelConfig loaded from YAML.
    :param train_corpus_path: Path to the pre-tokenized, chunked training corpus on disk.
    :param validation_corpus_path: Path to the pre-tokenized, chunked validation corpus on disk.
    :param wandb_run_id: Optional W&B run ID for resuming a specific run.
    """
    total_steps = config.total_steps

    # Initialise the model's pretrained tokenizer
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def model_init():
        """Return a new model instance, loading pretrained weights as the CPT starting point."""
        return AutoModelForMaskedLM.from_pretrained(config.model_name_or_path)

    logging.set_verbosity_warning()

    # Load the preprocessed training and evaluation corpora from disk
    chunked_train_corpus = load_from_disk(train_corpus_path)
    print(f"Loaded {len(chunked_train_corpus):,} pre-tokenized chunks from {train_corpus_path}", flush=True)

    chunked_validation_corpus = load_from_disk(validation_corpus_path)
    print(f"Loaded {len(chunked_validation_corpus):,} pre-tokenized chunks from {validation_corpus_path}", flush=True)

    # Compute the CPT checkpoint steps
    checkpoint_steps = compute_checkpoint_steps(total_steps, config=config.checkpoint_schedule)
    print(f"Checkpoint schedule ({len(checkpoint_steps)} checkpoints): {checkpoint_steps}", flush=True)

    # Initialise output directory
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure Weights and Biases for CPT tracking and resumption support
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    if config.wandb_project and wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"

    # Training Arguments (Checkpointing handled by CheckpointScheduleCallBack)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        max_steps=total_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        save_strategy="steps",  # Enables resumption checkpoints
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=RANDOM_SEED,
        report_to="wandb" if config.wandb_project else "none",
        run_name=config.wandb_run_name if config.wandb_project else None,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        bf16=IS_GPU_AVAILABLE,
        dataloader_num_workers=4,
        dataloader_pin_memory=IS_GPU_AVAILABLE,
        use_cpu=not IS_GPU_AVAILABLE,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=config.use_early_stopping
    )

    # Initialise training callbacks
    callbacks=[
            ProgressCallback(),
            CheckpointScheduleCallback(checkpoint_steps=checkpoint_steps, save_dir=checkpoint_dir, tokenizer=tokenizer),
    ]

    if config.use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    # Model Training
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=cast(Dataset, chunked_train_corpus),
        eval_dataset=cast(Dataset, chunked_validation_corpus),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
        ),
        processing_class=tokenizer,
        callbacks=callbacks
    )

    # Resume from the latest resumption checkpoint if one exists
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming training from {last_checkpoint}", flush=True)
    else:
        print(f"No existing checkpoint found. Starting training from scratch.", flush=True)

    print(f"Running CPT for {total_steps} steps on {config.model_name_or_path}...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model and full log history
    final_step = trainer.state.global_step
    trainer.save_model(str(checkpoint_dir / f"step-{final_step}"))
    tokenizer.save_pretrained(str(checkpoint_dir / f"step-{final_step}"))

    full_history = get_full_log_history(checkpoint_dir)
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(full_history, f, indent=2)

    print("CPT complete.", flush=True)

def main() -> None:
    """Parse CLI arguments and launch a CPT run."""
    load_dotenv()
    args = parse_args()
    set_reproducibility()

    # Log device information
    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load model configuration and launch CPT run
    config = ModelConfig.from_yaml(args.model_config)
    run_pretraining(
        config, 
        train_corpus_path=args.train_corpus, 
        validation_corpus_path=args.validation_corpus,
        wandb_run_id=args.wandb_run_id
    )

if __name__ == '__main__':
    main()