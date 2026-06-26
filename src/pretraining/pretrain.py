""" 
Continued Pretraining (CPT) script for encoder-only models on a given language.

The script performs the following tasks:
1. Loads a ModelConfig from YAML and a pre-tokenized corpus (produced by src/data/preprocess.py).
2. Runs dynamic masked language modelling using the HuggingFace Trainer.
3. Saves model checkpoints using the schedule defined in schedule.py.
""" 

import os
import argparse
import json
from dotenv import load_dotenv
from argparse import Namespace
from typing import cast
from pathlib import Path

import torch

from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerControl,
    TrainerCallback,
    TrainerState,
    logging,
)

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

    def __init__(self, checkpoint_steps: list[int], save_dir: Path) -> None:
        """
        Store the precomputed checkpoint and target directory.

        :param checkpoint_steps: Steps at which to save, from compute_schedule_step().
        :param save_dir: Directory under which step subfolders are created.
        """
        self.checkpoint_steps = set(checkpoint_steps)
        self.save_dir = save_dir
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """
        Save the initial baseline checkpoint before training starts, if scheduled.

        :param args: TrainingArguments for the current run.
        :param state: TrainerState tracking process.
        :param control: TrainerControl flags.
        """
        if 0 in self.checkpoint_steps and state.is_world_process_zero:
            model = kwargs.get("model")
            if model is not None:
                model.save_pretrained(self.save_dir / "step-0")
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
            print(f"Saved checkpoint: step-{state.global_step}", flush=True)
            
def set_reproducibility() -> None:
    """Configure reproducibility settings for training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run continued pretraining for a single model config."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to model YAML config."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to the pre-tokenized corpus on disk (output of src/data/preprocess.py)."
    )
    return parser.parse_args()           

def run_pretraining(config: ModelConfig, corpus_path: str):
    """
    Run continued MLM pretraining for a single model config.

    :param config: ModelConfig loaded from YAML.
    :param corpus_path: Path to the pre-tokenized, chunked corpus on disk.
    """
    total_steps = config.total_steps

    # Initialise the model's pretrained tokenizer
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def model_init():
        """Return a fresh model instance, loading pretrained weights as the CPT starting point."""
        return AutoModelForMaskedLM.from_pretrained(config.model_name_or_path)

    logging.set_verbosity_warning()

    # Load the preprocessed corpus from disk
    chunked_corpus = load_from_disk(corpus_path)
    print(f"Loaded {len(chunked_corpus):,} pre-tokenized chunks from {corpus_path}", flush=True)

    # Compute the CPT checkpoint steps
    checkpoint_steps = compute_checkpoint_steps(total_steps, config=config.checkpoint_schedule)
    print(f"Checkpoint schedule ({len(checkpoint_steps)} checkpoints): {checkpoint_steps}", flush=True)

    # Initialise output directory
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure Weights and Biases for CPT tracking 
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    # Training Arguments (Checkpointing handled by CheckpointScheduleCallBack)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        max_steps=total_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        save_strategy="no",
        seed=RANDOM_SEED,
        report_to="wandb" if config.wandb_project else "none",
        run_name=config.wandb_run_name if config.wandb_project else None,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        bf16= IS_GPU_AVAILABLE,
        dataloader_pin_memory=IS_GPU_AVAILABLE,
        use_cpu=not IS_GPU_AVAILABLE
    )

    # Model Training
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=cast(Dataset, chunked_corpus),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
        ),
        callbacks=[
            ProgressCallback(),
            CheckpointScheduleCallback(checkpoint_steps, save_dir=checkpoint_dir)
        ]
    )

    print(f"Running CPT for {total_steps} steps on {config.model_name_or_path}...")
    trainer.train()

    # Save final model and log history
    trainer.save_model(str(checkpoint_dir / f"step-{total_steps}"))
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("CPT complete.", flush=True)

def main() -> None:
    """Parse CLI arguments and launch a CPT run."""
    load_dotenv()
    args = parse_args()

    # Reproducibility configuration and device checks
    set_reproducibility()
    print(f"GPU available: {IS_GPU_AVAILABLE}", flush=True)
    if IS_GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load model configuration and launch CPT run
    config = ModelConfig.from_yaml(args.config)
    run_pretraining(config, corpus_path=args.corpus)

if __name__ == '__main__':
    main()