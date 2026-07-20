"""Configuration for continued pretraining (CPT) runs."""

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

from src.pretraining.schedule import CheckpointScheduleConfig

import yaml

@dataclass
class ModelConfig:
    """
    Configuration for a continued pretraining (CPT) run.

    Attributes:
        model_name_or_path: HuggingFace hub name or local path to the base model.
        max_seq_length: Maximum sequence length for tokenisation.
        learning_rate: Peak learning rate for training.
        per_device_batch_size: Per-device (per-GPU) micro-batch size.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        eval_steps: Frequency at which validation loss is computed.
        output_dir: Directory to write checkpoints and logs to.
        checkpoint_schedule: Checkpoint schedule configuration.
        use_early_stopping: Whether to apply early stopping based on validation loss.
        early_stopping_patience: Number of non-improving evals to tolerate before stopping.
        save_steps: Frequency at which resumption checkpoints are saved.
        save_total_limit: Maximum number of resumption checkpoints kept on disk.
        logging_steps: Frequency at which training metrics are logged in steps.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """
    model_name_or_path: str
    max_seq_length: int
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    total_steps: int
    warmup_steps: int
    eval_steps: int
    output_dir: str
    checkpoint_schedule: CheckpointScheduleConfig
    use_early_stopping: bool = False
    early_stopping_patience: int = 5
    save_steps: int = 5000
    save_total_limit: int = 2
    logging_steps: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.per_device_batch_size <= 0:
            raise ValueError(f"per_device_batch_size must be positive, got {self.per_device_batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) must be less than total_steps ({self.total_steps})")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be positive, got {self.eval_steps}")
        if self.save_steps <= 0:
            raise ValueError(f"save_steps must be positive, got {self.save_steps}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")
        if self.save_total_limit <= 0:
            raise ValueError(f"save_total_limit must be positive, got {self.save_total_limit}")
        if self.logging_steps <= 0:
            raise ValueError(f"logging_steps must be positive, got {self.logging_steps}")
        if self.save_steps % self.eval_steps != 0:
            raise ValueError(
                f"save_steps ({self.save_steps}) must be a multiple of eval_steps ({self.eval_steps}) "
                f"when load_best_model_at_end is used."
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """
        Load a ModelConfig from a YAML file.

        :param path: Path to the YAML config file.
        :return: Populated ModelConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Deserialise nested checkpoint_schedule field
        schedule_data = data.pop("checkpoint_schedule", {}) or {}
        checkpoint_schedule = CheckpointScheduleConfig(**schedule_data)

        return cls(checkpoint_schedule=checkpoint_schedule, **data)