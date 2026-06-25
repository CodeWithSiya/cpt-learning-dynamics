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
        batch_size: Per-step batch size.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        output_dir: Directory to write checkpoints and logs to.
        checkpoint_schedule: Checkpoint schedule configuration.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """
    model_name_or_path: str
    max_seq_length: int
    learning_rate: float
    batch_size: int
    total_steps: int
    warmup_steps: int
    output_dir: str
    checkpoint_schedule: CheckpointScheduleConfig
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than total_steps ({self.total_steps})"
            )
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")

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