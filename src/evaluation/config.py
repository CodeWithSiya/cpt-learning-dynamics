"""Configuration for downstream evaluation tasks."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import yaml

class TaskType(Enum):
    """Enum which represents a downstream evaluation task type."""
    TOKEN_CLASSIFICATION = "token_classification"
    SEQUENCE_CLASSIFICATION = "sequence_classification"

@dataclass
class EvaluationConfig:
    """
    Configuration for a single downstream evaluation task.

    Attributes:
        task_name: Short identifier for the task.
        task_type: Type of the downstream task.
        dataset_path: Local path to the downloaded evaluation dataset.
        input_field: Column name containing the model input.
        label_field: Column name containing the true labels.
        label_names: List of label names corresponding to integer label ids.
        learning_rate: Fine-tuning learning rate.
        epochs: Number of fine-tuning epochs.
        batch_size: Per-device batch size.
        warmup_steps: Number of linear warmup steps.
    """
    task_name: str
    task_type: TaskType
    dataset_path: str
    input_field: str
    label_field: str
    label_names: list[str]
    learning_rate: float
    epochs: int
    batch_size: int
    warmup_steps: int

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not self.label_names:
            raise ValueError("label_names must not be empty")
        
    @property
    def num_labels(self) -> int:
        """Number of labels for this task."""
        return len(self.label_names)
    
    @property
    def id2label(self) -> dict[int, str]:
        """Mapping from integer id to label name."""
        return {i: label for i, label in enumerate(self.label_names)}
    
    @property
    def label2id(self) -> dict[str, int]:
        """Mapping from label name to integer id."""
        return {label: i for i, label in enumerate(self.label_names)}
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        """
        Load an EvalConfig from a YAML file.

        :param path: Path to the YAML config file.
        :return: Populated EvalConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        data["task_type"] = TaskType(data["task_type"])
        return cls(**data)