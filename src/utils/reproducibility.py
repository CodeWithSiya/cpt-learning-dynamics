"""
Determinism settings shared by the pretraining and fine-tuning runs.
"""

import os
import warnings

import torch

def set_reproducibility() -> None:
    """Configure determinism settings for training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message="Flash Attention defaults to a non-deterministic algorithm")