"""
Helpers for extracting checkpoint information from a results directory.
"""

import re
from pathlib import Path

def checkpoint_step(path: Path, prefix: str = "step") -> int:
    """
    Extract the training step number from a checkpoint directory name.

    :param path: Path to the checkpoint directory.
    :param prefix: Directory name prefix preceding the step number.
    :return: Training step number.
    :raises ValueError: If the directory name does not match the expected pattern.
    """
    match = re.match(rf"{prefix}-(\d+)$", path.name)
    if not match:
        raise ValueError(f"Not a {prefix} checkpoint directory: {path}")
    return int(match.group(1))

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