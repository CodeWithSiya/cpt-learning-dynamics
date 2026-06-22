"""
Checkpointing schedule for continued pretraining (CPT).

Implements the two-phase checkpointing schedule from Elhady et al. (2025):
    - Phase 1 (first 10% of steps): checkpoint every 1% of total steps.
    - Phase 2 (remaining 90% of steps): checkpoint every 10% of total steps.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class CheckpointScheduleConfig:
    """
    Configuration for the two-phase checkpoint schedule.

    Attributes:
        initial_phase_ratio: Percentage of total steps in the initial dense checkpoint phase.
        initial_interval_ratio: Percentage of total steps between checkpoints during the initial phase.
        save_initial_checkpoint: Flag which specifies whether to save the initial pretraining model or not.
    """
    initial_phase_ratio: float = 0.10      # First 10% of training
    initial_interval_ratio: float = 0.01   # Checkpoint every 1%
    final_interval_ratio: float = 0.10     # Checkpoint every 10%
    save_initial_checkpoint: bool = True   # save model at step 0

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if not 0 < self.initial_phase_ratio < 1:
            raise ValueError(f"initial_phase_ratio must be in (0, 1), got {self.initial_phase_ratio}")
        if not 0 < self.initial_interval_ratio <= self.initial_phase_ratio:
            raise ValueError(f"initial_interval_ratio must be in (0, initial_phase_ratio], got {self.initial_interval_ratio}")
        if not 0 < self.final_interval_ratio <= 1:
            raise ValueError(f"final_interval_ratio must be in (0, 1], got {self.final_interval_ratio}")

def compute_checkpoint_steps(total_steps: int, config: Optional[CheckpointScheduleConfig] = None) -> list[int]:
    """
    Compute the training steps at which checkpoints should be saved.

    :param total_steps: Total number of training steps.
    :param config: Checkpoint schedule configuration. Uses default two-phase schedule if not provided.
    :return: Sorted list of steps at which to save checkpoints.
    """
    if total_steps <= 0: 
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    
    if config is None: 
        config = CheckpointScheduleConfig()

    checkpoint_steps = []

    # Include the initial step if required
    if config.save_initial_checkpoint: 
        checkpoint_steps.append(0)

    # Phase 1: Checkpoints during the initial training phase
    initial_checkpoint_interval = max(1, int(total_steps * config.initial_interval_ratio))
    initial_phase_end = int(total_steps * config.initial_phase_ratio)

    initial_phase_steps = list(
        range(initial_checkpoint_interval, initial_phase_end + 1, initial_checkpoint_interval)
    )
    checkpoint_steps.extend(initial_phase_steps)

    # Phase 2: Checkpoints for the remaining training steps
    final_checkpoint_interval = max(1, int(total_steps * config.final_interval_ratio))
    final_phase_start = initial_phase_end + final_checkpoint_interval

    final_phase_steps = list(
        range(final_phase_start, total_steps + 1, final_checkpoint_interval)
    )
    checkpoint_steps.extend(final_phase_steps)

    # Ensure final step is always included
    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)

    # Sort and remove any duplicates from the checkpoints
    checkpoint_steps = sorted(set(checkpoint_steps))

    return checkpoint_steps

def log_checkpoint_schedule(total_steps: int, steps: list[int]) -> None:
    """
    Log a summary of the checkpoint schedule.
    
    :param total_steps: Total number of training steps.
    :param steps: List of checkpoint steps.
    """
    print(f"Checkpoint schedule ({len(steps)} checkpoints over {total_steps:,} steps):")
    for i, step in enumerate(steps):
        pct = (step / total_steps) * 100
        print(f"  Checkpoint {i + 1:>2}: step {step:>7,} ({pct:.0f}%)")

def main() -> None:
    """Main entry point for verifying the checkpoint schedule."""
    # Verify schedule for encoder-only models (600k steps)
    steps = compute_checkpoint_steps(total_steps=600_000)
    log_checkpoint_schedule(600_000, steps)

if __name__ == "__main__":
    main()