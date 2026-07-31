"""
Aggregate fine-tuning results across multiple seeds.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Evaluation metric for each task
TASK_METRICS = {
    "ner": "f1",
    "pos": "f1",
    "ntc": "f1"
}

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate fine-tuning results across seeds."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing fine-tuning results."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_METRICS.keys()),
        help="Task to aggregate."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File path to save the aggregated results JSON to."
    )
    return parser.parse_args()

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a checkpoint directory name."""
    return int(path.name.split("-")[1])

def load_seed_results(results_dir: Path, task: str) -> dict[int, list[dict]]:
    """
    Load seed results for a given task, grouped by checkpoint step.

    :param results_dir: Root directory containing checkpoint result folders.
    :param task: Name of the evaluation task.
    :return: Mapping from checkpoint training step to a list of per-seed result dictionaries.
    """
    results_by_step = {}

    # Iterate through checkpoints in ascending step order
    for step_dir in sorted(results_dir.glob("step-*"), key=checkpoint_step):
        task_dir = step_dir / task
        if not task_dir.exists():
            continue

        seed_results = []

        # Load results from every available random seed
        for seed_dir in sorted(task_dir.glob("seed-*")):
            results_path = seed_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    seed_results.append(json.load(f))
            else:
                logger.warning(f"Missing results.json in {seed_dir}, skipping.")

        # Only keep checkpoints with at least one valid seed result
        if seed_results:
            results_by_step[checkpoint_step(step_dir)] = seed_results
        else:
            logger.warning(f"No seed results found for {step_dir.name}/{task}, skipping checkpoint.")

    logger.info(f"Loaded results for task '{task}' from {len(results_by_step)} checkpoints.")
    return results_by_step

def sample_std(values: list[float]) -> float:
    """
    Compute the sample standard deviation across seeds.

    :param values: Metric values, one per seed.
    :return: Sample standard deviation, or 0.0 for fewer than two values.
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))

def aggregate_step_results(seed_results: list[dict], overall_metric_key: str) -> dict:
    """
    Compute mean and standard deviation across seeds for one checkpoint.

    :param seed_results: List of results dicts, one per seed, for a single checkpoint.
    :param overall_metric_key: Key of the overall metric.
    :return: Dict with aggregated overall and per-class mean and standard deviation, plus seed count.
    """
    # Collect overall metric values across seeds
    overall_values = []

    for result in seed_results:
        test_metrics = result["test_metrics"]
        if overall_metric_key in test_metrics:
            overall_values.append(test_metrics[overall_metric_key])

    # Collect per-class F1 values across seeds.
    per_class_values = {}

    for result in seed_results:
        per_class_f1 = result["test_metrics"].get("eval_per_class_f1", {})
        for class_name, score in per_class_f1.items():
            per_class_values.setdefault(class_name, []).append(score)

    return {
        "num_seeds": len(seed_results),
        "seeds": [result.get("seed") for result in seed_results],
        "overall_mean": float(np.mean(overall_values)) if overall_values else None,
        "overall_std": sample_std(overall_values) if overall_values else None,
        "per_class_mean": {
            class_name: float(np.mean(scores))
            for class_name, scores in per_class_values.items()
        },
        "per_class_std": {
            class_name: sample_std(scores)
            for class_name, scores in per_class_values.items()
        }
    }

def main() -> None:
    """Main entry point for aggregating fine-tuning results across seeds."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    results_by_step = load_seed_results(results_dir, args.task)

    if not results_by_step:
        logger.error(f"No results found for task '{args.task}' in {results_dir}")
        return

    # Aggregate metrics across seeds for each checkpoint
    overall_metric_key = f"eval_{TASK_METRICS[args.task]}"
    aggregated = {
        step: aggregate_step_results(seed_results, overall_metric_key)
        for step, seed_results in results_by_step.items()
    }

    # Write aggregated results to disk
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(
        f"Aggregated {len(aggregated)} checkpoints across seeds, saved to {output_path}"
    )

if __name__ == "__main__":
    main()