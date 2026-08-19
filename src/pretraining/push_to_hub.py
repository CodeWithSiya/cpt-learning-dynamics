"""Push CPT checkpoints to the Hugging Face Hub, one branch per training step."""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

def checkpoint_step(path: Path) -> int:
    """Extract the training step number from a step-N checkpoint directory name."""
    match = re.match(r"step-(\d+)$", path.name)
    if not match:
        raise ValueError(f"Not a step checkpoint directory: {path}")
    return int(match.group(1))


def find_step_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """
    Find and sort all step-N checkpoint directories.

    :param checkpoint_dir: Directory containing step-N subfolders.
    :return: Checkpoint directories sorted by step, ascending.
    """
    checkpoints = [
        path for path in checkpoint_dir.iterdir()
        if path.is_dir() and re.match(r"step-\d+$", path.name)
    ]
    return sorted(checkpoints, key=checkpoint_step)

def is_branch_complete(api: HfApi, repo_id: str, branch: str, local_path: Path) -> bool:
    """
    Check whether a branch already holds a complete copy of a local checkpoint.

    A branch can exist but be empty, since create_branch runs before upload_folder
    commits, so an interrupted upload leaves the branch pointing at an empty main.

    :param api: Authenticated Hub API client.
    :param repo_id: Target Hub repo.
    :param branch: Branch to inspect.
    :param local_path: Local checkpoint directory the branch should mirror.
    :return: True if every local file is present on the branch.
    """
    try:
        remote_files = set(api.list_repo_files(repo_id, revision=branch))
    except Exception:
        return False

    local_files = {
        str(path.relative_to(local_path))
        for path in local_path.rglob("*")
        if path.is_file()
    }
    return local_files.issubset(remote_files)

def push_checkpoints(checkpoint_dir: Path, repo_id: str, final_step: int | None, private: bool) -> None:
    """
    Push every step-N checkpoint to its own branch of a single Hub repo.

    The highest-step checkpoint (or final_step, if given) is additionally pushed to main.
    Already-uploaded branches are skipped so the job can be safely resumed.

    :param checkpoint_dir: Directory containing step-N subfolders.
    :param repo_id: Target Hub repo, e.g. "username/xlmr-large-cpt-xho".
    :param final_step: Step to treat as the final model, pushed to main. Defaults to the highest step found.
    :param private: Whether to create the repo as private.
    """
    checkpoints = find_step_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"No step-N checkpoints found under {checkpoint_dir}")
        return

    if final_step is None:
        final_step = checkpoint_step(checkpoints[-1])

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)

    for path in checkpoints:
        step = checkpoint_step(path)
        branch = f"step-{step}"

        targets = [branch]
        if step == final_step:
            targets.append("main")

        for target in targets:
            if is_branch_complete(api, repo_id, target, path):
                print(f"Skipping {target}: already complete on the Hub", flush=True)
                continue

            if target != "main":
                api.create_branch(repo_id, branch=target, exist_ok=True)

            print(f"Uploading step {step} to {repo_id}@{target} ...", flush=True)
            api.upload_folder(
                folder_path=str(path),
                repo_id=repo_id,
                revision=target,
                commit_message=f"Upload checkpoint at step {step}",
                delete_patterns="*",
            )
            print(f"Finished {repo_id}@{target}", flush=True)

def main() -> None:
    """Parse command-line arguments and push checkpoints to the Hub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing step-N checkpoint subfolders (i.e. <output_dir>/checkpoints).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help='Target Hub repo id, e.g. "username/xlmr-large-cpt-xho".',
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=None,
        help="Step to push to the main branch. Defaults to the highest step found.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it doesn't already exist.",
    )
    args = parser.parse_args()

    push_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        final_step=args.final_step,
        private=args.private,
    )

if __name__ == "__main__":
    main()
    