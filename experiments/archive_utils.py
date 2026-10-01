"""
Utility to automatically archive previous benchmark outputs before new runs.
Guarantees compliance with .agents/rules/results_archiving.md.
"""

import os
import shutil
import time
from typing import List


def archive_previous_results(
    experiment_name: str,
    target_filenames: List[str],
    save_dir: str = "results",
    archive_root: str = "previous_results",
) -> str:
    """
    Checks if previous result files exist in save_dir. If so, moves them to a
    dedicated, timestamped subfolder in previous_results/<experiment_name>_<timestamp>/.

    Args:
        experiment_name: Descriptive name for the experiment run.
        target_filenames: List of filenames (plots, json files) to check and archive.
        save_dir: Directory where active results are output (default 'results').
        archive_root: Root archive directory (default 'previous_results').

    Returns:
        Path to archive directory if files were archived, else None.
    """
    existing = [f for f in target_filenames if os.path.exists(os.path.join(save_dir, f))]
    if not existing:
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(archive_root, f"{experiment_name}_{timestamp}")
    os.makedirs(dest_dir, exist_ok=True)

    for f in existing:
        src = os.path.join(save_dir, f)
        dst = os.path.join(dest_dir, f)
        shutil.move(src, dst)
        print(f"[Archived] {src} -> {dst}")

    return dest_dir
