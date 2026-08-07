"""Publish the curated release and optional teacher targets to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. user/name")
    parser.add_argument("--release-dir", default="artifacts/issue_release")
    parser.add_argument("--teacher-dir", default="artifacts/teacher")
    parser.add_argument("--dataset-card", default="DATASET_CARD.md")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo. Repos are private by default.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN before publishing.")

    release_dir = require_directory(Path(args.release_dir), "Release")
    teacher_dir = Path(args.teacher_dir)
    dataset_card = Path(args.dataset_card)
    if not dataset_card.is_file():
        raise FileNotFoundError(f"Dataset card does not exist: {dataset_card}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=release_dir,
        path_in_repo="data",
        allow_patterns=["*.jsonl", "*.parquet", "manifest.json"],
        commit_message="Upload curated CalibrateQwen release",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=dataset_card,
        path_in_repo="README.md",
        commit_message="Add dataset card",
    )

    teacher_files = list(teacher_dir.glob("*.jsonl")) if teacher_dir.is_dir() else []
    if teacher_files:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=teacher_dir,
            path_in_repo="teacher",
            allow_patterns=["*.jsonl", "*.parquet", "*.json"],
            commit_message="Upload Tinker teacher targets",
        )

    visibility = "public" if args.public else "private"
    print(f"Published {visibility} dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
