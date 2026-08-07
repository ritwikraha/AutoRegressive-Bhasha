"""Validate teacher targets and write Parquet plus an audit summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import Dataset

try:
    from data.common import read_jsonl
except ModuleNotFoundError:
    from common import read_jsonl  # type: ignore


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(record["id"]) for record in records]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise ValueError(f"Found {duplicate_ids} duplicate teacher-target IDs.")

    teachers = [record.get("teacher") for record in records]
    if any(not isinstance(teacher, dict) for teacher in teachers):
        raise ValueError("Every record must contain a teacher metadata object.")

    requested = sum(int(teacher["samples_requested"]) for teacher in teachers)
    valid = sum(int(teacher["valid_samples"]) for teacher in teachers)
    agreements = [float(teacher["agreement_confidence"]) for teacher in teachers]
    correct = sum(bool(teacher["is_modal_answer_correct"]) for teacher in teachers)
    sources = Counter(str(record["source"]) for record in records)
    models = Counter(str(teacher["model"]) for teacher in teachers)

    return {
        "records": len(records),
        "unique_ids": len(set(ids)),
        "sources": dict(sorted(sources.items())),
        "teacher_models": dict(sorted(models.items())),
        "samples_requested": requested,
        "valid_samples": valid,
        "sample_parse_rate": valid / requested if requested else 0.0,
        "modal_answer_accuracy": correct / len(records) if records else 0.0,
        "mean_agreement_confidence": mean(agreements) if agreements else 0.0,
        "unanimous_records": sum(agreement == 1.0 for agreement in agreements),
        "unanimous_rate": (
            sum(agreement == 1.0 for agreement in agreements) / len(records) if records else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--release-manifest", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    parquet_path = Path(args.parquet) if args.parquet else input_path.with_suffix(".parquet")
    summary_path = (
        Path(args.summary) if args.summary else input_path.with_name(f"{input_path.stem}_summary.json")
    )
    records = list(read_jsonl(input_path))
    summary = summarize(records)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(parquet_path))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.release_manifest:
        manifest_path = Path(args.release_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["teacher_targets"] = {
            "status": "generated",
            "file": str(input_path),
            **summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
