"""Convert the published MCQ data into Tinker conversation JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from datasets import load_dataset
from huggingface_hub import HfApi

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import (
    ConfidenceFormat,
    build_system_prompt,
    format_target,
    optional_hf_token,
    teacher_fields,
)

Variant = Literal["hard_label", "teacher"]


def assert_hub_dataset_access(repo_id: str, token: str | None) -> None:
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is empty. Add it to Colab Secrets and enable notebook access."
        )
    try:
        HfApi(token=token).repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as error:
        raise RuntimeError(
            f"Cannot access the private dataset {repo_id}. Refresh the HF_TOKEN Colab secret, "
            "enable notebook access for that secret, and rerun the authentication cell."
        ) from error


def load_training_records(
    *,
    repo_id: str,
    variant: Variant,
    limit: int | None = None,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    token = (hf_token or optional_hf_token() or "").strip()
    assert_hub_dataset_access(repo_id, token)
    config_name = "phase1_2000" if variant == "hard_label" else "teacher_phase1"
    dataset = load_dataset(
        repo_id,
        config_name,
        split="train",
        token=token,
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.to_list()


def record_to_conversation(
    record: dict[str, Any],
    *,
    variant: Variant,
    confidence_format: ConfidenceFormat,
    abstention_threshold: float,
) -> dict[str, Any]:
    if variant == "hard_label":
        answer = str(record["answer_label"])
        confidence = 1.0
        justification = f"Option {answer} is the labeled answer."
    else:
        answer, confidence, justification = teacher_fields(record)

    target = format_target(
        answer=answer,
        confidence=confidence,
        justification=justification,
        abstention_threshold=abstention_threshold,
        confidence_format=confidence_format,
    )
    return {
        "id": record["id"],
        "source": record["source"],
        "messages": [
            {"role": "system", "content": build_system_prompt(confidence_format)},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": target.to_json()},
        ],
    }


def prepare_training_file(
    *,
    output_path: str | Path,
    repo_id: str = "ritwikraha/calibrate-qwen-curated",
    variant: Variant = "teacher",
    confidence_format: ConfidenceFormat = "numeric",
    abstention_threshold: float = 0.6,
    limit: int | None = None,
    hf_token: str | None = None,
    records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    loaded = (
        list(records)
        if records is not None
        else load_training_records(
            repo_id=repo_id,
            variant=variant,
            limit=limit,
            hf_token=hf_token,
        )
    )
    conversations: list[dict[str, Any]] = []
    skipped = Counter()
    for record in loaded:
        try:
            conversations.append(
                record_to_conversation(
                    record,
                    variant=variant,
                    confidence_format=confidence_format,
                    abstention_threshold=abstention_threshold,
                )
            )
        except ValueError as error:
            skipped[str(error).split(".", 1)[0]] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    manifest = {
        "repo_id": repo_id,
        "variant": variant,
        "confidence_format": confidence_format,
        "abstention_threshold": abstention_threshold,
        "input_records": len(loaded),
        "written_records": len(conversations),
        "sources": dict(sorted(Counter(row["source"] for row in conversations).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="ritwikraha/calibrate-qwen-curated")
    parser.add_argument("--variant", choices=["hard_label", "teacher"], default="teacher")
    parser.add_argument(
        "--confidence-format", choices=["numeric", "bucket", "implicit"], default="numeric"
    )
    parser.add_argument("--abstention-threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = prepare_training_file(
        output_path=args.output,
        repo_id=args.repo_id,
        variant=args.variant,
        confidence_format=args.confidence_format,
        abstention_threshold=args.abstention_threshold,
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
