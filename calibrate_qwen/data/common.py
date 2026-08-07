"""Shared helpers for the CalibrateQwen dataset pipeline.

Designed for both:
  * normal execution: python data/build_dataset.py --config configs/data_config.yaml
  * Colab/notebook use: import functions from this file and call them directly.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def stable_id(*parts: Any, length: int = 16) -> str:
    """Create a deterministic ID from arbitrary JSON-serializable values."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_text(value: Any) -> str:
    """Normalize whitespace while preserving ordinary punctuation."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_choices(choices: Sequence[Any]) -> list[str]:
    cleaned = [normalize_text(choice) for choice in choices]
    return [choice for choice in cleaned if choice]


def label_to_index(label: Any, labels: Sequence[str] | None = None) -> int:
    """Convert a dataset-specific answer label to a zero-based choice index."""
    if isinstance(label, bool):
        raise ValueError("Boolean answer labels are not supported.")
    if isinstance(label, int):
        return label

    text = normalize_text(label)
    if labels:
        normalized_labels = [normalize_text(item) for item in labels]
        if text in normalized_labels:
            return normalized_labels.index(text)

    if text.isdigit():
        return int(text)
    upper = text.upper()
    if upper in OPTION_LABELS:
        return OPTION_LABELS.index(upper)
    raise ValueError(f"Cannot convert answer label {label!r} to an index.")


def index_to_label(index: int) -> str:
    if not 0 <= index < len(OPTION_LABELS):
        raise ValueError(f"Choice index {index} is outside the supported range.")
    return OPTION_LABELS[index]


def format_question_prompt(
    question: str,
    choices: Sequence[str],
    *,
    include_json_instruction: bool = True,
) -> str:
    lines = [normalize_text(question), ""]
    lines.extend(f"{index_to_label(i)}. {normalize_text(choice)}" for i, choice in enumerate(choices))
    if include_json_instruction:
        lines.extend(
            [
                "",
                "Return only valid JSON with this schema:",
                '{"answer":"A","justification":"brief reason"}',
                "The answer must be exactly one listed option label.",
            ]
        )
    return "\n".join(lines)


def canonical_record(
    *,
    source: str,
    source_split: str,
    source_id: Any,
    question: Any,
    choices: Sequence[Any],
    answer_index: int,
    subject: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_question = normalize_text(question)
    clean_choices = normalize_choices(choices)
    if len(clean_choices) < 2:
        raise ValueError("A multiple-choice example must contain at least two choices.")
    if not 0 <= answer_index < len(clean_choices):
        raise ValueError(
            f"answer_index={answer_index} is invalid for {len(clean_choices)} choices."
        )

    record_id = stable_id(source, source_split, source_id, clean_question, clean_choices)
    return {
        "id": record_id,
        "source": source,
        "source_split": source_split,
        "source_id": normalize_text(source_id),
        "subject": normalize_text(subject) or None,
        "question": clean_question,
        "choices": clean_choices,
        "answer_index": int(answer_index),
        "answer_label": index_to_label(answer_index),
        "prompt": format_question_prompt(clean_question, clean_choices),
        "metadata": dict(metadata or {}),
    }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def deterministic_sample(
    records: Sequence[dict[str, Any]],
    size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    items = list(records)
    if size is None or size >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, size)


def deduplicate_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate exact normalized question+choice combinations."""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for record in records:
        key = stable_id(record["question"], record["choices"], length=32)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(record)
    return output, removed


def split_records(
    records: Sequence[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split records while preserving source proportions approximately."""
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Invalid split fractions.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["source"], []).append(record)

    result = {"train": [], "validation": [], "test": []}
    for source, group in sorted(grouped.items()):
        rng = random.Random(stable_id(seed, source, length=12))
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_fraction)
        validation_end = train_end + int(n * validation_fraction)
        result["train"].extend(shuffled[:train_end])
        result["validation"].extend(shuffled[train_end:validation_end])
        result["test"].extend(shuffled[validation_end:])

    for split_name, split_rows in result.items():
        random.Random(stable_id(seed, split_name, length=12)).shuffle(split_rows)
        for row in split_rows:
            row["split"] = split_name
    return result
