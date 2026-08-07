# %% [markdown]
# CalibrateQwen dataset builder
#
# This file deliberately uses notebook-style `# %%` cells. In Colab you can:
# 1. upload/clone the repository,
# 2. import the functions below, or
# 3. paste individual cells into a notebook.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from datasets import Dataset, load_dataset

try:
    from data.common import (
        canonical_record,
        deduplicate_records,
        deterministic_sample,
        label_to_index,
        split_records,
        write_jsonl,
    )
except ModuleNotFoundError:  # Allows `python data/build_dataset.py` from repo root.
    from common import (  # type: ignore
        canonical_record,
        deduplicate_records,
        deterministic_sample,
        label_to_index,
        split_records,
        write_jsonl,
    )

Adapter = Callable[[dict[str, Any], str, int], dict[str, Any]]


# %%
def adapt_arc(row: dict[str, Any], source_split: str, row_index: int) -> dict[str, Any]:
    choice_block = row["choices"]
    labels = list(choice_block["label"])
    choices = list(choice_block["text"])
    answer_index = label_to_index(row["answerKey"], labels)
    return canonical_record(
        source="arc_challenge",
        source_split=source_split,
        source_id=row.get("id", row_index),
        question=row["question"],
        choices=choices,
        answer_index=answer_index,
        subject="science_reasoning",
        metadata={"original_labels": labels},
    )


def adapt_openbookqa(row: dict[str, Any], source_split: str, row_index: int) -> dict[str, Any]:
    choice_block = row["choices"]
    labels = list(choice_block["label"])
    choices = list(choice_block["text"])
    answer_index = label_to_index(row["answerKey"], labels)
    return canonical_record(
        source="openbookqa",
        source_split=source_split,
        source_id=row.get("id", row_index),
        question=row["question_stem"],
        choices=choices,
        answer_index=answer_index,
        subject="open_book_science",
        metadata={"original_labels": labels},
    )


def adapt_commonsenseqa(row: dict[str, Any], source_split: str, row_index: int) -> dict[str, Any]:
    choice_block = row["choices"]
    labels = list(choice_block["label"])
    choices = list(choice_block["text"])
    answer_index = label_to_index(row["answerKey"], labels)
    return canonical_record(
        source="commonsenseqa",
        source_split=source_split,
        source_id=row.get("id", row_index),
        question=row["question"],
        choices=choices,
        answer_index=answer_index,
        subject="commonsense_reasoning",
        metadata={"original_labels": labels},
    )


def adapt_mmlu(row: dict[str, Any], source_split: str, row_index: int) -> dict[str, Any]:
    choices = list(row["choices"])
    answer_index = label_to_index(row["answer"])
    subject = row.get("subject") or "mmlu"
    return canonical_record(
        source="mmlu",
        source_split=source_split,
        source_id=f"{subject}:{row_index}",
        question=row["question"],
        choices=choices,
        answer_index=answer_index,
        subject=subject,
    )


ADAPTERS: dict[str, Adapter] = {
    "arc_challenge": adapt_arc,
    "openbookqa": adapt_openbookqa,
    "commonsenseqa": adapt_commonsenseqa,
    "mmlu": adapt_mmlu,
}


# %%
def load_one_source(source_config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Load and normalize one source declared in the YAML config."""
    name = source_config["name"]
    if name not in ADAPTERS:
        raise KeyError(f"No adapter registered for source {name!r}.")

    dataset_name = source_config["dataset"]
    subset = source_config.get("subset")
    source_split = source_config.get("split", "train")
    max_examples = source_config.get("max_examples")
    trust_remote_code = bool(source_config.get("trust_remote_code", False))

    print(f"Loading {name}: dataset={dataset_name}, subset={subset}, split={source_split}")
    dataset: Dataset = load_dataset(
        dataset_name,
        subset,
        split=source_split,
        trust_remote_code=trust_remote_code,
    )

    rows: list[dict[str, Any]] = []
    adapter = ADAPTERS[name]
    failures = 0
    for index, row in enumerate(dataset):
        try:
            rows.append(adapter(dict(row), source_split, index))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            failures += 1
            if failures <= 3:
                print(f"Skipping malformed {name} row {index}: {exc}")

    rows = deterministic_sample(rows, max_examples, seed)
    print(f"  retained={len(rows)}, malformed={failures}")
    return rows


def build_dataset(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seed = int(config.get("seed", 42))
    records: list[dict[str, Any]] = []
    for source_config in config["sources"]:
        if source_config.get("enabled", True):
            records.extend(load_one_source(source_config, seed))

    records, duplicates_removed = deduplicate_records(records)
    print(f"Combined rows={len(records)}; exact duplicates removed={duplicates_removed}")

    split_config = config.get("splits", {})
    return split_records(
        records,
        train_fraction=float(split_config.get("train", 0.8)),
        validation_fraction=float(split_config.get("validation", 0.1)),
        seed=seed,
    )


def save_dataset(splits: dict[str, list[dict[str, Any]]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"splits": {}, "sources": {}}
    for split_name, records in splits.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        parquet_path = output_dir / f"{split_name}.parquet"
        write_jsonl(records, jsonl_path)
        Dataset.from_list(records).to_parquet(str(parquet_path))
        manifest["splits"][split_name] = len(records)
        for record in records:
            source = record["source"]
            manifest["sources"].setdefault(source, {}).setdefault(split_name, 0)
            manifest["sources"][source][split_name] += 1

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2))


# %%
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "sources" not in config:
        raise ValueError("Config must be a mapping containing a 'sources' list.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CalibrateQwen MCQ dataset.")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config.get("output_dir", "artifacts/base")
    splits = build_dataset(config)
    save_dataset(splits, output_dir)


if __name__ == "__main__":
    main()
