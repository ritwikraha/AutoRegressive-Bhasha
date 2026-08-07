# %% [markdown]
# Curate normalized multiple-choice records before teacher sampling or training.
#
# The builder keeps source adapters intentionally thin. This stage applies
# auditable quality filters, deterministic source balancing, and manifest
# reporting so downstream Tinker runs spend tokens on clean examples.

# %%
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from datasets import Dataset

try:
    from data.common import (
        OPTION_LABELS,
        format_question_prompt,
        normalize_text,
        read_jsonl,
        stable_id,
        write_jsonl,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore
        OPTION_LABELS,
        format_question_prompt,
        normalize_text,
        read_jsonl,
        stable_id,
        write_jsonl,
    )

DEFAULT_SPLITS = ("train", "validation", "test")


# %%
@dataclass(frozen=True)
class CurationConfig:
    min_choices: int = 2
    max_choices: int = 6
    min_question_chars: int = 8
    max_question_chars: int = 2500
    max_choice_chars: int = 600
    max_prompt_chars: int = 8000
    drop_duplicate_choices: bool = True
    drop_duplicate_questions_across_splits: bool = True
    balance_sources: bool = False
    split_limits: Mapping[str, int | None] = field(default_factory=dict)
    source_limits: Mapping[str, Mapping[str, int | None] | int | None] = field(default_factory=dict)
    blocked_question_patterns: Sequence[str] = field(default_factory=list)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def curation_config_from_mapping(config: Mapping[str, Any]) -> CurationConfig:
    raw = dict(config.get("curation") or {})
    return CurationConfig(
        min_choices=int(raw.get("min_choices", CurationConfig.min_choices)),
        max_choices=int(raw.get("max_choices", CurationConfig.max_choices)),
        min_question_chars=int(raw.get("min_question_chars", CurationConfig.min_question_chars)),
        max_question_chars=int(raw.get("max_question_chars", CurationConfig.max_question_chars)),
        max_choice_chars=int(raw.get("max_choice_chars", CurationConfig.max_choice_chars)),
        max_prompt_chars=int(raw.get("max_prompt_chars", CurationConfig.max_prompt_chars)),
        drop_duplicate_choices=bool(
            raw.get("drop_duplicate_choices", CurationConfig.drop_duplicate_choices)
        ),
        drop_duplicate_questions_across_splits=bool(
            raw.get(
                "drop_duplicate_questions_across_splits",
                CurationConfig.drop_duplicate_questions_across_splits,
            )
        ),
        balance_sources=bool(raw.get("balance_sources", CurationConfig.balance_sources)),
        split_limits=dict(raw.get("split_limits") or {}),
        source_limits=dict(raw.get("source_limits") or {}),
        blocked_question_patterns=list(raw.get("blocked_question_patterns") or []),
    )


# %%
def question_key(record: Mapping[str, Any]) -> str:
    return stable_id(
        normalize_text(record.get("question", "")).lower(),
        [normalize_text(choice).lower() for choice in record.get("choices", [])],
        length=32,
    )


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    choices = [normalize_text(choice) for choice in output.get("choices", [])]
    output["question"] = normalize_text(output.get("question", ""))
    output["choices"] = choices
    output["answer_index"] = int(output["answer_index"])
    output["answer_label"] = OPTION_LABELS[output["answer_index"]]
    output["prompt"] = format_question_prompt(output["question"], choices)
    output["metadata"] = dict(output.get("metadata") or {})
    return output


def reject_reasons(record: Mapping[str, Any], config: CurationConfig) -> list[str]:
    reasons: list[str] = []
    required_fields = ("id", "source", "source_split", "question", "choices", "answer_index")
    missing = [field for field in required_fields if field not in record]
    if missing:
        return [f"missing_{field}" for field in missing]

    question = normalize_text(record.get("question", ""))
    choices = [normalize_text(choice) for choice in record.get("choices", [])]
    answer_index = record.get("answer_index")

    if len(question) < config.min_question_chars:
        reasons.append("question_too_short")
    if len(question) > config.max_question_chars:
        reasons.append("question_too_long")
    if not config.min_choices <= len(choices) <= config.max_choices:
        reasons.append("invalid_choice_count")
    if any(not choice for choice in choices):
        reasons.append("empty_choice")
    if any(len(choice) > config.max_choice_chars for choice in choices):
        reasons.append("choice_too_long")
    if config.drop_duplicate_choices and len({choice.lower() for choice in choices}) != len(choices):
        reasons.append("duplicate_choice_text")
    if not isinstance(answer_index, int) or not 0 <= answer_index < len(choices):
        reasons.append("invalid_answer_index")
    if len(str(record.get("prompt") or format_question_prompt(question, choices))) > config.max_prompt_chars:
        reasons.append("prompt_too_long")

    lowered_question = question.lower()
    for pattern in config.blocked_question_patterns:
        if pattern.lower() in lowered_question:
            reasons.append("blocked_question_pattern")
            break

    return reasons


def source_limit_for_split(
    source_limits: Mapping[str, Mapping[str, int | None] | int | None],
    *,
    split_name: str,
    source: str,
) -> int | None:
    raw = source_limits.get(source)
    if isinstance(raw, Mapping):
        value = raw.get(split_name)
    else:
        value = raw
    return None if value is None else int(value)


def deterministic_take(records: Sequence[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit is not None:
        items = items[:limit]
    return items


def apply_source_limits(
    records: Sequence[dict[str, Any]],
    *,
    split_name: str,
    config: CurationConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source"])].append(record)

    retained: list[dict[str, Any]] = []
    dropped = Counter()
    for source, source_records in sorted(by_source.items()):
        limit = source_limit_for_split(config.source_limits, split_name=split_name, source=source)
        selected = deterministic_take(
            source_records,
            limit,
            int(stable_id(seed, split_name, source, "source_limit", length=12), 16),
        )
        retained.extend(selected)
        dropped["source_limit"] += max(0, len(source_records) - len(selected))

    return retained, dropped


def apply_split_limit(
    records: Sequence[dict[str, Any]],
    *,
    split_name: str,
    config: CurationConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    limit_value = config.split_limits.get(split_name)
    split_limit = None if limit_value is None else int(limit_value)
    if split_limit is None or split_limit >= len(records):
        return list(records), Counter()

    if not config.balance_sources:
        selected = deterministic_take(
            records,
            split_limit,
            int(stable_id(seed, split_name, "split_limit", length=12), 16),
        )
        return selected, Counter({"split_limit": len(records) - len(selected)})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source"])].append(record)

    shuffled_groups: dict[str, list[dict[str, Any]]] = {}
    for source, source_records in grouped.items():
        shuffled_groups[source] = deterministic_take(
            source_records,
            None,
            int(stable_id(seed, split_name, source, "balanced_limit", length=12), 16),
        )

    selected: list[dict[str, Any]] = []
    source_names = sorted(shuffled_groups)
    cursor = 0
    while len(selected) < split_limit and any(shuffled_groups.values()):
        source = source_names[cursor % len(source_names)]
        if shuffled_groups[source]:
            selected.append(shuffled_groups[source].pop())
        cursor += 1

    return selected, Counter({"split_limit": len(records) - len(selected)})


# %%
def curate_splits(
    splits: Mapping[str, Iterable[dict[str, Any]]],
    *,
    config: CurationConfig,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    curated: dict[str, list[dict[str, Any]]] = {}
    global_question_keys: set[str] = set()
    manifest: dict[str, Any] = {
        "config": {
            "min_choices": config.min_choices,
            "max_choices": config.max_choices,
            "min_question_chars": config.min_question_chars,
            "max_question_chars": config.max_question_chars,
            "max_choice_chars": config.max_choice_chars,
            "max_prompt_chars": config.max_prompt_chars,
            "drop_duplicate_choices": config.drop_duplicate_choices,
            "drop_duplicate_questions_across_splits": config.drop_duplicate_questions_across_splits,
            "balance_sources": config.balance_sources,
            "split_limits": dict(config.split_limits),
            "source_limits": dict(config.source_limits),
            "blocked_question_patterns": list(config.blocked_question_patterns),
        },
        "splits": {},
        "dropped": {},
    }

    for split_name in DEFAULT_SPLITS:
        input_records = list(splits.get(split_name, []))
        split_drops = Counter()
        valid_records: list[dict[str, Any]] = []

        for record in input_records:
            reasons = reject_reasons(record, config)
            if reasons:
                split_drops.update(reasons)
                continue

            normalized = normalize_record(record)
            valid_records.append(normalized)

        valid_records, source_drops = apply_source_limits(
            valid_records,
            split_name=split_name,
            config=config,
            seed=seed,
        )
        split_drops.update(source_drops)

        valid_records, limit_drops = apply_split_limit(
            valid_records,
            split_name=split_name,
            config=config,
            seed=seed,
        )
        split_drops.update(limit_drops)

        deduped_records: list[dict[str, Any]] = []
        for record in valid_records:
            key = question_key(record)
            if config.drop_duplicate_questions_across_splits and key in global_question_keys:
                split_drops["cross_split_duplicate"] += 1
                continue
            global_question_keys.add(key)
            deduped_records.append(record)

        final_rows = deterministic_take(
            deduped_records,
            None,
            int(stable_id(seed, split_name, "final_shuffle", length=12), 16),
        )
        for record in final_rows:
            record["split"] = split_name

        curated[split_name] = final_rows
        manifest["splits"][split_name] = {
            "input": len(input_records),
            "retained": len(final_rows),
            "sources": dict(sorted(Counter(row["source"] for row in final_rows).items())),
        }
        manifest["dropped"][split_name] = {
            reason: count for reason, count in sorted(split_drops.items()) if count
        }

    return curated, manifest


def load_input_splits(input_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_dir = Path(input_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in DEFAULT_SPLITS:
        path = input_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing input split: {path}")
        splits[split_name] = list(read_jsonl(path))
    return splits


def save_curated_splits(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        write_jsonl(records, output_dir / f"{split_name}.jsonl")
        Dataset.from_list(list(records)).to_parquet(str(output_dir / f"{split_name}.parquet"))

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Curate normalized MCQ dataset splits.")
    parser.add_argument("--input-dir", default="artifacts/base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--balance-sources", action="store_true")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    curation_config = curation_config_from_mapping(raw_config)
    split_limits = dict(curation_config.split_limits)
    for split_name, override in (
        ("train", args.train_limit),
        ("validation", args.validation_limit),
        ("test", args.test_limit),
    ):
        if override is not None:
            split_limits[split_name] = override

    curation_config = replace(
        curation_config,
        split_limits=split_limits,
        balance_sources=curation_config.balance_sources or args.balance_sources,
    )

    output_dir = args.output_dir or raw_config.get("curation", {}).get("output_dir", "artifacts/curated")
    input_splits = load_input_splits(args.input_dir)
    curated, manifest = curate_splits(
        input_splits,
        config=curation_config,
        seed=int(raw_config.get("seed", 42)),
    )
    save_curated_splits(curated, manifest, output_dir)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote curated splits to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
