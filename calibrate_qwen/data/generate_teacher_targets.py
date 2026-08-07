# %% [markdown]
# Generate teacher targets through Tinker sampling.
#
# Confidence is based on repeated teacher answer agreement, rather than asking
# the teacher to invent a percentage. Each output record retains all sampled
# answers and parsing diagnostics for auditability.

# %%
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer

try:
    from data.common import OPTION_LABELS, read_jsonl, write_jsonl
except ModuleNotFoundError:
    from common import OPTION_LABELS, read_jsonl, write_jsonl  # type: ignore

SYSTEM_PROMPT = (
    "You answer multiple-choice questions. Return only valid JSON with exactly "
    "two keys: answer and justification. answer must be one uppercase option "
    "label from the prompt. justification must be concise and must not mention "
    "confidence, uncertainty, or hidden reasoning."
)


# %%
def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object without relying on greedy regex."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def parse_teacher_completion(text: str, allowed_labels: set[str]) -> dict[str, Any]:
    parsed = extract_first_json_object(text)
    answer = ""
    justification = ""
    parse_method = "invalid"

    if parsed is not None:
        answer = str(parsed.get("answer", "")).strip().upper()
        justification = str(parsed.get("justification", "")).strip()
        parse_method = "json"

    if answer not in allowed_labels:
        # Conservative fallback: accept an explicit answer field or standalone label.
        match = re.search(r"(?i)\banswer\s*[:=]\s*[\"']?([A-Z])\b", text)
        if not match:
            match = re.search(r"(?m)^\s*([A-Z])\s*[\).:]?\s*$", text)
        if match and match.group(1).upper() in allowed_labels:
            answer = match.group(1).upper()
            parse_method = "fallback_regex"

    return {
        "answer": answer if answer in allowed_labels else None,
        "justification": justification,
        "parse_method": parse_method,
        "raw_text": text,
    }


def make_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["prompt"]},
    ]


# %%
async def annotate_records(
    records: Iterable[dict[str, Any]],
    *,
    teacher_model: str,
    renderer_name: str,
    samples_per_question: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    output_path: str | Path,
    resume: bool = True,
) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise EnvironmentError("Set TINKER_API_KEY before running teacher generation.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if resume and output_path.exists():
        completed_ids = {row["id"] for row in read_jsonl(output_path)}
        print(f"Resume enabled: found {len(completed_ids)} completed records.")

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=teacher_model)
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def process_one(record: dict[str, Any]) -> None:
        if record["id"] in completed_ids:
            return
        messages = make_messages(record)
        model_input = renderer.build_generation_prompt(messages)

        async with semaphore:
            response = await sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=samples_per_question,
            )

        allowed_labels = set(OPTION_LABELS[: len(record["choices"])])
        samples: list[dict[str, Any]] = []
        for sequence in response.sequences:
            text = tokenizer.decode(sequence.tokens)
            samples.append(parse_teacher_completion(text, allowed_labels))

        valid_answers = [sample["answer"] for sample in samples if sample["answer"]]
        counts = Counter(valid_answers)
        modal_answer = counts.most_common(1)[0][0] if counts else None
        agreement = counts[modal_answer] / samples_per_question if modal_answer else 0.0
        representative = next(
            (sample for sample in samples if sample["answer"] == modal_answer),
            {"justification": ""},
        )

        output = dict(record)
        output["teacher"] = {
            "model": teacher_model,
            "samples_requested": samples_per_question,
            "valid_samples": len(valid_answers),
            "answer_counts": dict(sorted(counts.items())),
            "modal_answer": modal_answer,
            "agreement_confidence": agreement,
            "is_modal_answer_correct": modal_answer == record["answer_label"],
            "representative_justification": representative.get("justification", ""),
            "samples": samples,
        }

        async with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    pending = [record for record in records if record["id"] not in completed_ids]
    print(f"Generating targets for {len(pending)} records with concurrency={concurrency}.")
    tasks = [asyncio.create_task(process_one(record)) for record in pending]
    for completed_index, task in enumerate(asyncio.as_completed(tasks), start=1):
        await task
        if completed_index % 25 == 0 or completed_index == len(tasks):
            print(f"Completed {completed_index}/{len(tasks)}")


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated-sampling teacher targets via Tinker.")
    parser.add_argument("--input", default="artifacts/base/train.jsonl")
    parser.add_argument("--output", default="artifacts/teacher/train_teacher.jsonl")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--renderer", default="qwen3_5")
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    asyncio.run(
        annotate_records(
            records,
            teacher_model=args.teacher_model,
            renderer_name=args.renderer,
            samples_per_question=args.samples_per_question,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            output_path=args.output,
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
