"""Sample a base model or Tinker checkpoint and optionally score every option."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.renderers import get_renderer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.common import OPTION_LABELS
from evaluation.parsing import parse_completion
from training.common import ConfidenceFormat, build_system_prompt, require_tinker_key


@dataclass(frozen=True)
class SamplingConfig:
    output_path: str
    repo_id: str = "ritwikraha/calibrate-qwen-curated"
    dataset_config: str = "calibrated_mcq"
    split: str = "test"
    model_name: str = "Qwen/Qwen3.5-4B"
    checkpoint_path: str | None = None
    renderer_name: str = "qwen3_5_disable_thinking"
    confidence_format: ConfidenceFormat = "numeric"
    temperature: float = 0.2
    max_tokens: int = 192
    concurrency: int = 8
    limit: int | None = None
    score_options: bool = False
    resume: bool = True


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {label: math.exp(score - maximum) for label, score in scores.items()}
    total = sum(weights.values())
    return {label: value / total for label, value in weights.items()}


async def _score_options(
    *,
    sampling_client: tinker.SamplingClient,
    renderer: Any,
    tokenizer: Any,
    record: dict[str, Any],
) -> dict[str, float]:
    messages = [
        {
            "role": "system",
            "content": "Return exactly one uppercase option label and no other text.",
        },
        {"role": "user", "content": record["prompt"]},
    ]
    prompt = renderer.build_generation_prompt(messages)
    prompt_tokens = prompt.to_ints()
    labels = OPTION_LABELS[: len(record["choices"])]

    async def score(label: str) -> tuple[str, float]:
        candidate_tokens = tokenizer.encode(label)
        sequence = types.ModelInput.from_ints(prompt_tokens + candidate_tokens)
        logprobs = await sampling_client.compute_logprobs_async(sequence)
        candidate_logprobs = logprobs[-len(candidate_tokens) :]
        return label, sum(float(value) for value in candidate_logprobs if value is not None)

    scores = dict(await asyncio.gather(*(score(label) for label in labels)))
    return _softmax(scores)


async def sample_dataset(config: SamplingConfig) -> None:
    require_tinker_key()
    dataset = load_dataset(
        config.repo_id,
        config.dataset_config,
        split=config.split,
        token=os.environ.get("HF_TOKEN"),
    )
    if config.limit is not None:
        dataset = dataset.select(range(min(config.limit, len(dataset))))
    records = dataset.to_list()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if config.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            completed_ids = {json.loads(line)["id"] for line in handle if line.strip()}

    service_client = tinker.ServiceClient()
    if config.checkpoint_path:
        sampling_client = await service_client.create_sampling_client_async(
            model_path=config.checkpoint_path
        )
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=config.model_name
        )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(config.renderer_name, tokenizer)
    params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()

    async def process(record: dict[str, Any]) -> None:
        if record["id"] in completed_ids:
            return
        labels = set(OPTION_LABELS[: len(record["choices"])])
        messages = [
            {"role": "system", "content": build_system_prompt(config.confidence_format)},
            {"role": "user", "content": record["prompt"]},
        ]
        model_input = renderer.build_generation_prompt(messages)
        async with semaphore:
            response = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )
            text = tokenizer.decode(response.sequences[0].tokens)
            prediction = parse_completion(
                text,
                labels,
                require_confidence=config.confidence_format != "implicit",
            )
            option_probabilities = (
                await _score_options(
                    sampling_client=sampling_client,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    record=record,
                )
                if config.score_options
                else None
            )

        output = dict(record)
        output["model"] = {
            "base_model": config.model_name,
            "checkpoint_path": config.checkpoint_path,
            "renderer": config.renderer_name,
        }
        output["prediction"] = prediction
        if option_probabilities is not None:
            output["option_probabilities"] = option_probabilities
            answer = prediction.get("answer")
            output["answer_probability"] = option_probabilities.get(answer) if answer else None

        async with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    pending = [record for record in records if record["id"] not in completed_ids]
    tasks = [asyncio.create_task(process(record)) for record in pending]
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        await task
        if index % 25 == 0 or index == len(tasks):
            print(f"Completed {index}/{len(tasks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repo-id", default="ritwikraha/calibrate-qwen-curated")
    parser.add_argument("--dataset-config", default="calibrated_mcq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--renderer-name", default="qwen3_5_disable_thinking")
    parser.add_argument(
        "--confidence-format", choices=["numeric", "bucket", "implicit"], default="numeric"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-options", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["resume"] = not values.pop("no_resume")
    asyncio.run(sample_dataset(SamplingConfig(**values)))


if __name__ == "__main__":
    main()
