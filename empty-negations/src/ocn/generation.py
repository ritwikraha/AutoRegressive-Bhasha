from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import random
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    stage: str
    use_chat_template: bool = False
    loader_type: str = "causal_lm"


@dataclass(frozen=True)
class DecodingSpec:
    name: str
    temperature: float
    top_p: float = 0.95
    max_new_tokens: int = 180


DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("google/gemma-4-E2B", "gemma4", "base", False, "multimodal_lm"),
    ModelSpec("google/gemma-4-E2B-it", "gemma4", "instruct", True, "multimodal_lm"),
    ModelSpec("Qwen/Qwen3.5-2B-Base", "qwen3.5", "base", False, "multimodal_lm"),
    ModelSpec("Qwen/Qwen3.5-2B", "qwen3.5", "instruct", True, "multimodal_lm"),
)

DEFAULT_DECODING_SPECS: tuple[DecodingSpec, ...] = (
    DecodingSpec("greedy", 0.0, 1.0, 160),
    DecodingSpec("low_temp", 0.2, 0.95, 180),
    DecodingSpec("normal_temp", 0.7, 0.95, 180),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_text_generation_model(
    model_id: str,
    quantize_4bit: bool = False,
    loader_type: str = "causal_lm",
):
    import torch

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    if loader_type == "multimodal_lm":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        return processor, model

    if loader_type != "causal_lm":
        raise ValueError(f"Unsupported loader type: {loader_type}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tokenizer, model


def _text_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def prepare_text_inputs(processor, prompt: str, use_chat_template: bool):
    tokenizer = _text_tokenizer(processor)
    if use_chat_template and hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
    return tokenizer(prompt, return_tensors="pt")


def prepare_text_batch_inputs(processor, prompts: list[str], use_chat_template: bool):
    if not prompts:
        raise ValueError("prompts must not be empty")

    tokenizer = _text_tokenizer(processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_chat_template and hasattr(processor, "apply_chat_template"):
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        if processor is tokenizer:
            kwargs["padding"] = True
        else:
            kwargs["processor_kwargs"] = {"padding": True}
        try:
            return processor.apply_chat_template(
                conversations,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return processor.apply_chat_template(conversations, **kwargs)

    return tokenizer(prompts, return_tensors="pt", padding=True)


def generate_one(
    tokenizer,
    model,
    prompt: str,
    decoding: DecodingSpec,
    use_chat_template: bool,
    seed: int,
) -> str:
    import torch

    set_seed(seed)
    processor = tokenizer
    text_tokenizer = _text_tokenizer(processor)
    inputs = prepare_text_inputs(processor, prompt, use_chat_template).to(model.device)
    do_sample = decoding.temperature > 0
    generation_kwargs = {
        "max_new_tokens": decoding.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": text_tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = decoding.temperature
        generation_kwargs["top_p"] = decoding.top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_batch(
    tokenizer,
    model,
    prompts: list[str],
    decoding: DecodingSpec,
    use_chat_template: bool,
    seed: int,
) -> list[str]:
    import torch

    set_seed(seed)
    processor = tokenizer
    text_tokenizer = _text_tokenizer(processor)
    inputs = prepare_text_batch_inputs(processor, prompts, use_chat_template).to(model.device)
    do_sample = decoding.temperature > 0
    generation_kwargs = {
        "max_new_tokens": decoding.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": text_tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = decoding.temperature
        generation_kwargs["top_p"] = decoding.top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[:, inputs["input_ids"].shape[-1] :]
    return [
        response.strip()
        for response in text_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    ]


def generation_rows(
    prompts: pd.DataFrame,
    model_spec: ModelSpec,
    decoding: DecodingSpec,
    tokenizer,
    model,
    seeds: list[int],
) -> list[dict]:
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    prompt_records = prompts.to_dict("records")
    prompt_texts = [record["prompt"] for record in prompt_records]
    generation_seeds = seeds if decoding.temperature > 0 else seeds[:1]
    responses_by_seed = {
        seed: generate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=prompt_texts,
            decoding=decoding,
            use_chat_template=model_spec.use_chat_template,
            seed=seed,
        )
        for seed in generation_seeds
    }

    for prompt_index, prompt_row in enumerate(prompt_records):
        for seed in seeds:
            response_seed = seed if decoding.temperature > 0 else generation_seeds[0]
            rows.append(
                {
                    **prompt_row,
                    "model_id": model_spec.model_id,
                    "model_family": model_spec.family,
                    "model_stage": model_spec.stage,
                    "decoding": decoding.name,
                    "temperature": decoding.temperature,
                    "top_p": decoding.top_p,
                    "max_new_tokens": decoding.max_new_tokens,
                    "seed": seed,
                    "response": responses_by_seed[response_seed][prompt_index],
                    "created_at": created_at,
                }
            )
    return rows
