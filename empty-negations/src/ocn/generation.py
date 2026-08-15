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


@dataclass(frozen=True)
class DecodingSpec:
    name: str
    temperature: float
    top_p: float = 0.95
    max_new_tokens: int = 180


DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen/Qwen2.5-0.5B", "qwen", "base", False),
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "qwen", "instruct", True),
    ModelSpec("Qwen/Qwen2.5-1.5B", "qwen", "base", False),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "qwen", "instruct", True),
    ModelSpec("google/gemma-2-2b", "gemma", "base", False),
    ModelSpec("google/gemma-2-2b-it", "gemma", "instruct", True),
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


def load_text_generation_model(model_id: str, quantize_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tokenizer, model


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


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
    formatted = format_prompt(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    do_sample = decoding.temperature > 0
    generation_kwargs = {
        "max_new_tokens": decoding.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = decoding.temperature
        generation_kwargs["top_p"] = decoding.top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    for _, prompt_row in prompts.iterrows():
        for seed in seeds:
            response = generate_one(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt_row["prompt"],
                decoding=decoding,
                use_chat_template=model_spec.use_chat_template,
                seed=seed,
            )
            rows.append(
                {
                    **prompt_row.to_dict(),
                    "model_id": model_spec.model_id,
                    "model_family": model_spec.family,
                    "model_stage": model_spec.stage,
                    "decoding": decoding.name,
                    "temperature": decoding.temperature,
                    "top_p": decoding.top_p,
                    "max_new_tokens": decoding.max_new_tokens,
                    "seed": seed,
                    "response": response,
                    "created_at": created_at,
                }
            )
    return rows
