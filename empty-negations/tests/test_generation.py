import pandas as pd

import ocn.generation as generation
from ocn.generation import (
    DecodingSpec,
    ModelSpec,
    generation_rows,
    prepare_text_batch_inputs,
    prepare_text_inputs,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    pad_token_id = None
    eos_token_id = 99
    eos_token = "<eos>"

    def __call__(self, prompt, return_tensors, padding=False):
        self.calls.append((prompt, return_tensors, padding))
        return {"input_ids": [[1, 2, 3]]}


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chat_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_kwargs = {"messages": messages, **kwargs}
        return {"input_ids": [[4, 5, 6]]}


def test_model_spec_defaults_to_causal_loader():
    spec = ModelSpec("example/model", "example", "base")
    assert spec.loader_type == "causal_lm"


def test_base_model_uses_raw_text_tokenization():
    processor = FakeProcessor()
    result = prepare_text_inputs(processor, "Explain photosynthesis.", False)

    assert result == {"input_ids": [[1, 2, 3]]}
    assert processor.tokenizer.calls == [("Explain photosynthesis.", "pt", False)]
    assert processor.chat_kwargs is None


def test_post_trained_model_disables_thinking():
    processor = FakeProcessor()
    result = prepare_text_inputs(processor, "Explain photosynthesis.", True)

    assert result == {"input_ids": [[4, 5, 6]]}
    assert processor.chat_kwargs["enable_thinking"] is False
    assert processor.chat_kwargs["add_generation_prompt"] is True


def test_base_model_batch_uses_left_padding():
    processor = FakeProcessor()
    prompts = ["First", "Second"]

    prepare_text_batch_inputs(processor, prompts, False)

    assert processor.tokenizer.padding_side == "left"
    assert processor.tokenizer.pad_token == "<eos>"
    assert processor.tokenizer.calls == [(prompts, "pt", True)]


def test_post_trained_batch_disables_thinking():
    processor = FakeProcessor()
    prompts = ["First", "Second"]

    prepare_text_batch_inputs(processor, prompts, True)

    assert processor.chat_kwargs["messages"] == [
        [{"role": "user", "content": "First"}],
        [{"role": "user", "content": "Second"}],
    ]
    assert processor.chat_kwargs["enable_thinking"] is False
    assert processor.chat_kwargs["padding"] is True


def test_generation_rows_batches_and_reuses_greedy_output(monkeypatch):
    calls = []

    def fake_generate_batch(**kwargs):
        calls.append((kwargs["prompts"], kwargs["seed"]))
        return [f"response-{index}-seed-{kwargs['seed']}" for index, _ in enumerate(kwargs["prompts"])]

    monkeypatch.setattr(generation, "generate_batch", fake_generate_batch)
    prompts = pd.DataFrame(
        [
            {"prompt_id": "p1", "prompt": "First"},
            {"prompt_id": "p2", "prompt": "Second"},
        ]
    )

    rows = generation_rows(
        prompts=prompts,
        model_spec=ModelSpec("example/model", "example", "base"),
        decoding=DecodingSpec("greedy", 0.0),
        tokenizer=object(),
        model=object(),
        seeds=[1, 2],
    )

    assert calls == [(["First", "Second"], 1)]
    assert [(row["prompt_id"], row["seed"]) for row in rows] == [
        ("p1", 1),
        ("p1", 2),
        ("p2", 1),
        ("p2", 2),
    ]
    assert rows[0]["response"] == rows[1]["response"]


def test_generation_rows_batches_sampled_output_by_seed(monkeypatch):
    calls = []

    def fake_generate_batch(**kwargs):
        calls.append(kwargs["seed"])
        return [f"seed-{kwargs['seed']}"] * len(kwargs["prompts"])

    monkeypatch.setattr(generation, "generate_batch", fake_generate_batch)
    prompts = pd.DataFrame([{"prompt_id": "p1", "prompt": "First"}])

    rows = generation_rows(
        prompts=prompts,
        model_spec=ModelSpec("example/model", "example", "instruct", True),
        decoding=DecodingSpec("normal_temp", 0.7),
        tokenizer=object(),
        model=object(),
        seeds=[1, 2],
    )

    assert calls == [1, 2]
    assert [row["response"] for row in rows] == ["seed-1", "seed-2"]
