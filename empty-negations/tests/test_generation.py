from ocn.generation import ModelSpec, prepare_text_inputs


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, return_tensors):
        self.calls.append((prompt, return_tensors))
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
    assert processor.tokenizer.calls == [("Explain photosynthesis.", "pt")]
    assert processor.chat_kwargs is None


def test_post_trained_model_disables_thinking():
    processor = FakeProcessor()
    result = prepare_text_inputs(processor, "Explain photosynthesis.", True)

    assert result == {"input_ids": [[4, 5, 6]]}
    assert processor.chat_kwargs["enable_thinking"] is False
    assert processor.chat_kwargs["add_generation_prompt"] is True
