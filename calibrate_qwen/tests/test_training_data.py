import json

from training.prepare_training_data import prepare_training_file, record_to_conversation


def base_record() -> dict:
    return {
        "id": "example-1",
        "source": "test",
        "prompt": "Question: Example?\nA. First\nB. Second",
        "answer_label": "B",
        "teacher": {
            "modal_answer": "A",
            "agreement_confidence": 0.4,
            "representative_justification": "First follows from the premise.",
        },
    }


def test_teacher_conversation_uses_agreement_and_abstention() -> None:
    conversation = record_to_conversation(
        base_record(),
        variant="teacher",
        confidence_format="numeric",
        abstention_threshold=0.6,
    )
    target = json.loads(conversation["messages"][-1]["content"])
    assert target["answer"] == "A"
    assert target["confidence"] == 0.4
    assert target["abstain"] is True


def test_prepare_training_file_writes_manifest(tmp_path) -> None:
    output = tmp_path / "teacher.jsonl"
    manifest = prepare_training_file(
        output_path=output,
        variant="teacher",
        records=[base_record()],
    )
    assert manifest["written_records"] == 1
    assert output.exists()
    assert (tmp_path / "teacher_manifest.json").exists()


def test_implicit_confidence_omits_field() -> None:
    conversation = record_to_conversation(
        base_record(),
        variant="teacher",
        confidence_format="implicit",
        abstention_threshold=0.6,
    )
    target = json.loads(conversation["messages"][-1]["content"])
    assert "confidence" not in target
