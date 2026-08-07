"""Option-order robustness metrics joined through parent IDs and answer text."""

from __future__ import annotations

from typing import Any, Sequence


def selected_choice(record: dict[str, Any]) -> str | None:
    prediction = record.get("prediction") or {}
    answer = prediction.get("answer")
    if not isinstance(answer, str) or len(answer) != 1:
        return None
    index = ord(answer.upper()) - ord("A")
    choices = record.get("choices") or []
    return str(choices[index]) if 0 <= index < len(choices) else None


def evaluate_robustness(
    base_records: Sequence[dict[str, Any]],
    perturbed_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base_by_id = {str(record["id"]): record for record in base_records}
    pairs = [
        (base_by_id[str(record["parent_id"])], record)
        for record in perturbed_records
        if str(record.get("parent_id")) in base_by_id
    ]
    base_correct = []
    perturbed_correct = []
    choice_consistent = []
    for base, perturbed in pairs:
        base_prediction = (base.get("prediction") or {}).get("answer")
        perturbed_prediction = (perturbed.get("prediction") or {}).get("answer")
        base_correct.append(base_prediction == base.get("answer_label"))
        perturbed_correct.append(perturbed_prediction == perturbed.get("answer_label"))
        base_choice = selected_choice(base)
        perturbed_choice = selected_choice(perturbed)
        choice_consistent.append(base_choice is not None and base_choice == perturbed_choice)

    count = len(pairs)
    base_accuracy = sum(base_correct) / count if count else 0.0
    perturbed_accuracy = sum(perturbed_correct) / count if count else 0.0
    consistency = sum(choice_consistent) / count if count else 0.0
    return {
        "paired_records": count,
        "base_accuracy_on_pairs": base_accuracy,
        "perturbed_accuracy": perturbed_accuracy,
        "accuracy_delta": perturbed_accuracy - base_accuracy,
        "choice_text_consistency": consistency,
        "choice_flip_rate": 1.0 - consistency if count else 0.0,
    }
