"""Shared structured-output and configuration helpers for training."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

ConfidenceFormat = Literal["numeric", "bucket", "implicit"]


@dataclass(frozen=True)
class StructuredOutput:
    answer: str
    justification: str
    abstain: bool
    confidence: float | str | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        if self.confidence is None:
            payload.pop("confidence")
        ordered = {"answer": payload.pop("answer")}
        if "confidence" in payload:
            ordered["confidence"] = payload.pop("confidence")
        ordered.update(payload)
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def confidence_bucket(confidence: float) -> str:
    if confidence < 0.5:
        return "low"
    if confidence < 0.8:
        return "medium"
    return "high"


def build_system_prompt(confidence_format: ConfidenceFormat = "numeric") -> str:
    fields = (
        '"answer", "confidence", "justification", and "abstain"'
        if confidence_format != "implicit"
        else '"answer", "justification", and "abstain"'
    )
    confidence_rule = {
        "numeric": "confidence must be a number between 0 and 1.",
        "bucket": 'confidence must be one of "low", "medium", or "high".',
        "implicit": "Confidence will be computed from answer-token probabilities.",
    }[confidence_format]
    return (
        "Answer the multiple-choice question. Return one JSON object with exactly the keys "
        f"{fields}. answer must be one uppercase option label from the prompt. "
        f"{confidence_rule} justification must be one concise sentence. "
        "abstain must be a JSON boolean."
    )


def format_target(
    *,
    answer: str,
    confidence: float,
    justification: str,
    abstention_threshold: float,
    confidence_format: ConfidenceFormat = "numeric",
) -> StructuredOutput:
    clipped = min(max(float(confidence), 0.0), 1.0)
    rendered_confidence: float | str | None
    if confidence_format == "numeric":
        rendered_confidence = round(clipped, 4)
    elif confidence_format == "bucket":
        rendered_confidence = confidence_bucket(clipped)
    else:
        rendered_confidence = None
    return StructuredOutput(
        answer=str(answer).strip().upper(),
        confidence=rendered_confidence,
        justification=" ".join(str(justification).split()),
        abstain=clipped < abstention_threshold,
    )


def require_tinker_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise EnvironmentError("Set TINKER_API_KEY before starting a Tinker run.")


def optional_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def teacher_fields(record: dict[str, Any]) -> tuple[str, float, str]:
    teacher = record.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError(f"Record {record.get('id')} has no teacher metadata.")
    answer = teacher.get("modal_answer")
    if not answer:
        raise ValueError(f"Record {record.get('id')} has no valid teacher answer.")
    return (
        str(answer),
        float(teacher.get("agreement_confidence", 0.0)),
        str(teacher.get("representative_justification") or "The selected option best fits the evidence."),
    )
