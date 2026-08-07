"""Calibration, accuracy, and selective-prediction metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence


def expected_calibration_error(
    correctness: Sequence[bool], confidence: Sequence[float], bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    if len(correctness) != len(confidence):
        raise ValueError("correctness and confidence must have equal length")
    n = len(correctness)
    if n == 0:
        return 0.0, []
    details: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = []
        for i, value in enumerate(confidence):
            in_bin = lower <= value <= upper if index == bins - 1 else lower <= value < upper
            if in_bin:
                selected.append(i)
        if not selected:
            details.append(
                {"lower": lower, "upper": upper, "count": 0, "accuracy": 0.0, "confidence": 0.0}
            )
            continue
        accuracy = sum(correctness[i] for i in selected) / len(selected)
        mean_confidence = sum(confidence[i] for i in selected) / len(selected)
        ece += len(selected) / n * abs(accuracy - mean_confidence)
        details.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return ece, details


def risk_coverage_curve(
    correctness: Sequence[bool], confidence: Sequence[float]
) -> list[dict[str, float]]:
    order = sorted(range(len(correctness)), key=lambda index: confidence[index], reverse=True)
    errors = 0
    curve = []
    total = len(order)
    for rank, index in enumerate(order, start=1):
        errors += int(not correctness[index])
        curve.append(
            {
                "coverage": rank / total,
                "risk": errors / rank,
                "accuracy": 1.0 - errors / rank,
                "threshold": confidence[index],
            }
        )
    return curve


def threshold_metrics(
    correctness: Sequence[bool],
    confidence: Sequence[float],
    thresholds: Iterable[float],
) -> list[dict[str, float | int]]:
    total = len(correctness)
    output = []
    for threshold in thresholds:
        answered = [index for index, value in enumerate(confidence) if value >= threshold]
        correct = sum(correctness[index] for index in answered)
        output.append(
            {
                "threshold": threshold,
                "answered": len(answered),
                "coverage": len(answered) / total if total else 0.0,
                "selective_accuracy": correct / len(answered) if answered else 0.0,
                "risk": 1.0 - correct / len(answered) if answered else 0.0,
            }
        )
    return output


def _confidence(record: dict[str, Any], source: str) -> float:
    if source == "answer_probability":
        value = record.get("answer_probability")
    else:
        value = (record.get("prediction") or {}).get("confidence")
    return min(max(float(value), 0.0), 1.0) if value is not None else 0.0


def _multiclass_scores(records: Sequence[dict[str, Any]]) -> tuple[float | None, float | None]:
    brier_values = []
    nll_values = []
    for record in records:
        probabilities = record.get("option_probabilities")
        if not isinstance(probabilities, dict):
            continue
        gold = str(record["answer_label"])
        labels = [chr(ord("A") + index) for index in range(len(record["choices"]))]
        brier_values.append(
            sum((float(probabilities.get(label, 0.0)) - float(label == gold)) ** 2 for label in labels)
        )
        nll_values.append(-math.log(max(float(probabilities.get(gold, 0.0)), 1e-12)))
    return (
        sum(brier_values) / len(brier_values) if brier_values else None,
        sum(nll_values) / len(nll_values) if nll_values else None,
    )


def evaluate_records(
    records: Sequence[dict[str, Any]],
    *,
    confidence_source: str = "verbal",
    bins: int = 10,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    include_slices: bool = True,
) -> dict[str, Any]:
    predictions = [record.get("prediction") or {} for record in records]
    correctness = [
        bool(prediction.get("answer") == record.get("answer_label"))
        for prediction, record in zip(predictions, records)
    ]
    confidence = [_confidence(record, confidence_source) for record in records]
    valid = [bool(prediction.get("schema_valid")) for prediction in predictions]
    json_valid = [bool(prediction.get("json_valid")) for prediction in predictions]
    n = len(records)
    ece, calibration_bins = expected_calibration_error(correctness, confidence, bins)
    binary_brier = (
        sum((value - float(correct)) ** 2 for value, correct in zip(confidence, correctness)) / n
        if n
        else 0.0
    )
    binary_nll = (
        sum(
            -(
                float(correct) * math.log(max(value, 1e-12))
                + float(not correct) * math.log(max(1.0 - value, 1e-12))
            )
            for value, correct in zip(confidence, correctness)
        )
        / n
        if n
        else 0.0
    )
    curve = risk_coverage_curve(correctness, confidence)
    multiclass_brier, multiclass_nll = _multiclass_scores(records)
    errors = [confidence[index] for index, correct in enumerate(correctness) if not correct]
    model_answered = [
        index for index, prediction in enumerate(predictions) if prediction.get("abstain") is False
    ]
    model_correct = sum(correctness[index] for index in model_answered)

    result: dict[str, Any] = {
        "records": n,
        "accuracy": sum(correctness) / n if n else 0.0,
        "format_validity": sum(valid) / n if n else 0.0,
        "json_validity": sum(json_valid) / n if n else 0.0,
        "confidence_source": confidence_source,
        "ece": ece,
        "binary_brier": binary_brier,
        "binary_correctness_nll": binary_nll,
        "multiclass_brier": multiclass_brier,
        "multiclass_nll": multiclass_nll,
        "mean_confidence": sum(confidence) / n if n else 0.0,
        "mean_error_confidence": sum(errors) / len(errors) if errors else 0.0,
        "aurc": sum(point["risk"] for point in curve) / len(curve) if curve else 0.0,
        "model_abstention": {
            "answered": len(model_answered),
            "coverage": len(model_answered) / n if n else 0.0,
            "selective_accuracy": model_correct / len(model_answered) if model_answered else 0.0,
            "risk": 1.0 - model_correct / len(model_answered) if model_answered else 0.0,
        },
        "thresholds": threshold_metrics(correctness, confidence, thresholds),
        "calibration_bins": calibration_bins,
        "risk_coverage": curve,
    }

    if include_slices:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_source[str(record.get("source", "unknown"))].append(record)
        result["by_source"] = {
            source: evaluate_records(
                source_records,
                confidence_source=confidence_source,
                bins=bins,
                thresholds=thresholds,
                include_slices=False,
            )
            for source, source_records in sorted(by_source.items())
        }
    return result
