"""Fit temperature scaling and an abstention threshold on validation predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.common import read_jsonl


def temperature_scale(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    labels = sorted(probabilities)
    logits = np.log(np.maximum([probabilities[label] for label in labels], 1e-12)) / temperature
    logits -= logits.max()
    values = np.exp(logits)
    values /= values.sum()
    return {label: float(value) for label, value in zip(labels, values)}


def multiclass_nll(records: Sequence[dict[str, Any]], temperature: float) -> float:
    losses = []
    for record in records:
        probabilities = record.get("option_probabilities")
        if not isinstance(probabilities, dict):
            continue
        scaled = temperature_scale(probabilities, temperature)
        losses.append(-math.log(max(scaled.get(str(record["answer_label"]), 0.0), 1e-12)))
    if not losses:
        raise ValueError("Validation records require option_probabilities for temperature scaling.")
    return sum(losses) / len(losses)


def fit_temperature(
    records: Sequence[dict[str, Any]],
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
    candidates: int = 300,
) -> tuple[float, float]:
    temperatures = np.geomspace(minimum, maximum, candidates)
    scored = [(float(value), multiclass_nll(records, float(value))) for value in temperatures]
    return min(scored, key=lambda item: item[1])


def select_abstention_threshold(
    records: Sequence[dict[str, Any]],
    *,
    temperature: float,
    target_coverage: float = 0.8,
) -> dict[str, float | int]:
    rows = []
    for record in records:
        probabilities = record.get("option_probabilities")
        if not isinstance(probabilities, dict):
            continue
        scaled = temperature_scale(probabilities, temperature)
        answer = (record.get("prediction") or {}).get("answer")
        confidence = scaled.get(str(answer), 0.0)
        rows.append((confidence, answer == record.get("answer_label")))
    if not rows:
        raise ValueError("Validation records require option_probabilities for threshold selection.")
    rows.sort(key=lambda item: item[0], reverse=True)
    answered = max(1, min(len(rows), round(target_coverage * len(rows))))
    selected = rows[:answered]
    threshold = selected[-1][0]
    accuracy = sum(correct for _, correct in selected) / answered
    return {
        "threshold": threshold,
        "target_coverage": target_coverage,
        "validation_coverage": answered / len(rows) if rows else 0.0,
        "validation_selective_accuracy": accuracy,
        "validation_risk": 1.0 - accuracy,
        "answered": answered,
    }


def fit_calibration(
    records: Sequence[dict[str, Any]],
    *,
    target_coverage: float = 0.8,
) -> dict[str, Any]:
    temperature, validation_nll = fit_temperature(records)
    return {
        "temperature": temperature,
        "validation_multiclass_nll": validation_nll,
        "abstention": select_abstention_threshold(
            records,
            temperature=temperature,
            target_coverage=target_coverage,
        ),
    }


def apply_calibration(record: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    probabilities = output.get("option_probabilities")
    if not isinstance(probabilities, dict):
        return output
    scaled = temperature_scale(probabilities, float(calibration["temperature"]))
    prediction = dict(output.get("prediction") or {})
    answer = prediction.get("answer")
    confidence = scaled.get(str(answer), 0.0)
    prediction["confidence"] = confidence
    prediction["abstain"] = confidence < float(calibration["abstention"]["threshold"])
    output["prediction"] = prediction
    output["option_probabilities"] = scaled
    output["answer_probability"] = confidence
    output["calibration"] = calibration
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--validation", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--target-coverage", type=float, default=0.8)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--input", required=True)
    apply_parser.add_argument("--calibration", required=True)
    apply_parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "fit":
        calibration = fit_calibration(
            list(read_jsonl(args.validation)),
            target_coverage=args.target_coverage,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(calibration, indent=2))
        return

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    records = [apply_calibration(record, calibration) for record in read_jsonl(args.input)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} calibrated predictions to {output}")


if __name__ == "__main__":
    main()
