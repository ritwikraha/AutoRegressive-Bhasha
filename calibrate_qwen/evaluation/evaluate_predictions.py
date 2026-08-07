"""Compute all CalibrateQwen metrics from sampled prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.common import read_jsonl
from evaluation.metrics import evaluate_records
from evaluation.robustness import evaluate_robustness


def evaluate_file(
    *,
    predictions_path: str | Path,
    output_path: str | Path,
    confidence_source: str = "verbal",
    base_predictions_path: str | Path | None = None,
) -> dict:
    records = list(read_jsonl(predictions_path))
    metrics = evaluate_records(records, confidence_source=confidence_source)
    if base_predictions_path:
        base_records = list(read_jsonl(base_predictions_path))
        metrics["robustness"] = evaluate_robustness(base_records, records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--confidence-source", choices=["verbal", "answer_probability"], default="verbal"
    )
    parser.add_argument("--base-predictions", default=None)
    args = parser.parse_args()
    metrics = evaluate_file(
        predictions_path=args.predictions,
        output_path=args.output,
        confidence_source=args.confidence_source,
        base_predictions_path=args.base_predictions,
    )
    summary_keys = [
        "records",
        "accuracy",
        "ece",
        "binary_brier",
        "multiclass_brier",
        "aurc",
        "mean_error_confidence",
        "format_validity",
    ]
    print(json.dumps({key: metrics.get(key) for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
