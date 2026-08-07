import math

from evaluation.calibrate import fit_calibration, temperature_scale
from evaluation.metrics import evaluate_records, expected_calibration_error, risk_coverage_curve
from evaluation.parsing import parse_completion
from evaluation.robustness import evaluate_robustness


def prediction_record(
    record_id: str,
    answer: str,
    gold: str,
    confidence: float,
    *,
    source: str = "test",
) -> dict:
    return {
        "id": record_id,
        "source": source,
        "answer_label": gold,
        "choices": ["alpha", "beta"],
        "prediction": {
            "answer": answer,
            "confidence": confidence,
            "justification": "A concise reason.",
            "abstain": False,
            "json_valid": True,
            "schema_valid": True,
        },
    }


def test_parser_accepts_balanced_json_and_bucket() -> None:
    parsed = parse_completion(
        'prefix {"answer":"B","confidence":"medium","justification":"Reason.","abstain":false}',
        {"A", "B"},
    )
    assert parsed["answer"] == "B"
    assert parsed["confidence"] == 0.65
    assert parsed["schema_valid"] is True


def test_ece_for_perfect_predictions() -> None:
    ece, bins = expected_calibration_error([True, False], [1.0, 0.0], bins=2)
    assert ece == 0.0
    assert sum(item["count"] for item in bins) == 2


def test_metrics_and_risk_order() -> None:
    records = [
        prediction_record("1", "A", "A", 0.9),
        prediction_record("2", "A", "B", 0.2),
    ]
    metrics = evaluate_records(records)
    curve = risk_coverage_curve([True, False], [0.9, 0.2])
    assert metrics["accuracy"] == 0.5
    assert math.isclose(metrics["binary_brier"], 0.025)
    assert curve[0]["risk"] == 0.0
    assert curve[-1]["risk"] == 0.5


def test_option_shuffle_consistency_compares_choice_text() -> None:
    base = prediction_record("parent", "A", "A", 0.9)
    perturbed = prediction_record("child", "B", "B", 0.9)
    perturbed["parent_id"] = "parent"
    perturbed["choices"] = ["beta", "alpha"]
    metrics = evaluate_robustness([base], [perturbed])
    assert metrics["perturbed_accuracy"] == 1.0
    assert metrics["choice_text_consistency"] == 1.0


def test_temperature_scaling_and_threshold_fit() -> None:
    records = [
        {
            **prediction_record("1", "A", "A", 0.8),
            "option_probabilities": {"A": 0.8, "B": 0.2},
        },
        {
            **prediction_record("2", "B", "A", 0.8),
            "option_probabilities": {"A": 0.2, "B": 0.8},
        },
    ]
    scaled = temperature_scale(records[0]["option_probabilities"], 2.0)
    assert math.isclose(sum(scaled.values()), 1.0)
    assert scaled["A"] < 0.8
    calibration = fit_calibration(records, target_coverage=0.5)
    assert 0.25 <= calibration["temperature"] <= 4.0
    assert calibration["abstention"]["answered"] == 1
