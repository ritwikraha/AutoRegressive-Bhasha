import pandas as pd
import pytest

from ocn.annotation import (
    ANNOTATION_FIELDS,
    ANNOTATION_PROMPT_VERSION,
    agreement_summary,
    annotation_calibration_frame,
    build_span_population,
    compare_annotations,
    finalize_adjudications,
    make_annotation_prompt,
    parse_annotation_json,
    stratified_response_sample,
)


def detection_rows():
    response = "The policy is not just a tax, but a long-term investment signal."
    second = "The result goes beyond accuracy and includes reliability."
    return pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "model_id": "model-a",
                "model_stage": "instruct",
                "decoding": "greedy",
                "seed": seed,
                "prompt": "Explain the policy.",
                "response": response,
                "has_ocn": True,
            }
            for seed in [1, 2]
        ]
        + [
            {
                "prompt_id": "p2",
                "model_id": "model-b",
                "model_stage": "base",
                "decoding": "normal_temp",
                "seed": 1,
                "prompt": "Summarize the result.",
                "response": second,
                "has_ocn": True,
            }
        ]
    )


def annotation(example_id, label="scope_inflation", prompt_support=2, meaning=2):
    return {
        "example_id": example_id,
        "rejected_x": "a tax",
        "asserted_y": "an investment signal",
        "taxonomy_label": label,
        "prompt_support": prompt_support,
        "common_misconception": 2,
        "x_y_distinctness": 4,
        "negation_adds_meaning": meaning,
        "straw_position": 3,
        "formulaic_ai_style": 4,
        "rewrite_loss": 2,
        "notes": "The prompt did not introduce the rejected view.",
    }


def test_build_population_collapses_reused_seed_and_creates_span_rows():
    population = build_span_population(detection_rows())

    assert len(population) == 2
    first = population.loc[population["prompt_id"].eq("p1")].iloc[0]
    assert first["source_row_multiplicity"] == 2
    assert first["source_seeds"] == "1|2"
    assert first["span_pattern"] == "isnt_just_but"
    assert first["example_id"].startswith("span_")


def test_stratified_sample_is_deterministic_and_keeps_response_spans_together():
    population = build_span_population(detection_rows())
    first = stratified_response_sample(population, target_responses_per_stratum=1, seed=7)
    second = stratified_response_sample(population, target_responses_per_stratum=1, seed=7)

    assert first["example_id"].tolist() == second["example_id"].tolist()
    assert set(first["sample_weight"]) == {1.0}


def test_parse_annotation_json_accepts_fences_and_rejects_invalid_ratings():
    record = annotation("span_1")
    record.pop("example_id")
    parsed = parse_annotation_json(f"```json\n{pd.Series(record).to_json()}\n```")
    assert parsed["taxonomy_label"] == "scope_inflation"
    assert parsed["prompt_support"] == 2

    record["prompt_support"] = 7
    with pytest.raises(ValueError, match="prompt_support"):
        parse_annotation_json(pd.Series(record).to_json())


def test_parse_annotation_json_repairs_json_and_zero_based_absence():
    record = annotation("span_1")
    record.pop("example_id")
    record["asserted_y"] = ""
    record["common_misconception"] = 0
    malformed = pd.Series(record).to_json()[:-1] + ",}"

    parsed = parse_annotation_json(malformed)

    assert parsed["asserted_y"] == ""
    assert parsed["common_misconception"] == 1
    assert "zero_to_one:common_misconception" in parsed["annotation_parse_repairs"]


def test_compare_and_full_adjudication_resolve_semantic_outcomes():
    sample = build_span_population(detection_rows()).iloc[[0]].copy()
    example_id = sample.iloc[0]["example_id"]
    a = pd.DataFrame([annotation(example_id, "scope_inflation", 2, 2)])
    b = pd.DataFrame([annotation(example_id, "genuine_contrast", 5, 5)])

    comparison = compare_annotations(sample, a, b)
    assert comparison.iloc[0]["adjudication_required"]
    assert set(comparison.iloc[0]["disagreement_reasons"].split("|")) == {
        "taxonomy",
        "prompt_support",
        "negation_adds_meaning",
    }

    adjudication = pd.DataFrame(
        [annotation(example_id, "false_correction", 1, 1)]
    )
    final = finalize_adjudications(comparison, adjudication, adjudicate_all=True)
    assert final.iloc[0]["taxonomy_label"] == "false_correction"
    assert final.iloc[0]["strict_misuse"]
    assert final.iloc[0]["unsupported_contrast"]
    assert set(ANNOTATION_FIELDS).issubset(final.columns)


def test_agreement_summary_reports_taxonomy_and_rating_metrics():
    sample = build_span_population(detection_rows())
    a = pd.DataFrame([annotation(value) for value in sample["example_id"]])
    b = pd.DataFrame([annotation(value) for value in sample["example_id"]])
    summary = agreement_summary(compare_annotations(sample, a, b))

    assert len(summary) == 8
    assert summary.loc[summary["field"].eq("taxonomy_label"), "exact_agreement"].item() == 1.0


def test_calibration_cases_cover_distinct_boundaries_and_prompt_is_explicit():
    calibration = annotation_calibration_frame()
    prompt = make_annotation_prompt(calibration.iloc[0].to_dict())

    assert ANNOTATION_PROMPT_VERSION == "v2_calibrated"
    assert len(calibration) == 8
    assert calibration["expected_taxonomy_label"].nunique() == 8
    assert "considers only the USER PROMPT" in prompt
    assert "Apply this order" in prompt
    assert "Calibration anchors" in prompt
