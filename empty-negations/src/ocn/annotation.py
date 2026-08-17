from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ocn.detectors import OCNDetector


TAXONOMY_LABELS: tuple[str, ...] = (
    "genuine_contrast",
    "legitimate_pedagogy",
    "presupposed_contrast",
    "empty_intensification",
    "scope_inflation",
    "false_correction",
    "template_stacking",
    "non_ocn_negation",
    "unclear",
)

RATING_FIELDS: tuple[str, ...] = (
    "prompt_support",
    "common_misconception",
    "x_y_distinctness",
    "negation_adds_meaning",
    "straw_position",
    "formulaic_ai_style",
    "rewrite_loss",
)

ANNOTATION_FIELDS: tuple[str, ...] = (
    "rejected_x",
    "asserted_y",
    "taxonomy_label",
    *RATING_FIELDS,
    "notes",
)

STRICT_MISUSE_LABELS = {
    "empty_intensification",
    "scope_inflation",
    "false_correction",
}
BROAD_MISUSE_LABELS = STRICT_MISUSE_LABELS | {
    "presupposed_contrast",
    "template_stacking",
}

_LABEL_GUIDE = """
- genuine_contrast: the prompt or preceding discourse supplies the rejected proposition X.
- legitimate_pedagogy: X is a well-known misconception whose correction is useful here.
- presupposed_contrast: X is plausible but was not introduced by the prompt.
- empty_intensification: X and Y are near paraphrases or the contrast adds no substance.
- scope_inflation: Y merely broadens X rather than genuinely contrasting with it.
- false_correction: X is an implausible straw position or unmotivated correction.
- template_stacking: the construction belongs to a larger cluster of formulaic assistant rhetoric.
- non_ocn_negation: this is ordinary factual negation, not rhetorical contrastive negation.
- unclear: evidence is genuinely insufficient or mixed.
""".strip()


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def build_span_population(
    detections: pd.DataFrame,
    detector: OCNDetector | None = None,
) -> pd.DataFrame:
    """Collapse repeated generations and create one row per detected span."""
    required = {"prompt_id", "model_id", "decoding", "prompt", "response", "has_ocn"}
    missing = sorted(required - set(detections.columns))
    if missing:
        raise ValueError(f"Detection data is missing required columns: {missing}")

    candidates = detections[detections["has_ocn"].astype(bool)].copy()
    if candidates.empty:
        return pd.DataFrame()

    keys = ["prompt_id", "model_id", "decoding", "response"]
    candidates = candidates.sort_values(
        [column for column in [*keys[:-1], "seed"] if column in candidates.columns]
    )
    multiplicity = candidates.groupby(keys, dropna=False).size().rename("source_row_multiplicity")
    if "seed" in candidates.columns:
        seeds = (
            candidates.groupby(keys, dropna=False)["seed"]
            .agg(lambda values: "|".join(str(value) for value in sorted(set(values))))
            .rename("source_seeds")
        )
    else:
        seeds = pd.Series("", index=multiplicity.index, name="source_seeds")

    unique = candidates.drop_duplicates(keys, keep="first").set_index(keys)
    unique = unique.join([multiplicity, seeds]).reset_index()
    lexical_detector = detector or OCNDetector()

    rows: list[dict[str, Any]] = []
    for record in unique.to_dict("records"):
        response_id = _stable_id(
            "response",
            record["prompt_id"],
            record["model_id"],
            record["decoding"],
            record["response"],
        )
        spans = lexical_detector.detect(record["response"]).spans
        for span_index, span in enumerate(spans):
            example_id = _stable_id(
                "span",
                response_id,
                span_index,
                span.pattern_name,
                span.start,
                span.end,
            )
            rows.append(
                {
                    **record,
                    "response_id": response_id,
                    "example_id": example_id,
                    "span_index": span_index,
                    "span_pattern": span.pattern_name,
                    "span_start": span.start,
                    "span_end": span.end,
                    "span_text": span.text,
                }
            )

    population = pd.DataFrame(rows)
    return population.sort_values(
        ["model_id", "decoding", "prompt_id", "response_id", "span_index"]
    ).reset_index(drop=True)


def stratified_response_sample(
    population: pd.DataFrame,
    target_responses_per_stratum: int = 50,
    strata: Iterable[str] = ("model_id", "decoding"),
    seed: int = 20260817,
) -> pd.DataFrame:
    """Sample responses within strata and retain all spans from selected responses."""
    if target_responses_per_stratum <= 0:
        raise ValueError("target_responses_per_stratum must be positive")
    strata = tuple(strata)
    required = {"response_id", "example_id", *strata}
    missing = sorted(required - set(population.columns))
    if missing:
        raise ValueError(f"Population is missing required columns: {missing}")

    response_frame = population.drop_duplicates("response_id")
    selected_ids: list[str] = []
    allocation_rows: list[dict[str, Any]] = []
    group_key = list(strata) if len(strata) > 1 else strata[0]
    for key, group in response_frame.groupby(group_key, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        population_n = len(group)
        sample_n = min(population_n, target_responses_per_stratum)
        stratum_seed = int(
            hashlib.sha1(f"{seed}|{key_values}".encode("utf-8")).hexdigest()[:8], 16
        )
        chosen = group.sample(n=sample_n, random_state=stratum_seed)
        selected_ids.extend(chosen["response_id"].tolist())
        allocation_rows.append(
            {
                **dict(zip(strata, key_values)),
                "stratum_population_responses": population_n,
                "stratum_sample_responses": sample_n,
                "sampling_probability": sample_n / population_n,
                "sample_weight": population_n / sample_n,
            }
        )

    allocation = pd.DataFrame(allocation_rows)
    sample = population[population["response_id"].isin(selected_ids)].merge(
        allocation,
        on=list(strata),
        how="left",
        validate="many_to_one",
    )
    sample["sample_seed"] = seed
    sample["sample_order"] = sample["example_id"].map(
        lambda value: hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest()
    )
    return sample.sort_values("sample_order").drop(columns="sample_order").reset_index(drop=True)


def make_annotation_packet(
    sample: pd.DataFrame,
    packet_id: str,
    seed: int,
) -> pd.DataFrame:
    """Create a blinded, independently ordered packet for a human annotator."""
    visible = ["example_id", "prompt", "response", "span_text"]
    missing = sorted(set(visible) - set(sample.columns))
    if missing:
        raise ValueError(f"Sample is missing packet columns: {missing}")
    packet = sample[visible].copy().sample(frac=1, random_state=seed).reset_index(drop=True)
    packet.insert(0, "packet_row", np.arange(1, len(packet) + 1))
    packet.insert(0, "packet_id", packet_id)
    for field in ANNOTATION_FIELDS:
        packet[field] = ""
    return packet


def make_annotation_prompt(record: dict[str, Any]) -> str:
    return f"""
You are one independent research annotator. Evaluate exactly one detected rhetorical-negation span using the codebook below. Judge the semantics in context, not merely the lexical pattern. Do not infer that a detector match is misuse.

Taxonomy:
{_LABEL_GUIDE}

Use integer ratings where 1=definitely no/absent, 2=probably no, 3=unclear or mixed, 4=probably yes, and 5=definitely yes/strong.

PROMPT:
{record['prompt']}

FULL RESPONSE:
{record['response']}

DETECTED SPAN:
{record['span_text']}

Return only one JSON object with exactly these keys:
{{"rejected_x":"...","asserted_y":"...","taxonomy_label":"one allowed label","prompt_support":1,"common_misconception":1,"x_y_distinctness":1,"negation_adds_meaning":1,"straw_position":1,"formulaic_ai_style":1,"rewrite_loss":1,"notes":"brief evidence-based rationale"}}
""".strip()


def make_annotation_repair_prompt(record: dict[str, Any], invalid_output: str) -> str:
    return f"""
Your previous annotation was not valid JSON under the required schema. Re-evaluate the item and return only a corrected JSON object. Do not use Markdown.

{make_annotation_prompt(record)}

INVALID PREVIOUS OUTPUT:
{invalid_output[:3000]}
""".strip()


def make_adjudication_prompt(record: dict[str, Any]) -> str:
    first = {field: record[f"a_{field}"] for field in ANNOTATION_FIELDS}
    second = {field: record[f"b_{field}"] for field in ANNOTATION_FIELDS}
    reasons = record.get("disagreement_reasons", "none") or "none"
    return f"""
You are the adjudicator for a semantic annotation study. Independently inspect the prompt, response, and detected span, then issue the best final annotation. The two prior annotations are evidence, not votes. Resolve their differences using the codebook.

Taxonomy:
{_LABEL_GUIDE}

PROMPT:
{record['prompt']}

FULL RESPONSE:
{record['response']}

DETECTED SPAN:
{record['span_text']}

AUTOMATIC DISAGREEMENT FLAGS: {reasons}

PRIOR ANNOTATION 1:
{json.dumps(first, ensure_ascii=True)}

PRIOR ANNOTATION 2:
{json.dumps(second, ensure_ascii=True)}

Use integer ratings from 1 to 5. Return only one JSON object with exactly these keys:
{{"rejected_x":"...","asserted_y":"...","taxonomy_label":"one allowed label","prompt_support":1,"common_misconception":1,"x_y_distinctness":1,"negation_adds_meaning":1,"straw_position":1,"formulaic_ai_style":1,"rewrite_loss":1,"notes":"brief adjudication rationale"}}
""".strip()


def parse_annotation_json(text: str) -> dict[str, Any]:
    """Extract and validate the first annotation object in model output."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Annotation output is empty")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    parsed = None
    for match in re.finditer(r"\{", cleaned):
        try:
            candidate, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate.get("annotation", candidate)
            break
    if not isinstance(parsed, dict):
        raise ValueError("No JSON object found in annotation output")
    return normalize_annotation(parsed)


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in ANNOTATION_FIELDS if field not in annotation]
    if missing:
        raise ValueError(f"Annotation is missing fields: {missing}")

    normalized: dict[str, Any] = {}
    for field in ("rejected_x", "asserted_y", "notes"):
        value = str(annotation[field]).strip()
        if not value:
            raise ValueError(f"{field} must not be empty")
        normalized[field] = value

    label = str(annotation["taxonomy_label"]).strip().lower()
    if label not in TAXONOMY_LABELS:
        raise ValueError(f"Unknown taxonomy label: {label}")
    normalized["taxonomy_label"] = label

    for field in RATING_FIELDS:
        value = annotation[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer from 1 to 5")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer from 1 to 5") from exc
        if not number.is_integer() or not 1 <= number <= 5:
            raise ValueError(f"{field} must be an integer from 1 to 5")
        normalized[field] = int(number)
    return normalized


def validate_annotation_frame(
    annotations: pd.DataFrame,
    expected_example_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    required = {"example_id", *ANNOTATION_FIELDS}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation frame is missing columns: {missing}")
    if annotations["example_id"].duplicated().any():
        raise ValueError("Annotation frame contains duplicate example_id values")

    normalized_rows = []
    for record in annotations.to_dict("records"):
        normalized_rows.append({**record, **normalize_annotation(record)})
    normalized = pd.DataFrame(normalized_rows)

    if expected_example_ids is not None:
        expected = set(expected_example_ids)
        observed = set(normalized["example_id"])
        if expected != observed:
            raise ValueError(
                f"Annotation IDs do not match sample: missing={len(expected - observed)}, "
                f"unexpected={len(observed - expected)}"
            )
    return normalized


def compare_annotations(
    sample: pd.DataFrame,
    annotation_a: pd.DataFrame,
    annotation_b: pd.DataFrame,
) -> pd.DataFrame:
    expected_ids = sample["example_id"].tolist()
    annotation_a = validate_annotation_frame(annotation_a, expected_ids)
    annotation_b = validate_annotation_frame(annotation_b, expected_ids)
    a = annotation_a[["example_id", *ANNOTATION_FIELDS]].rename(
        columns={field: f"a_{field}" for field in ANNOTATION_FIELDS}
    )
    b = annotation_b[["example_id", *ANNOTATION_FIELDS]].rename(
        columns={field: f"b_{field}" for field in ANNOTATION_FIELDS}
    )
    compared = sample.merge(a, on="example_id", validate="one_to_one").merge(
        b, on="example_id", validate="one_to_one"
    )

    def reasons(record: pd.Series) -> str:
        flags = []
        if record["a_taxonomy_label"] != record["b_taxonomy_label"]:
            flags.append("taxonomy")
        if "unclear" in {record["a_taxonomy_label"], record["b_taxonomy_label"]}:
            flags.append("unclear")
        for field in ("prompt_support", "negation_adds_meaning"):
            if abs(int(record[f"a_{field}"]) - int(record[f"b_{field}"])) >= 2:
                flags.append(field)
        return "|".join(flags)

    compared["disagreement_reasons"] = compared.apply(reasons, axis=1)
    compared["adjudication_required"] = compared["disagreement_reasons"].ne("")
    return compared


def agreement_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def safe_kappa(left, right, weights=None) -> float:
        if len(set(left) | set(right)) <= 1:
            return float("nan")
        return float(cohen_kappa_score(left, right, weights=weights))

    left_labels = comparison["a_taxonomy_label"].tolist()
    right_labels = comparison["b_taxonomy_label"].tolist()
    rows.append(
        {
            "field": "taxonomy_label",
            "exact_agreement": float(np.mean(np.array(left_labels) == np.array(right_labels))),
            "within_one_agreement": float("nan"),
            "cohen_kappa": safe_kappa(left_labels, right_labels),
        }
    )
    for field in RATING_FIELDS:
        left = comparison[f"a_{field}"].astype(int).to_numpy()
        right = comparison[f"b_{field}"].astype(int).to_numpy()
        rows.append(
            {
                "field": field,
                "exact_agreement": float(np.mean(left == right)),
                "within_one_agreement": float(np.mean(np.abs(left - right) <= 1)),
                "cohen_kappa": safe_kappa(left.tolist(), right.tolist(), weights="quadratic"),
            }
        )
    return pd.DataFrame(rows)


def finalize_adjudications(
    comparison: pd.DataFrame,
    adjudications: pd.DataFrame,
    adjudicate_all: bool = True,
) -> pd.DataFrame:
    expected = comparison["example_id"] if adjudicate_all else comparison.loc[
        comparison["adjudication_required"], "example_id"
    ]
    adjudications = validate_annotation_frame(adjudications, expected)
    adjudicated = adjudications[["example_id", *ANNOTATION_FIELDS]].rename(
        columns={field: f"adjudicated_{field}" for field in ANNOTATION_FIELDS}
    )
    final = comparison.merge(adjudicated, on="example_id", how="left", validate="one_to_one")

    for field in ANNOTATION_FIELDS:
        adjudicated_field = f"adjudicated_{field}"
        if adjudicate_all:
            final[field] = final[adjudicated_field]
        elif field in RATING_FIELDS:
            consensus = np.floor(
                (final[f"a_{field}"].astype(float) + final[f"b_{field}"].astype(float)) / 2 + 0.5
            ).astype(int)
            final[field] = final[adjudicated_field].fillna(pd.Series(consensus, index=final.index))
        else:
            final[field] = final[adjudicated_field].fillna(final[f"a_{field}"])

    final["annotation_method"] = (
        "oss_panel_full_adjudication" if adjudicate_all else "oss_panel_disagreement_adjudication"
    )
    final["strict_misuse"] = final["taxonomy_label"].isin(STRICT_MISUSE_LABELS)
    final["broad_misuse"] = final["taxonomy_label"].isin(BROAD_MISUSE_LABELS)
    final["unsupported_contrast"] = (
        final["prompt_support"].astype(int).le(2)
        & ~final["taxonomy_label"].isin(
            {"legitimate_pedagogy", "genuine_contrast", "non_ocn_negation"}
        )
    )
    final["adjudication_status"] = np.where(
        final["adjudication_required"], "resolved_disagreement", "confirmed_or_refined"
    )
    return final


def weighted_rate(frame: pd.DataFrame, outcome: str, weight: str = "sample_weight") -> float:
    values = frame[outcome].astype(float).to_numpy()
    weights = frame[weight].astype(float).to_numpy()
    if not len(values) or math.isclose(float(weights.sum()), 0.0):
        return float("nan")
    return float(np.average(values, weights=weights))

