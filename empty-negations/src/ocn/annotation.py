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

ANNOTATION_PROMPT_VERSION = "v2_calibrated"

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
- genuine_contrast: the USER PROMPT, before the model response begins, explicitly supplies or strongly implies the rejected proposition X. Do not count the response's own mention of X as prompt support.
- legitimate_pedagogy: X is a widely recognized factual misconception whose correction is useful here. A merely simplistic or narrow view is not automatically a common misconception.
- presupposed_contrast: X is plausible but was not introduced by the prompt.
- empty_intensification: X and Y are near paraphrases or the contrast adds no substance.
- scope_inflation: Y merely broadens X rather than genuinely contrasting with it.
- false_correction: X is an implausible straw position or unmotivated correction.
- template_stacking: several formulaic rhetorical templates cluster in the response and that stacking is the dominant issue. Do not use this label for one isolated phrase when a more specific relation applies.
- non_ocn_negation: this is ordinary factual negation, not rhetorical contrastive negation.
- unclear: evidence is genuinely insufficient or mixed.
""".strip()

_DECISION_ORDER = """
Apply this order and stop at the first clearly satisfied rule:
1. Ordinary factual negation with no rhetorical X-to-Y upgrade -> non_ocn_negation.
2. The user prompt itself supplies X -> genuine_contrast.
3. X is a documented, widely known factual misconception -> legitimate_pedagogy.
4. X and Y are near paraphrases -> empty_intensification.
5. Y includes, expands, or lists consequences of X without incompatibility -> scope_inflation.
6. X is implausible or unmotivated -> false_correction.
7. Multiple formulaic templates dominate the passage -> template_stacking.
8. X is plausible but unprompted -> presupposed_contrast.
9. Otherwise -> unclear.
""".strip()

_CALIBRATION_ANCHORS = """
Calibration anchors:

A. Prompt: "Explain photosynthesis." Response span: "Photosynthesis not only fuels plant growth but also releases oxygen." Label: scope_inflation. The user did not supply X, and Y broadens the effects of the process.

B. Prompt: "Some people say a library is only a warehouse for books. Explain its wider role." Response span: "A library is not just a warehouse for books; it is a civic learning space." Label: genuine_contrast. The user explicitly supplied X.

C. Prompt: "Explain evolution to a beginner." Response span: "Evolution is not a march toward perfection; it is change in inherited traits across generations." Label: legitimate_pedagogy. The rejected teleological view is a widely recognized misconception.

D. Prompt: "Report the server status." Response span: "The server is not running." Label: non_ocn_negation. This is factual negation without a rhetorical upgrade.

E. Prompt: "Describe leadership." Response span: "Leadership is not just guiding people; it is helping people find direction." Label: empty_intensification. X and Y are near paraphrases.

F. Prompt: "Describe a museum." Response span: "A museum is more than just a building; it is a place of memory and interpretation." Label: presupposed_contrast. The narrow view is plausible but was not supplied by the user.
""".strip()


def annotation_calibration_frame() -> pd.DataFrame:
    """Return held-out boundary cases for reporting prompt calibration accuracy."""
    rows = [
        {
            "example_id": "calibration_genuine",
            "prompt": "A manager claims remote work is only a cost-cutting tool. Explain what that misses.",
            "response": "Remote work is not merely a cost-cutting tool; it also changes coordination, hiring, and employee autonomy.",
            "span_text": "not merely a cost-cutting tool; it also",
            "expected_taxonomy_label": "genuine_contrast",
        },
        {
            "example_id": "calibration_legitimate",
            "prompt": "Explain how antibiotics work.",
            "response": "Antibiotics are not a cure for viral infections; they target susceptible bacteria.",
            "span_text": "not a cure for viral infections; they target susceptible bacteria",
            "expected_taxonomy_label": "legitimate_pedagogy",
        },
        {
            "example_id": "calibration_scope",
            "prompt": "Describe the value of public parks.",
            "response": "A public park is not just green space; it also supports exercise, cooling, and community events.",
            "span_text": "not just green space; it also",
            "expected_taxonomy_label": "scope_inflation",
        },
        {
            "example_id": "calibration_empty",
            "prompt": "Define mentoring.",
            "response": "Mentoring is not merely offering guidance; it is guiding someone through growth.",
            "span_text": "not merely offering guidance; it is guiding someone",
            "expected_taxonomy_label": "empty_intensification",
        },
        {
            "example_id": "calibration_false",
            "prompt": "Describe a basic calendar application.",
            "response": "A calendar app is not just a grid of dates; it is a declaration that time can be conquered.",
            "span_text": "not just a grid of dates; it is a declaration that time can be conquered",
            "expected_taxonomy_label": "false_correction",
        },
        {
            "example_id": "calibration_template",
            "prompt": "Describe customer support software.",
            "response": "It is not just a ticketing tool. It goes beyond efficiency to create deeply human connections, trust, and agency.",
            "span_text": "not just a ticketing tool. It goes beyond efficiency",
            "expected_taxonomy_label": "template_stacking",
        },
        {
            "example_id": "calibration_non_ocn",
            "prompt": "State whether the sample contains lead.",
            "response": "The sample does not contain detectable lead.",
            "span_text": "does not contain detectable lead",
            "expected_taxonomy_label": "non_ocn_negation",
        },
        {
            "example_id": "calibration_presupposed",
            "prompt": "Describe a university.",
            "response": "A university is more than just classrooms; it is a community for research and public service.",
            "span_text": "more than just classrooms; it is a community",
            "expected_taxonomy_label": "presupposed_contrast",
        },
    ]
    return pd.DataFrame(rows)


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

Decision procedure:
{_DECISION_ORDER}

{_CALIBRATION_ANCHORS}

Use integer ratings where 1=definitely no/absent, 2=probably no, 3=unclear or mixed, 4=probably yes, and 5=definitely yes/strong.

PROMPT:
{record['prompt']}

FULL RESPONSE:
{record['response']}

DETECTED SPAN:
{record['span_text']}

Critical boundary checks:
- `prompt_support` considers only the USER PROMPT, never wording introduced inside the model response.
- `common_misconception` is high only for a widely recognized factual misconception, not any shallow interpretation.
- Choose one dominant taxonomy label using the decision order, even if secondary stylistic issues are present.

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

Decision procedure:
{_DECISION_ORDER}

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
    parse_repairs: list[str] = []
    for match in re.finditer(r"\{", cleaned):
        try:
            candidate, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate.get("annotation", candidate)
            break
    if not isinstance(parsed, dict):
        trailing_comma_repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
        if trailing_comma_repaired != cleaned:
            try:
                candidate = json.loads(trailing_comma_repaired)
                if isinstance(candidate, dict):
                    parsed = candidate.get("annotation", candidate)
                    parse_repairs.append("trailing_comma")
            except json.JSONDecodeError:
                pass
    if not isinstance(parsed, dict):
        try:
            from json_repair import loads as load_repaired_json

            candidate = load_repaired_json(cleaned)
            if isinstance(candidate, dict):
                parsed = candidate.get("annotation", candidate)
                parse_repairs.append("json_repair")
        except Exception:
            pass
    if not isinstance(parsed, dict):
        raise ValueError("No JSON object found in annotation output")
    normalized = normalize_annotation(parsed)
    rating_repairs = [
        field
        for field in RATING_FIELDS
        if str(parsed.get(field, "")).strip() in {"0", "0.0"}
    ]
    if rating_repairs:
        parse_repairs.append("zero_to_one:" + "|".join(rating_repairs))
    normalized["annotation_parse_repairs"] = ";".join(parse_repairs)
    return normalized


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in ANNOTATION_FIELDS if field not in annotation]
    if missing:
        raise ValueError(f"Annotation is missing fields: {missing}")

    normalized: dict[str, Any] = {}
    for field in ("rejected_x", "asserted_y"):
        value = str(annotation[field]).strip()
        normalized[field] = "" if value.lower() in {"nan", "none", "null"} else value

    for field in ("notes",):
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
        if number == 0:
            number = 1.0
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
    final = add_semantic_outcomes(final)
    final["adjudication_status"] = np.where(
        final["adjudication_required"], "resolved_disagreement", "confirmed_or_refined"
    )
    return final


def add_semantic_outcomes(
    frame: pd.DataFrame,
    taxonomy_column: str = "taxonomy_label",
    prompt_support_column: str = "prompt_support",
) -> pd.DataFrame:
    result = frame.copy()
    labels = result[taxonomy_column].astype(str)
    result["strict_misuse"] = labels.isin(STRICT_MISUSE_LABELS)
    result["broad_misuse"] = labels.isin(BROAD_MISUSE_LABELS)
    result["unsupported_contrast"] = (
        result[prompt_support_column].astype(int).le(2)
        & ~labels.isin({"legitimate_pedagogy", "genuine_contrast", "non_ocn_negation"})
    )
    return result


def weighted_rate(frame: pd.DataFrame, outcome: str, weight: str = "sample_weight") -> float:
    values = frame[outcome].astype(float).to_numpy()
    weights = frame[weight].astype(float).to_numpy()
    if not len(values) or math.isclose(float(weights.sum()), 0.0):
        return float("nan")
    return float(np.average(values, weights=weights))
