from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re

import pandas as pd

from ocn.detectors import OCNDetector, approximate_token_count


@dataclass(frozen=True)
class PropositionSet:
    question_id: str
    topic: str
    plain_subject: str
    point_a: str
    point_b: str


def build_variants(item: PropositionSet) -> dict[str, str]:
    """Create controlled plain and OCN variants from shared propositions."""
    point_a = _phrase(item.point_a)
    point_b = _phrase(item.point_b)
    return {
        "plain": f"{item.plain_subject} {point_a}. It also {point_b}.",
        "justified_ocn": (
            f"{item.plain_subject} is not only about {point_a}; it also {point_b}."
        ),
        "empty_ocn": (
            f"{item.plain_subject} is not merely about {point_a}; it concerns {point_a}."
        ),
        "explicit_genuine_contrast": (
            f"Although the prompt frames {item.topic} around {point_a}, it also {point_b}."
        ),
    }


def variants_to_frame(items: list[PropositionSet], shuffle: bool = True, seed: int = 0) -> pd.DataFrame:
    rows = []
    rng = random.Random(seed)
    for item in items:
        variants = build_variants(item)
        order = list(variants)
        if shuffle:
            rng.shuffle(order)
        for position, variant_name in enumerate(order):
            rows.append(
                {
                    "question_id": item.question_id,
                    "topic": item.topic,
                    "variant_position": position,
                    "variant_type": variant_name,
                    "response": variants[variant_name],
                }
            )
    return pd.DataFrame(rows)


REWRITE_SYSTEM_INSTRUCTION = """Rewrite the response using direct affirmative prose.

Requirements:
- Preserve every substantive claim, name, number, qualification, and conclusion.
- Remove contrastive-negation rhetoric such as "not just", "not only", "more than just", "goes beyond", and "rather than simply".
- Do not add facts, commentary, headings, or an explanation of the edit.
- Keep the tone and approximate length of the original.
- Return only the rewritten response.
"""

_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "in", "is", "it", "its", "of", "on", "or", "our", "ours", "she", "that", "the",
    "their", "theirs", "them", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "you", "your", "yours",
}


def make_plain_rewrite_prompt(response: str) -> str:
    """Build the controlled rewrite instruction for one detected response."""
    return f"{REWRITE_SYSTEM_INSTRUCTION}\n<response>\n{response.strip()}\n</response>"


def normalize_plain_rewrite(text: str | None) -> str:
    """Remove common wrapper text without changing the generated prose."""
    value = re.sub(r"\s+", " ", (text or "").strip())
    value = re.sub(
        r"^(?:rewritten response|rewrite|direct rewrite)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def select_unique_ocn_candidates(
    detections: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select stable, non-duplicated lexical OCN candidates for rewriting."""
    required = {"prompt_id", "model_id", "decoding", "seed", "response", "has_ocn"}
    missing = sorted(required - set(detections.columns))
    if missing:
        raise ValueError(f"Detection dataset is missing required columns: {missing}")

    candidates = detections[
        detections["has_ocn"].astype(bool)
        & detections["response"].fillna("").str.strip().ne("")
    ].copy()
    candidates = candidates.sort_values(
        ["model_id", "prompt_id", "decoding", "seed"],
        kind="stable",
    )
    candidates = candidates.drop_duplicates(
        ["model_id", "prompt_id", "decoding", "response"],
        keep="first",
    )
    candidates["source_candidate_id"] = [
        _candidate_id(row) for row in candidates.to_dict("records")
    ]
    if limit is not None:
        candidates = candidates.head(limit)
    return candidates.reset_index(drop=True)


def assess_plain_rewrite(
    source_response: str,
    plain_response: str | None,
    detector: OCNDetector | None = None,
    min_length_ratio: float = 0.60,
    max_length_ratio: float = 1.35,
    min_content_overlap: float = 0.55,
) -> dict[str, object]:
    """Evaluate whether a rewrite is usable as a content-controlled plain variant."""
    detector = detector or OCNDetector()
    plain_response = normalize_plain_rewrite(plain_response)
    source_tokens = approximate_token_count(source_response)
    plain_tokens = approximate_token_count(plain_response)
    length_ratio = plain_tokens / max(source_tokens, 1)
    content_overlap = _content_overlap(source_response, plain_response)
    source_has_ocn = detector.detect(source_response).has_ocn
    plain_has_ocn = detector.detect(plain_response).has_ocn
    is_distinct = _normalized_text(source_response) != _normalized_text(plain_response)
    quality_pass = bool(
        source_has_ocn
        and plain_response
        and not plain_has_ocn
        and is_distinct
        and min_length_ratio <= length_ratio <= max_length_ratio
        and content_overlap >= min_content_overlap
    )
    return {
        "plain_response": plain_response,
        "source_has_ocn": source_has_ocn,
        "plain_has_ocn": plain_has_ocn,
        "source_tokens_approx": source_tokens,
        "plain_tokens_approx": plain_tokens,
        "length_ratio": length_ratio,
        "content_overlap": content_overlap,
        "is_distinct": is_distinct,
        "quality_pass": quality_pass,
    }


def build_counterfactual_pair_frame(
    rewrites: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form blinded reward pairs and return pair rows plus rewrite QA."""
    required = {"source_candidate_id", "response", "plain_response"}
    missing = sorted(required - set(rewrites.columns))
    if missing:
        raise ValueError(f"Rewrite table is missing required columns: {missing}")

    detector = OCNDetector()
    quality_rows = []
    for record in rewrites.to_dict("records"):
        quality_rows.append(
            {
                **record,
                **assess_plain_rewrite(
                    source_response=str(record["response"]),
                    plain_response=record["plain_response"],
                    detector=detector,
                ),
            }
        )
    quality = pd.DataFrame(quality_rows)

    pair_rows = []
    source_detection_columns = {
        "has_ocn",
        "ocn_count",
        "ocn_patterns",
        "ocn_spans",
        "response_tokens_approx",
        "ocn_per_1k_tokens",
    }
    for record in quality[quality["quality_pass"]].to_dict("records"):
        pair_id = f"rp-{record['source_candidate_id']}"
        order = ["candidate_ocn", "plain_rewrite"]
        random.Random(f"{seed}:{pair_id}").shuffle(order)
        positions = {variant: position for position, variant in enumerate(order)}
        shared = {
            key: value
            for key, value in record.items()
            if key not in source_detection_columns | {"response", "plain_response"}
        }
        shared.update(
            {
                f"source_{key}": record.get(key)
                for key in source_detection_columns
                if key in record
            }
        )
        for variant_type, response in (
            ("candidate_ocn", record["response"]),
            ("plain_rewrite", record["plain_response"]),
        ):
            position = positions[variant_type]
            pair_rows.append(
                {
                    **shared,
                    "pair_id": pair_id,
                    "question_id": pair_id,
                    "variant_type": variant_type,
                    "presentation_position": position,
                    "presentation_label": "A" if position == 0 else "B",
                    "response": response,
                }
            )

    if pair_rows:
        pairs = OCNDetector().annotate_rows(
            pd.DataFrame(pair_rows), text_column="response"
        )
        pairs = pairs.sort_values(
            ["pair_id", "presentation_position"], kind="stable"
        ).reset_index(drop=True)
    else:
        pairs = pd.DataFrame(
            columns=[
                "pair_id",
                "question_id",
                "variant_type",
                "presentation_position",
                "presentation_label",
                "response",
                "has_ocn",
                "ocn_count",
                "ocn_patterns",
                "ocn_spans",
                "response_tokens_approx",
                "ocn_per_1k_tokens",
            ]
        )
    return pairs, quality


def starter_proposition_sets() -> list[PropositionSet]:
    return [
        PropositionSet(
            question_id="r001",
            topic="payment accuracy project",
            plain_subject="The project",
            point_a="Improved payment accuracy.",
            point_b="gave teams better visibility into discrepancies.",
        ),
        PropositionSet(
            question_id="r002",
            topic="climate policy",
            plain_subject="The policy",
            point_a="Reduces emissions.",
            point_b="affects industrial investment and energy security.",
        ),
        PropositionSet(
            question_id="r003",
            topic="data privacy program",
            plain_subject="The program",
            point_a="Protects customer information.",
            point_b="builds trust across product and compliance teams.",
        ),
    ]


def _phrase(text: str) -> str:
    text = text.strip().rstrip(".")
    if not text:
        return text
    return text[0].lower() + text[1:]


def _candidate_id(record: dict) -> str:
    identity = "\n".join(
        str(record.get(key, ""))
        for key in ("prompt_id", "model_id", "decoding", "seed", "response")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _content_overlap(source: str, rewrite: str) -> float:
    source_tokens = _content_tokens(source)
    rewrite_tokens = _content_tokens(rewrite)
    if not source_tokens or not rewrite_tokens:
        return 0.0
    overlap = len(source_tokens & rewrite_tokens)
    precision = overlap / len(rewrite_tokens)
    recall = overlap / len(source_tokens)
    return (2 * precision * recall) / max(precision + recall, 1e-12)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _CONTENT_STOPWORDS and len(token) > 1
    }


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
