from __future__ import annotations

from dataclasses import dataclass
import random

import pandas as pd


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
