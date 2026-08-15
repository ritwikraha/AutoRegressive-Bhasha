from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re

import pandas as pd


@dataclass(frozen=True)
class TopicSpec:
    topic: str
    category: str
    base_question: str
    misconception: str | None = None


DEFAULT_TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec("photosynthesis", "factual_explanation", "Explain photosynthesis."),
    TopicSpec("printing_press", "historical_significance", "Why was the printing press important?", "The printing press was only a faster copying device."),
    TopicSpec("leadership", "concept_definition", "What is leadership?"),
    TopicSpec("career_uncertainty", "personal_advice", "How should someone handle career uncertainty?"),
    TopicSpec("budgeting_app", "product_description", "Describe the value of a budgeting app."),
    TopicSpec("great_gatsby_green_light", "literary_analysis", "What does the green light represent in The Great Gatsby?"),
    TopicSpec("crispr", "scientific_interpretation", "Why is CRISPR important?", "CRISPR is simply a pair of molecular scissors."),
    TopicSpec("process_improvement", "business_writing", "Explain the impact of a process improvement."),
    TopicSpec("sincere_apology", "moral_question", "What makes an apology sincere?"),
    TopicSpec("router_reset", "procedural", "Explain how to reset a router."),
    TopicSpec("cities", "creative_writing", "Write an introduction about cities."),
    TopicSpec("data_privacy", "business_writing", "Explain why data privacy matters for a company."),
    TopicSpec("virus", "scientific_interpretation", "Explain what a virus is.", "A virus is a kind of cell."),
    TopicSpec("democracy", "concept_definition", "What is democracy?"),
    TopicSpec("api_design", "technical_documentation", "Explain why API design matters."),
    TopicSpec("remote_work", "business_writing", "Explain the impact of remote work on organizations."),
)

VARIANT_INSTRUCTIONS: dict[str, str] = {
    "neutral": "",
    "brief": "Answer briefly.",
    "comprehensive": "Give a comprehensive answer.",
    "nuanced": "Give a deep and nuanced answer.",
    "plain_factual": "Use plain factual language. Avoid rhetorical framing.",
    "prohibit_ocn": "Do not use contrastive frames such as not just, not only, merely, simply, more than just, or goes beyond.",
    "explicit_misconception": "A prior speaker said: \"{misconception}\" Correct the claim carefully.",
}

PERSONA_PREFIXES: dict[str, str] = {
    "assistant": "You are a helpful assistant.",
    "encyclopedia": "Write in the style of a concise encyclopedia entry.",
    "technical_manual": "Write in the style of a technical manual.",
    "terse_analyst": "Write as a terse analyst using direct claims.",
    "casual_human": "Write like a casual but informed person.",
    "marketing_copywriter": "Write as a polished marketing copywriter.",
}


def build_prompt_dataset(
    topics: tuple[TopicSpec, ...] = DEFAULT_TOPICS,
    variants: tuple[str, ...] = (
        "neutral",
        "brief",
        "comprehensive",
        "nuanced",
        "plain_factual",
        "prohibit_ocn",
        "explicit_misconception",
    ),
    personas: tuple[str, ...] = ("assistant", "encyclopedia", "terse_analyst"),
    length_targets: tuple[int, ...] = (75, 150),
) -> pd.DataFrame:
    rows = []
    counter = 1
    for topic, variant, persona, length_target in product(
        topics, variants, personas, length_targets
    ):
        if variant == "explicit_misconception" and not topic.misconception:
            continue
        prompt = compose_prompt(topic, variant, persona, length_target)
        rows.append(
            {
                "prompt_id": f"ocn_{counter:05d}",
                "topic": topic.topic,
                "category": topic.category,
                "variant": variant,
                "persona": persona,
                "length_target": length_target,
                "contrast_availability": (
                    "explicit_misconception"
                    if variant == "explicit_misconception"
                    else "no_misconception"
                ),
                "prompt": prompt,
            }
        )
        counter += 1
    return pd.DataFrame(rows)


def compose_prompt(topic: TopicSpec, variant: str, persona: str, length_target: int) -> str:
    prefix = PERSONA_PREFIXES[persona]
    instruction = VARIANT_INSTRUCTIONS[variant].format(
        misconception=topic.misconception or ""
    )
    parts = [
        prefix,
        f"Target length: about {length_target} words.",
        instruction,
        topic.base_question,
    ]
    return "\n".join(part for part in parts if part)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-")
