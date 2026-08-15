from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class OCNSpan:
    """One lexical match for a contrastive negation construction."""

    pattern_name: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class DetectionResult:
    text: str
    spans: tuple[OCNSpan, ...]

    @property
    def has_ocn(self) -> bool:
        return bool(self.spans)

    @property
    def count(self) -> int:
        return len(self.spans)


class OCNDetector:
    """Lexical detector for contrastive-negation constructions.

    This intentionally detects candidate spans, not pragmatic misuse. Human or
    semantic annotation is still needed to distinguish valid contrast from OCN.
    """

    DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
        ("not_just_but", r"\bnot\s+(?:just|only|merely|simply)\b.{0,160}?\bbut(?:\s+also)?\b"),
        ("not_just_clause_shift", r"\bnot\s+(?:just|only|merely|simply)\b.{0,120}?[;.!?]\s*(?:it|this|that|they|he|she|we)\s+(?:also\s+)?[a-z]+\b.{0,120}"),
        ("isnt_just_but", r"\b(?:isn['’]?t|is\s+not)\s+(?:just|only|merely|simply)\b.{0,160}?\bbut(?:\s+also)?\b"),
        ("isnt_just_clause_shift", r"\b(?:isn['’]?t|is\s+not)\s+(?:just|only|merely|simply)\b.{0,120}?[;.!?]\s*(?:it|this|that|they|he|she|we)\s+(?:also\s+)?[a-z]+\b.{0,120}"),
        ("doesnt_just_but", r"\b(?:doesn['’]?t|does\s+not)\s+(?:just|only|merely|simply)\b.{0,160}?\bbut(?:\s+also)?\b"),
        ("goes_beyond", r"\bgo(?:es|ing)?\s+beyond\b.{0,140}"),
        ("more_than_just", r"\bmore\s+than\s+(?:just|merely|simply)\b.{0,140}"),
        ("rather_than_simply", r"\brather\s+than\s+(?:just|merely|simply|only)\b.{0,140}"),
        ("not_so_much_as", r"\bnot\s+so\s+much\b.{0,160}?\bas\b"),
        ("not_x_alone", r"\bnot\b.{0,80}?\balone\b.{0,120}?\bbut\b"),
        ("far_from_merely", r"\bfar\s+from\s+being\s+(?:just|merely|simply|only)\b.{0,140}"),
        ("cannot_be_reduced", r"\b(?:cannot|can['’]?t)\s+be\s+reduced\s+to\b.{0,140}"),
        ("not_matter_of", r"\bnot\s+(?:simply|merely|just|only)?\s*(?:a\s+)?matter\s+of\b.{0,140}"),
    )

    def __init__(self, patterns: Iterable[tuple[str, str]] | None = None) -> None:
        source = patterns if patterns is not None else self.DEFAULT_PATTERNS
        self.patterns = tuple(
            (name, re.compile(pattern, flags=re.IGNORECASE | re.DOTALL))
            for name, pattern in source
        )

    def detect(self, text: str | None) -> DetectionResult:
        if not text:
            return DetectionResult(text="", spans=())

        spans: list[OCNSpan] = []
        for name, pattern in self.patterns:
            for match in pattern.finditer(text):
                spans.append(
                    OCNSpan(
                        pattern_name=name,
                        start=match.start(),
                        end=match.end(),
                        text=_trim_span(match.group(0)),
                    )
                )

        deduped = _dedupe_overlaps(spans)
        return DetectionResult(text=text, spans=tuple(deduped))

    def annotate_rows(self, rows, text_column: str = "response"):
        """Return a pandas DataFrame with detection columns added."""
        import pandas as pd

        df = pd.DataFrame(rows).copy()
        results = [self.detect(value) for value in df[text_column].fillna("")]
        df["has_ocn"] = [result.has_ocn for result in results]
        df["ocn_count"] = [result.count for result in results]
        df["ocn_patterns"] = [
            "|".join(span.pattern_name for span in result.spans) for result in results
        ]
        df["ocn_spans"] = [
            " || ".join(span.text for span in result.spans) for result in results
        ]
        df["response_tokens_approx"] = [
            approximate_token_count(value) for value in df[text_column].fillna("")
        ]
        df["ocn_per_1k_tokens"] = [
            (count / max(tokens, 1)) * 1000
            for count, tokens in zip(df["ocn_count"], df["response_tokens_approx"])
        ]
        return df


def approximate_token_count(text: str) -> int:
    """Cheap tokenizer-independent count for normalized rates."""
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def _trim_span(span: str) -> str:
    span = re.sub(r"\s+", " ", span.strip())
    sentence_end = re.search(r"(?<=[.!?;])\s", span)
    if sentence_end:
        return span[: sentence_end.start() + 1]
    return span


def _dedupe_overlaps(spans: Iterable[OCNSpan]) -> list[OCNSpan]:
    ordered = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
    kept: list[OCNSpan] = []
    for span in ordered:
        if any(_overlap(span, existing) for existing in kept):
            continue
        kept.append(span)
    return kept


def _overlap(a: OCNSpan, b: OCNSpan) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)
