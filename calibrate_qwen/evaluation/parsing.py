"""Conservative parsing for CalibrateQwen structured completions."""

from __future__ import annotations

import json
import re
from typing import Any

BUCKET_CONFIDENCE = {"low": 0.25, "medium": 0.65, "high": 0.9}


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
    elif isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in BUCKET_CONFIDENCE:
            return BUCKET_CONFIDENCE[stripped]
        percent = stripped.endswith("%")
        try:
            confidence = float(stripped.rstrip("%"))
        except ValueError:
            return None
        if percent:
            confidence /= 100.0
    else:
        return None
    return confidence if 0.0 <= confidence <= 1.0 else None


def parse_completion(
    text: str,
    allowed_labels: set[str],
    *,
    require_confidence: bool = True,
) -> dict[str, Any]:
    parsed = extract_first_json_object(text)
    answer: str | None = None
    confidence: float | None = None
    justification = ""
    abstain: bool | None = None
    parse_method = "invalid"

    if parsed is not None:
        candidate = str(parsed.get("answer", "")).strip().upper()
        answer = candidate if candidate in allowed_labels else None
        confidence = normalize_confidence(parsed.get("confidence"))
        justification = " ".join(str(parsed.get("justification", "")).split())
        abstain_value = parsed.get("abstain")
        abstain = abstain_value if isinstance(abstain_value, bool) else None
        parse_method = "json"

    if answer is None:
        match = re.search(r"(?i)\banswer\s*[:=]\s*[\"']?([A-Z])\b", text)
        if match and match.group(1).upper() in allowed_labels:
            answer = match.group(1).upper()
            parse_method = "fallback_regex"

    required_fields_valid = bool(answer and justification and abstain is not None)
    if require_confidence:
        required_fields_valid = required_fields_valid and confidence is not None
    return {
        "answer": answer,
        "confidence": confidence,
        "justification": justification,
        "abstain": abstain,
        "json_valid": parsed is not None,
        "schema_valid": required_fields_valid,
        "parse_method": parse_method,
        "raw_text": text,
    }
