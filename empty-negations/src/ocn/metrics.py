from __future__ import annotations

import pandas as pd


def detection_summary(df: pd.DataFrame) -> pd.Series:
    """Return top-line OCN metrics for a scored generation table."""
    response_count = len(df)
    ocn_count = int(df["ocn_count"].sum()) if response_count else 0
    has_ocn = int(df["has_ocn"].sum()) if response_count else 0
    token_count = int(df["response_tokens_approx"].sum()) if response_count else 0
    return pd.Series(
        {
            "responses": response_count,
            "responses_with_ocn": has_ocn,
            "ocn_rate": has_ocn / response_count if response_count else 0.0,
            "ocn_constructions": ocn_count,
            "ocn_per_1k_tokens": (ocn_count / max(token_count, 1)) * 1000,
            "approx_tokens": token_count,
        }
    )


def grouped_ocn_rates(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Compute OCN rates by one or more columns."""
    available = [column for column in group_columns if column in df.columns]
    if not available:
        return detection_summary(df).to_frame().T

    grouped = (
        df.groupby(available, dropna=False)
        .apply(detection_summary)
        .reset_index()
        .sort_values(["ocn_rate", "responses"], ascending=[False, False])
    )
    return grouped


def top_patterns(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Count pattern names from the pipe-separated `ocn_patterns` column."""
    if "ocn_patterns" not in df.columns:
        return pd.DataFrame(columns=["pattern", "count"])

    rows: list[str] = []
    for value in df["ocn_patterns"].dropna():
        rows.extend([part for part in str(value).split("|") if part])
    return (
        pd.Series(rows, name="pattern")
        .value_counts()
        .head(n)
        .rename_axis("pattern")
        .reset_index(name="count")
    )
