from __future__ import annotations

from statistics import NormalDist

import numpy as np
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


def deduplicate_greedy_seed_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse only deterministic greedy rows duplicated across bookkeeping seeds."""
    required = {"prompt_id", "model_id", "decoding", "response"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Generation table is missing required columns: {missing}")

    greedy_mask = df["decoding"].astype(str).str.lower().eq("greedy")
    greedy = df[greedy_mask].copy()
    sampled = df[~greedy_mask].copy()
    keys = ["prompt_id", "model_id", "decoding", "response"]
    if "seed" in greedy.columns:
        greedy = greedy.sort_values(["prompt_id", "model_id", "decoding", "seed"])
    greedy = greedy.drop_duplicates(keys, keep="first")
    result = pd.concat([greedy, sampled], ignore_index=True)
    sort_columns = [
        column
        for column in ["model_id", "decoding", "prompt_id", "seed"]
        if column in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(center - margin), float(center + margin)


def grouped_ocn_rates_with_ci(
    df: pd.DataFrame,
    group_columns: list[str],
    alpha: float = 0.05,
    cluster_column: str | None = None,
    n_boot: int = 2000,
    seed: int = 20260819,
) -> pd.DataFrame:
    rates = grouped_ocn_rates(df, group_columns)
    if cluster_column is not None:
        bootstrap_frame = df.copy()
        bootstrap_frame["_unit_weight"] = 1.0
        bootstrap_frame["_has_ocn_binary"] = bootstrap_frame["has_ocn"].astype(int)
        intervals = weighted_group_bootstrap(
            bootstrap_frame,
            outcomes=["_has_ocn_binary"],
            group_columns=group_columns,
            cluster_column=cluster_column,
            weight_column="_unit_weight",
            n_boot=n_boot,
            seed=seed,
        )
        intervals = intervals.rename(
            columns={
                "_has_ocn_binary_ci_low": "ocn_rate_ci_low",
                "_has_ocn_binary_ci_high": "ocn_rate_ci_high",
            }
        )
        return rates.merge(
            intervals[[*group_columns, "ocn_rate_ci_low", "ocn_rate_ci_high"]],
            on=group_columns,
            validate="one_to_one",
        )

    intervals = [
        wilson_interval(int(successes), int(total), alpha=alpha)
        for successes, total in zip(rates["responses_with_ocn"], rates["responses"])
    ]
    rates["ocn_rate_ci_low"] = [interval[0] for interval in intervals]
    rates["ocn_rate_ci_high"] = [interval[1] for interval in intervals]
    return rates


def weighted_group_bootstrap(
    frame: pd.DataFrame,
    outcomes: list[str],
    group_columns: list[str],
    cluster_column: str = "response_id",
    weight_column: str = "sample_weight",
    n_boot: int = 2000,
    seed: int = 20260819,
) -> pd.DataFrame:
    """Cluster bootstrap weighted binary rates within reporting groups."""
    required = {cluster_column, weight_column, *outcomes, *group_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Bootstrap frame is missing required columns: {missing}")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    group_key = group_columns if len(group_columns) > 1 else group_columns[0]
    for key, group in frame.groupby(group_key, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        clusters = group[cluster_column].drop_duplicates().to_numpy()
        weights = group[weight_column].astype(float).to_numpy()
        row = {
            **dict(zip(group_columns, key_values)),
            "clusters": len(clusters),
            "rows": len(group),
        }
        for outcome in outcomes:
            values = group[outcome].astype(float).to_numpy()
            row[f"{outcome}_rate"] = float(np.average(values, weights=weights))

        cluster_codes = pd.Categorical(group[cluster_column], categories=clusters).codes
        cluster_weights = np.bincount(cluster_codes, weights=weights)
        sampled_codes = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        sampled_denominators = cluster_weights[sampled_codes].sum(axis=1)
        draws = {}
        for outcome in outcomes:
            weighted_values = group[outcome].astype(float).to_numpy() * weights
            cluster_numerators = np.bincount(cluster_codes, weights=weighted_values)
            sampled_numerators = cluster_numerators[sampled_codes].sum(axis=1)
            draws[outcome] = sampled_numerators / sampled_denominators
        for outcome in outcomes:
            row[f"{outcome}_ci_low"] = float(np.quantile(draws[outcome], 0.025))
            row[f"{outcome}_ci_high"] = float(np.quantile(draws[outcome], 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


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
