import pandas as pd

from ocn.metrics import (
    deduplicate_greedy_seed_reuse,
    grouped_ocn_rates_with_ci,
    weighted_group_bootstrap,
    wilson_interval,
)


def test_deduplicate_greedy_seed_reuse_preserves_sampled_repetitions():
    rows = []
    for decoding in ["greedy", "normal_temp"]:
        for seed in [1, 2]:
            rows.append(
                {
                    "prompt_id": "p1",
                    "model_id": "m1",
                    "decoding": decoding,
                    "seed": seed,
                    "response": "same response",
                    "has_ocn": True,
                    "ocn_count": 1,
                    "response_tokens_approx": 10,
                }
            )

    deduplicated = deduplicate_greedy_seed_reuse(pd.DataFrame(rows))

    assert len(deduplicated) == 3
    assert len(deduplicated[deduplicated["decoding"].eq("greedy")]) == 1
    assert len(deduplicated[deduplicated["decoding"].eq("normal_temp")]) == 2


def test_wilson_interval_and_grouped_rates_are_bounded():
    low, high = wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1

    frame = pd.DataFrame(
        {
            "model_id": ["m1"] * 4,
            "has_ocn": [True, False, True, False],
            "ocn_count": [1, 0, 1, 0],
            "response_tokens_approx": [10] * 4,
        }
    )
    rates = grouped_ocn_rates_with_ci(frame, ["model_id"])
    assert rates.loc[0, "ocn_rate"] == 0.5
    assert rates.loc[0, "ocn_rate_ci_low"] < 0.5 < rates.loc[0, "ocn_rate_ci_high"]


def test_grouped_rates_can_cluster_intervals_by_prompt():
    frame = pd.DataFrame(
        {
            "model_id": ["m1"] * 6,
            "prompt_id": ["p1", "p1", "p1", "p2", "p2", "p3"],
            "has_ocn": [True, True, False, False, False, True],
            "ocn_count": [1, 1, 0, 0, 0, 1],
            "response_tokens_approx": [10] * 6,
        }
    )
    rates = grouped_ocn_rates_with_ci(
        frame,
        ["model_id"],
        cluster_column="prompt_id",
        n_boot=100,
        seed=9,
    )

    assert rates.loc[0, "ocn_rate"] == 0.5
    assert 0 <= rates.loc[0, "ocn_rate_ci_low"] <= 0.5
    assert 0.5 <= rates.loc[0, "ocn_rate_ci_high"] <= 1


def test_weighted_group_bootstrap_is_reproducible_and_contains_point_estimate():
    frame = pd.DataFrame(
        {
            "model_id": ["m1"] * 4,
            "response_id": ["r1", "r1", "r2", "r3"],
            "sample_weight": [2.0, 2.0, 1.0, 1.0],
            "strict_misuse": [True, False, True, False],
        }
    )
    first = weighted_group_bootstrap(
        frame,
        outcomes=["strict_misuse"],
        group_columns=["model_id"],
        n_boot=100,
        seed=7,
    )
    second = weighted_group_bootstrap(
        frame,
        outcomes=["strict_misuse"],
        group_columns=["model_id"],
        n_boot=100,
        seed=7,
    )

    pd.testing.assert_frame_equal(first, second)
    point = first.loc[0, "strict_misuse_rate"]
    assert first.loc[0, "strict_misuse_ci_low"] <= point
    assert point <= first.loc[0, "strict_misuse_ci_high"]
