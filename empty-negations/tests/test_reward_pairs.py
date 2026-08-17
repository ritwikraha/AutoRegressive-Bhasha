import pandas as pd

from ocn.reward_pairs import (
    assess_plain_rewrite,
    build_counterfactual_pair_frame,
    make_plain_rewrite_prompt,
    make_plain_rewrite_retry_prompt,
    normalize_plain_rewrite,
    select_best_plain_rewrites,
    select_unique_ocn_candidates,
)


SOURCE_RESPONSE = (
    "The policy is not just a way to reduce emissions, but also a mechanism "
    "that changes industrial investment and improves long-term energy security."
)
PLAIN_RESPONSE = (
    "The policy reduces emissions, changes industrial investment, and improves "
    "long-term energy security."
)


def test_select_unique_ocn_candidates_deduplicates_greedy_seed_reuse():
    detections = pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "model_id": "model-a",
                "decoding": "greedy",
                "seed": 1,
                "response": SOURCE_RESPONSE,
                "has_ocn": True,
            },
            {
                "prompt_id": "p1",
                "model_id": "model-a",
                "decoding": "greedy",
                "seed": 2,
                "response": SOURCE_RESPONSE,
                "has_ocn": True,
            },
            {
                "prompt_id": "p2",
                "model_id": "model-a",
                "decoding": "greedy",
                "seed": 1,
                "response": "The policy reduces emissions.",
                "has_ocn": False,
            },
        ]
    )

    candidates = select_unique_ocn_candidates(detections)

    assert len(candidates) == 1
    assert candidates.loc[0, "seed"] == 1
    assert len(candidates.loc[0, "source_candidate_id"]) == 20


def test_plain_rewrite_prompt_and_normalization_do_not_add_wrappers():
    prompt = make_plain_rewrite_prompt(SOURCE_RESPONSE)

    assert SOURCE_RESPONSE in prompt
    assert "Return only the rewritten response" in prompt
    assert normalize_plain_rewrite(f'Rewritten response: "{PLAIN_RESPONSE}"') == PLAIN_RESPONSE

    retry_prompt = make_plain_rewrite_retry_prompt(SOURCE_RESPONSE, SOURCE_RESPONSE)
    assert "previous draft did not satisfy" in retry_prompt
    assert "Remove every rhetorical construction" in retry_prompt


def test_assess_plain_rewrite_enforces_detector_and_content_controls():
    accepted = assess_plain_rewrite(SOURCE_RESPONSE, PLAIN_RESPONSE)
    still_ocn = assess_plain_rewrite(SOURCE_RESPONSE, SOURCE_RESPONSE)
    unrelated = assess_plain_rewrite(
        SOURCE_RESPONSE,
        "A short paragraph about an unrelated subject with different claims.",
    )

    assert accepted["quality_pass"] is True
    assert accepted["plain_has_ocn"] is False
    assert still_ocn["quality_pass"] is False
    assert unrelated["quality_pass"] is False


def test_build_counterfactual_pair_frame_is_blinded_and_detector_separated():
    rewrites = pd.DataFrame(
        [
            {
                "source_candidate_id": "abc123",
                "prompt_id": "p1",
                "model_id": "model-a",
                "model_stage": "instruct",
                "decoding": "greedy",
                "seed": 1,
                "response": SOURCE_RESPONSE,
                "plain_response": PLAIN_RESPONSE,
                "has_ocn": True,
                "ocn_count": 1,
                "ocn_patterns": "isnt_just_but",
            }
        ]
    )

    pairs, quality = build_counterfactual_pair_frame(rewrites, seed=7)

    assert quality["quality_pass"].tolist() == [True]
    assert len(pairs) == 2
    assert set(pairs["variant_type"]) == {"candidate_ocn", "plain_rewrite"}
    assert set(pairs["presentation_label"]) == {"A", "B"}
    assert pairs["pair_id"].nunique() == 1
    assert pairs.loc[pairs["variant_type"].eq("candidate_ocn"), "has_ocn"].all()
    assert not pairs.loc[pairs["variant_type"].eq("plain_rewrite"), "has_ocn"].any()


def test_select_best_plain_rewrites_prefers_passing_retry():
    attempts = pd.DataFrame(
        [
            {
                "source_candidate_id": "abc123",
                "response": SOURCE_RESPONSE,
                "plain_response": SOURCE_RESPONSE,
                "rewrite_attempt": 1,
            },
            {
                "source_candidate_id": "abc123",
                "response": SOURCE_RESPONSE,
                "plain_response": PLAIN_RESPONSE,
                "rewrite_attempt": 2,
            },
        ]
    )

    best, quality = select_best_plain_rewrites(attempts)

    assert len(quality) == 2
    assert best.loc[0, "rewrite_attempt"] == 2
    assert best.loc[0, "quality_pass"]
