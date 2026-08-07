# %% [markdown]
# Generate option-order perturbations for robustness evaluation.
#
# The gold answer is remapped automatically. By default, perturbed examples are
# written as a separate evaluation set, not mixed into training.

# %%
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterable

try:
    from data.common import format_question_prompt, index_to_label, read_jsonl, stable_id, write_jsonl
except ModuleNotFoundError:
    from common import format_question_prompt, index_to_label, read_jsonl, stable_id, write_jsonl  # type: ignore


# %%
def permute_example(
    record: dict[str, Any],
    *,
    permutation_index: int,
    seed: int,
) -> dict[str, Any]:
    choices = list(record["choices"])
    if len(choices) < 2:
        raise ValueError("Cannot permute an example with fewer than two choices.")

    rng = random.Random(stable_id(seed, record["id"], permutation_index, length=16))
    order = list(range(len(choices)))
    rng.shuffle(order)

    # Avoid producing a nominal perturbation identical to the original ordering.
    if order == list(range(len(choices))):
        order = order[1:] + order[:1]

    new_choices = [choices[old_index] for old_index in order]
    old_answer_index = int(record["answer_index"])
    new_answer_index = order.index(old_answer_index)

    output = dict(record)
    output["id"] = stable_id(record["id"], "option_permutation", permutation_index)
    output["parent_id"] = record["id"]
    output["perturbation"] = {
        "type": "option_permutation",
        "index": permutation_index,
        "new_to_old_order": order,
    }
    output["choices"] = new_choices
    output["answer_index"] = new_answer_index
    output["answer_label"] = index_to_label(new_answer_index)
    output["prompt"] = format_question_prompt(record["question"], new_choices)
    return output


def create_option_perturbations(
    records: Iterable[dict[str, Any]],
    *,
    permutations_per_example: int,
    seed: int,
    include_original: bool = False,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if include_original:
            output.append(record)
        for permutation_index in range(permutations_per_example):
            output.append(
                permute_example(
                    record,
                    permutation_index=permutation_index,
                    seed=seed,
                )
            )
    return output


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Create option-shuffled robustness examples.")
    parser.add_argument("--input", default="artifacts/base/test.jsonl")
    parser.add_argument("--output", default="artifacts/perturbed/test_option_shuffled.jsonl")
    parser.add_argument("--permutations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    perturbed = create_option_perturbations(
        records,
        permutations_per_example=args.permutations,
        seed=args.seed,
        include_original=args.include_original,
    )
    count = write_jsonl(perturbed, args.output)
    print(f"Wrote {count} records to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
