---
license: mit
task_categories:
- text-classification
- preference-ranking
language:
- en
pretty_name: OCN Reward Preference Pairs
---

# OCN Reward Preference Pairs

This dataset contains matched response variants for testing whether reward models, LLM judges, or humans prefer contrastive-negation framing when core information is controlled.

## Variant Types

- `plain`: direct affirmative response.
- `justified_ocn`: OCN form with a plausible expansion.
- `empty_ocn`: OCN form with redundant or low-information contrast.
- `explicit_genuine_contrast`: contrast made explicit by the prompt framing.

## Intended Use

Score variants on correctness, clarity, depth, naturalness, professionalism, and overall preference.

## Limitations

The initial helper-generated pairs are a starter set. For a paper-grade reward experiment, manually audit proposition counts, length, and fluency.
