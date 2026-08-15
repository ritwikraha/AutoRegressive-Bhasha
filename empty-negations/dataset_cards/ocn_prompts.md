---
license: mit
task_categories:
- text-generation
- text-classification
language:
- en
pretty_name: OCN Prompt Bank
---

# OCN Prompt Bank

This dataset contains controlled prompts for studying **Overgeneralized Contrastive Negation (OCN)** in model-generated text.

OCN refers to constructions such as `not just X, but Y`, `not merely X`, `more than just X`, or `goes beyond X`, especially when the rejected proposition was not supplied by the user or is semantically redundant.

## Columns

- `prompt_id`: stable prompt identifier.
- `topic`: semantic topic.
- `category`: prompt category.
- `variant`: prompt variant.
- `persona`: requested speaker/register.
- `length_target`: approximate word target.
- `contrast_availability`: whether a misconception is present.
- `prompt`: final model-facing prompt.

## Intended Use

Use this dataset to generate model responses across base and instruction-tuned open models, then measure OCN frequency and subtype.

## Limitations

This prompt bank is designed for English pilot experiments. Cross-lingual claims require additional prompt construction and validation.
