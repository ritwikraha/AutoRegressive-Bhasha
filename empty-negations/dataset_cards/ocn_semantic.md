---
pretty_name: OCN Semantic Annotations - Gemma 4 and Qwen 3.5
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - pragmatics
  - rhetoric
  - llm-evaluation
  - semantic-annotation
---

# OCN Semantic Annotations

This dataset contains a stratified span-level semantic audit of lexical contrastive-negation candidates from the main Gemma 4 E2B and Qwen 3.5 2B experiment.

## Sampling

Exact duplicate generations caused by greedy seed reuse are collapsed within prompt, model, and decoding. Responses are sampled within each `model_id` by `decoding` stratum, and every detected span in a selected response is retained. `sampling_probability` and `sample_weight` support population-weighted estimates.

## Annotation

Two independently prompted open-weight models annotate every span using the project codebook. A third open-weight model reviews every item and supplies the final annotation. The dataset preserves both initial annotations, automatic disagreement flags, and final adjudication fields.

The taxonomy is:

- `genuine_contrast`
- `legitimate_pedagogy`
- `presupposed_contrast`
- `empty_intensification`
- `scope_inflation`
- `false_correction`
- `template_stacking`
- `non_ocn_negation`
- `unclear`

## Important Limitation

These are model-assisted semantic annotations, not human gold labels. The `human_audit` split is deliberately left unlabeled for an independent human audit. Paper claims about semantic misuse should report that audit and must not describe the model panel as human annotation.

## Configurations

- `sample`: sampled span population with sampling weights (`train` split).
- `annotator_a`: first independent model annotation (`train` split).
- `annotator_b`: second independent model annotation (`train` split).
- `adjudicated`: final panel annotation plus both initial records (`train` split).
- `agreement`: field-level exact agreement and Cohen's kappa diagnostics (`train` split).
- `human_audit`: blinded paper-validation packet (`train` split).
