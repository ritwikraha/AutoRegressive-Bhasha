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
configs:
  - config_name: sample
    data_files:
      - split: train
        path: sample/train-*
  - config_name: annotator_a
    data_files:
      - split: train
        path: annotator_a/train-*
  - config_name: annotator_b
    data_files:
      - split: train
        path: annotator_b/train-*
  - config_name: adjudicated
    data_files:
      - split: train
        path: adjudicated/train-*
  - config_name: agreement
    data_files:
      - split: train
        path: agreement/train-*
  - config_name: human_audit
    data_files:
      - split: train
        path: human_audit/train-*
  - config_name: calibration
    data_files:
      - split: train
        path: calibration/train-*
---

# OCN Semantic Annotations

This dataset contains a stratified span-level semantic audit of lexical contrastive-negation candidates from the main Gemma 4 E2B and Qwen 3.5 2B experiment.

## Sampling

Exact duplicate generations caused by greedy seed reuse are collapsed within prompt, model, and decoding. Responses are sampled within each `model_id` by `decoding` stratum, and every detected span in a selected response is retained. `sampling_probability` and `sample_weight` support population-weighted estimates.

## Annotation

Two independently prompted open-weight models annotate every span using the project codebook. A third open-weight model reviews every item and supplies the final annotation. The dataset preserves both initial annotations, automatic disagreement flags, and final adjudication fields. Prompt version `v2_calibrated` adds an explicit decision hierarchy and an eight-item held-out boundary set; the `calibration` configuration preserves every panel prediction on that set.

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

## Calibrated Main Run

Prompt version `v2_calibrated` sampled 310 unique responses containing 356 spans from a population of 593 unique candidate responses and 653 spans. The two model annotators agreed on the taxonomy for 8.1% of spans (Cohen's kappa 0.002), and 353 of 356 spans met at least one adjudication trigger. Qwen 3 14B reviewed all 356 spans.

Held-out taxonomy accuracy was 5/8 for Mistral, 2/8 for OLMo 2, and 3/8 for Qwen. The adjudicator's population-weighted estimates were 17.9% strict misuse (response-clustered 95% CI 12.9%-23.2%), 18.4% broad misuse (13.5%-23.9%), and 18.2% unsupported contrast (13.2%-23.6%). The annotators' strict rates ranged from 3.7% to 59.1%, so these are sensitivity estimates rather than stable prevalence estimates.

The earlier uncalibrated diagnostic run estimated 4.4% strict misuse. Prompt versions must be analyzed separately; model-panel versions should not be blended.

## Important Limitation

These are model-assisted semantic annotations, not human gold labels. The `human_audit` split is deliberately left unlabeled for an independent human audit. Paper claims about semantic misuse should report that audit and must not describe the model panel as human annotation.

## Configurations

- `sample`: sampled span population with sampling weights (`train` split).
- `annotator_a`: first independent model annotation (`train` split).
- `annotator_b`: second independent model annotation (`train` split).
- `adjudicated`: final panel annotation plus both initial records (`train` split).
- `agreement`: field-level exact agreement and Cohen's kappa diagnostics (`train` split).
- `human_audit`: blinded paper-validation packet (`train` split).
- `calibration`: case-level held-out calibration predictions for every model-panel member (`train` split).
