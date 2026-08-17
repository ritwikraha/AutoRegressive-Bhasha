# Semantic Annotation Results

## Calibrated Run

- Date: 2026-08-17
- Branch commit: `3ff9f01`
- Hardware: NVIDIA A100-SXM4-40GB
- Prompt version: `v2_calibrated`
- W&B: https://wandb.ai/ritwik/ocn-empty-negations/runs/cjowez2h
- Hugging Face: https://huggingface.co/datasets/ritwikraha/ocn-empty-negations-semantic-main-gemma4-qwen35

## Sampling

- Detection rows: 9,504
- Lexical candidate rows: 768
- Unique candidate responses after exact-generation deduplication: 593
- Candidate spans: 653
- Sampled responses: 310
- Sampled spans: 356
- Strata: `model_id` by `decoding`
- Target allocation: up to 50 unique responses per stratum

All detected spans in a selected response were retained. The published `sample` configuration includes response-level inclusion probabilities and inverse-probability weights.

## Model Panel

- Annotator A: `mistralai/Mistral-7B-Instruct-v0.3`
- Annotator B: `allenai/OLMo-2-1124-13B-Instruct`
- Adjudicator: `Qwen/Qwen3-14B`
- Precision: BF16
- Quantization: none

The calibrated prompt applies a fixed decision hierarchy, distinguishes support in the user prompt from support introduced by the response, and includes boundary anchors. Eight held-out cases cover eight taxonomy labels. All raw outputs and parser repair markers remain available.

Held-out taxonomy accuracy was:

| Panel member | Correct | Accuracy |
| --- | ---: | ---: |
| Mistral 7B Instruct | 5/8 | 0.625 |
| OLMo 2 13B Instruct | 2/8 | 0.250 |
| Qwen 3 14B adjudicator | 3/8 | 0.375 |

## Agreement

| Field | Exact agreement | Within one | Cohen's kappa |
| --- | ---: | ---: | ---: |
| taxonomy_label | 0.081 | - | 0.002 |
| prompt_support | 0.868 | 0.871 | 0.046 |
| common_misconception | 0.736 | 0.812 | 0.065 |
| x_y_distinctness | 0.329 | 0.348 | 0.010 |
| negation_adds_meaning | 0.191 | 0.660 | 0.164 |
| straw_position | 0.992 | 0.992 | 0.000 |
| formulaic_ai_style | 0.567 | 0.902 | 0.033 |
| rewrite_loss | 0.947 | 0.949 | -0.003 |

Prompt-support consistency improved substantially, but taxonomy agreement did not become reliable. Several ordinal fields remain prevalence-dominated: near-constant ratings produce low or zero kappa despite high exact agreement. Overall, 353 of 356 spans triggered adjudication.

## Adjudicated Labels

| Label | Count |
| --- | ---: |
| genuine_contrast | 254 |
| scope_inflation | 57 |
| legitimate_pedagogy | 33 |
| non_ocn_negation | 7 |
| empty_intensification | 3 |
| template_stacking | 1 |
| presupposed_contrast | 1 |

Population-weighted estimates from the adjudicator, with response-clustered bootstrap intervals, were:

- Strict misuse: 0.179 (95% CI 0.129-0.232)
- Broad misuse: 0.184 (95% CI 0.135-0.239)
- Unsupported contrast: 0.182 (95% CI 0.132-0.236)

Strict-misuse sensitivity by panel member was 0.591 for Mistral, 0.037 for OLMo 2, and 0.179 for Qwen. This spread is larger than the sampling uncertainty and is the dominant source of uncertainty.

## Interpretation

This run creates a complete, reproducible model-panel dataset, not a human gold standard. The decision hierarchy improved agreement on prompt support but did not resolve taxonomy instability. The adjudicated rates must therefore be treated as provisional sensitivity estimates rather than paper-ready prevalence claims.

Notebook `05` saved two independently ordered 100-item human-audit packets to Google Drive with the `v2_calibrated` suffix. Two human annotators must complete those packets and adjudicate disagreements before the semantic rates can support the minimal viable paper.

## Earlier Diagnostic Run

The uncalibrated run at W&B run `uarxu57d` produced 4.2% taxonomy agreement and an adjudicator strict-misuse estimate of 0.044. It remains useful as a prompt-sensitivity diagnostic but must not be pooled with `v2_calibrated`.
