# Semantic Annotation Run Results

## Run

- Date: 2026-08-17
- Branch commit: `d7fd7f3`
- Hardware: NVIDIA A100-SXM4-40GB
- W&B: https://wandb.ai/ritwik/ocn-empty-negations/runs/uarxu57d
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

Mistral initially produced 12 unresolved strict-parser failures. A documented parser update recovered all records by repairing malformed JSON, allowing empty X/Y propositions where the schema permits them, and mapping a model-produced `0` (absent) to scale point `1` (definitely absent). Raw outputs and repair markers remain in the annotator configuration.

## Agreement

| Field | Exact agreement | Within one | Cohen's kappa |
| --- | ---: | ---: | ---: |
| taxonomy_label | 0.042 | - | 0.026 |
| prompt_support | 0.073 | 0.458 | 0.037 |
| common_misconception | 0.447 | 0.584 | 0.184 |
| x_y_distinctness | 0.267 | 0.466 | 0.022 |
| negation_adds_meaning | 0.343 | 0.567 | 0.160 |
| straw_position | 0.997 | 0.997 | 0.000 |
| formulaic_ai_style | 0.966 | 0.966 | 0.279 |
| rewrite_loss | 0.966 | 0.966 | 0.000 |

The apparent agreement on the last three ordinal fields is prevalence-dominated: near-constant ratings produce low or zero kappa despite high exact agreement. Overall, 353 of 356 spans triggered adjudication.

## Adjudicated Labels

| Label | Count |
| --- | ---: |
| genuine_contrast | 252 |
| legitimate_pedagogy | 63 |
| template_stacking | 17 |
| scope_inflation | 12 |
| empty_intensification | 6 |
| non_ocn_negation | 6 |

Population-weighted estimates from the adjudicator were:

- Strict misuse: 0.044
- Broad misuse: 0.078
- Unsupported contrast: 0.045

## Interpretation

This run creates a complete, reproducible model-panel dataset, not a human gold standard. The near-total taxonomy disagreement shows that open-weight judges are poorly calibrated for this pragmatic distinction under the current codebook. The adjudicated rates should therefore be treated as provisional sensitivity estimates rather than paper-ready prevalence claims.

Notebook `05` saved two independently ordered 100-item human-audit packets to Google Drive. Two human annotators must complete those packets and adjudicate disagreements before the semantic rates can support the minimal viable paper.
