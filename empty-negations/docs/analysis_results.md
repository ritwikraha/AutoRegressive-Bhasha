# Main Analysis Results

## Run

- Date: 2026-08-17
- Branch commit: `3ff9f01`
- Analysis version: `v2_robust`
- W&B: https://wandb.ai/ritwik/ocn-empty-negations/runs/boihq6og
- Detection dataset: https://huggingface.co/datasets/ritwikraha/ocn-empty-negations-detection-main-gemma4-qwen35
- Semantic dataset: https://huggingface.co/datasets/ritwikraha/ocn-empty-negations-semantic-main-gemma4-qwen35

## Lexical Prevalence

The analysis removes 2,376 deterministic greedy rows duplicated across bookkeeping seeds. This leaves 7,128 response rows, 593 responses with at least one lexical OCN candidate, 653 detected constructions, and approximately 890,553 response tokens.

- Response-level lexical OCN rate: 0.0832
- OCN constructions per 1,000 approximate tokens: 0.733

Prompt-clustered model estimates were:

| Model | Decoding | Rate | 95% CI |
| --- | --- | ---: | ---: |
| Qwen 3.5 2B | greedy | 0.157 | 0.130-0.185 |
| Qwen 3.5 2B | normal temperature | 0.152 | 0.128-0.176 |
| Qwen 3.5 2B Base | normal temperature | 0.104 | 0.085-0.123 |
| Gemma 4 E2B-it | normal temperature | 0.073 | 0.056-0.092 |
| Gemma 4 E2B-it | greedy | 0.069 | 0.049-0.089 |
| Qwen 3.5 2B Base | greedy | 0.056 | 0.037-0.074 |
| Gemma 4 E2B | normal temperature | 0.024 | 0.015-0.033 |
| Gemma 4 E2B | greedy | 0.013 | 0.005-0.024 |

## Clustered Regression

A binomial GLM clusters covariance by `prompt_id` and controls for model family, model stage, decoding, prompt variant, persona, category, and requested length. Key conditional effects were:

| Effect | Odds ratio | p-value |
| --- | ---: | ---: |
| Instruct stage at the Gemma reference level | 4.44 | <0.001 |
| Qwen family at the base reference level | 5.75 | <0.001 |
| Instruct by Qwen interaction | 0.50 | 0.008 |
| Normal-temperature decoding | 1.28 | 0.008 |
| Explicit-misconception prompt variant | 21.40 | <0.001 |
| Nuanced prompt variant | 4.67 | <0.001 |
| Plain-factual prompt variant | 0.42 | 0.002 |
| Prohibit-OCN prompt variant | 1.63 | 0.031 |

The interaction means the instruct-stage odds ratio is smaller for Qwen than for Gemma; main effects should not be interpreted as unconditional averages.

## Semantic Sensitivity

The adjudicator's weighted strict-misuse estimate is 0.179 (response-clustered 95% CI 0.129-0.232). The same outcome computed from each panel member ranges from 0.037 for OLMo 2 to 0.591 for Mistral. This judge spread dominates sampling uncertainty.

The lexical findings and clustered regression are suitable for the minimal paper's primary quantitative result. Semantic prevalence must remain explicitly provisional until two humans complete and adjudicate the blinded 100-item audit.

## Drive Outputs

Tables and the generated report are under:

```text
/content/drive/MyDrive/ocn_empty_negations/artifacts/data/analysis/main_gemma4_qwen35/v2_robust/
```

The dashboard is under:

```text
/content/drive/MyDrive/ocn_empty_negations/artifacts/figures/analysis/main_gemma4_qwen35/v2_robust/06_analysis_dashboard.png
```
