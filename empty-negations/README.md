# Empty Negations / OCN Research Notebooks

This subproject turns the research proposal on **Overgeneralized Contrastive Negation (OCN)** into a Colab-first experimental pipeline inside `AutoRegressive-Bhasha`.

OCN covers constructions such as:

> "This is not just X; it is Y."

The research question is whether language models overuse these frames, especially when the negated proposition was not introduced by the prompt or is semantically redundant.

## What Is Included

- `docs/research_protocol.md` - paper-oriented study design and hypotheses.
- `docs/experiment_matrix.md` - concrete experiment menu and rollout plan.
- `docs/semantic_annotation_results.md` - completed model-panel run, agreement diagnostics, and human-validation requirements.
- `docs/analysis_results.md` - deduplicated lexical results, clustered regression, semantic sensitivity, and paper-facing limitations.
- `experiments/configs/main_gemma4_qwen35.yaml` - exact main-run model, decoding, seed, hardware, and publication manifest.
- `annotation/codebook.md` - human annotation rules for OCN examples.
- `dataset_cards/` - Hugging Face dataset cards for each published dataset.
- `requirements-colab.txt` - Colab Pro runtime dependencies.
- `src/ocn/` - local detector, prompt factory, Colab helpers, OSS generation utilities, metrics, and reward-pair helpers.
- `scripts/run_detection.py` - CLI for scoring cached generations.
- `scripts/build_colab_notebooks.py` - reproducible notebook generator.
- `notebooks/` - numbered research notebooks.

## Colab Research Sequence

Run these in order:

1. `00_colab_setup_and_config.ipynb`
   Mounts Google Drive, reads Colab secrets, logs into Hugging Face and W&B, and writes shared config.
2. `01_create_prompt_dataset.ipynb`
   Creates the factorial OCN prompt dataset, saves it to Drive, logs plots to W&B, and publishes to Hugging Face.
3. `02_generate_oss_model_responses.ipynb`
   Runs the full 594-prompt main experiment on matched Gemma 4 E2B and Qwen 3.5 2B base/post-trained pairs. Requires an A100, autosaves resumable chunks to Drive, and republishes the combined dataset after each chunk.
4. `03_detect_and_publish_ocn_dataset.ipynb`
   Runs the lexical OCN detector, saves scored generations, publishes the candidate dataset to Hugging Face, and logs charts to W&B.
5. `04_create_reward_pair_dataset.ipynb`
   Uses Qwen 3.5 2B with a Gemma 4 E2B-it fallback on an L4 to create resumable direct-affirmative rewrites of real OCN candidates from notebook `03`, filters for detector separation and content retention, saves audit artifacts to Drive, and publishes blinded pairs to Hugging Face.
6. `05_annotation_sampling_and_adjudication.ipynb`
   Collapses duplicate candidates, samples unique responses within each model-by-decoding stratum, annotates every selected span with two independent open-weight models, adjudicates every item with a third model on an A100, autosaves checkpoints to Drive, publishes the semantic dataset to Hugging Face, and creates two blinded human-audit packets.
7. `06_analysis_and_reporting.ipynb`
   Removes deterministic greedy-seed duplicates, computes prompt-clustered lexical intervals, response-clustered weighted semantic intervals, panel sensitivity, and a prompt-clustered binomial regression. Saves tables, a report, and plots to Drive and logs them to W&B.
8. `07_optional_reward_model_scoring.ipynb`
   Scores reward pairs with an open reward model and publishes the score dataset.
9. `08_optional_lora_style_intervention.ipynb`
   Runs a small Qwen LoRA SFT intervention for plain-vs-OCN style acquisition.

Required Colab secrets:

```text
HF_WRITE_ACCESS
WANDB_KEY
```

The setup notebook derives Hugging Face dataset repo names from your authenticated HF username:

```text
<user>/ocn-empty-negations-prompts
<user>/ocn-empty-negations-generations
<user>/ocn-empty-negations-detection
<user>/ocn-empty-negations-generations-main-gemma4-qwen35
<user>/ocn-empty-negations-detection-main-gemma4-qwen35
<user>/ocn-empty-negations-reward-pairs-main-gemma4-qwen35
<user>/ocn-empty-negations-reward-scores-main-gemma4-qwen35
<user>/ocn-empty-negations-semantic-main-gemma4-qwen35
<user>/ocn-empty-negations-reward-pairs
<user>/ocn-empty-negations-reward-scores
```

All notebooks autosave artifacts under:

```text
/content/drive/MyDrive/ocn_empty_negations/artifacts/
```

Notebook `02` keeps the completed Qwen 2.5 pilot dataset separate from the main Gemma 4/Qwen 3.5 dataset. The Gemma 4 checkpoints are public Apache 2.0 models and require no separate access request. Before the main run, start a fresh Colab A100 runtime and rerun notebook `00` so the pinned Transformers version is installed.

Notebook `05` uses Mistral 7B Instruct and OLMo 2 13B Instruct as independent annotators, then Qwen 3 14B as the adjudicator. All three run sequentially in BF16 on an A100, with no 4-bit dependency. Its published labels are explicitly model-assisted. Complete the two human-audit CSV packets saved to Drive before describing semantic labels as human-validated in a paper.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For local model experimentation:

```bash
pip install -r requirements-colab.txt
```

Run detector tests:

```bash
pytest
```

Score a cached generations file:

```bash
python scripts/run_detection.py \
  --input data/raw/example_generations.csv \
  --text-column response \
  --output data/processed/example_detection.csv
```

Expected generation CSV columns:

```text
prompt_id,model,stage,temperature,seed,prompt,response
```

Only `response` is required for detection; the other columns are used for grouped analysis.

## Research Scope

The first publishable core is:

1. Base-vs-instruct measurement.
2. Prompt-trigger factorial study.
3. Reward preference test on matched plain/OCN pairs.
4. Small controlled SFT/DPO intervention.
5. Early activation-probe result on a small open model.

The notebooks emphasize the first three as the main runnable path and include optional reward scoring and LoRA intervention notebooks for the causal follow-up.
