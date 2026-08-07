# CalibrateQwen — Dataset Layer

This folder prepares a compact, auditable multiple-choice dataset for calibration and distillation experiments.

## Colab setup

```python
!git clone <YOUR_REPOSITORY_URL>
%cd calibrate_qwen
!pip install -q -r requirements.txt
```

Tinker runs the GPU-heavy sampling/training remotely, so Colab is mainly the control environment. Add `TINKER_API_KEY` and `HF_TOKEN` in the Colab Secrets panel, then load them without hard-coding values:

```python
import os
from google.colab import userdata

os.environ["TINKER_API_KEY"] = userdata.get("TINKER_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

## Where the data lives

The pipeline uses replaceable intermediate directories and one experiment-ready release:

```text
artifacts/issue_base/       # normalized source records; safe to rebuild
artifacts/issue_curated/    # quality-filtered train/validation/test + audit manifest
artifacts/issue_release/    # final local JSONL and Parquet splits
artifacts/teacher/          # paid Tinker outputs; resumable and worth backing up
```

`artifacts/` is intentionally git-ignored. Use the Hugging Face dataset repo as the durable shared copy, while keeping the configs and scripts in Git.

## Issue #1 end-to-end run

Build ARC-Challenge, OpenBookQA, and MMLU using the issue-specific config:

```bash
python data/build_dataset.py --config configs/issue_data_config.yaml
python data/curate_dataset.py \
  --input-dir artifacts/issue_base \
  --output-dir artifacts/issue_curated \
  --config configs/issue_data_config.yaml
```

The curation command normalizes whitespace and labels, validates every record, removes duplicate choices and cross-split duplicate questions, applies deterministic limits, and records source counts and every drop reason in `artifacts/issue_curated/manifest.json`.

## 1. Build the normalized dataset

Command-line form:

```python
!python data/build_dataset.py --config configs/data_config.yaml
```

Notebook form:

```python
from data.build_dataset import load_config, build_dataset, save_dataset

config = load_config("configs/data_config.yaml")
splits = build_dataset(config)
save_dataset(splits, "artifacts/base")
```

Outputs:

```text
artifacts/base/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── train.parquet
├── validation.parquet
├── test.parquet
└── manifest.json
```

Each record has a stable schema:

```json
{
  "id": "...",
  "source": "arc_challenge",
  "source_split": "train",
  "subject": "science_reasoning",
  "question": "...",
  "choices": ["...", "...", "...", "..."],
  "answer_index": 2,
  "answer_label": "C",
  "prompt": "...",
  "metadata": {}
}
```

## 2. Curate the normalized splits

Run this before spending Tinker sampling tokens. It validates schema, removes malformed or duplicate-choice examples, optionally caps splits, and writes a manifest with retention/drop reasons.

```python
!python data/curate_dataset.py \
  --input-dir artifacts/base \
  --output-dir artifacts/curated \
  --config configs/data_config.yaml
```

For a balanced debug set:

```python
!python data/curate_dataset.py \
  --input-dir artifacts/base \
  --output-dir artifacts/curated_debug \
  --config configs/data_config.yaml \
  --train-limit 4000 \
  --validation-limit 500 \
  --test-limit 500 \
  --balance-sources
```

Outputs:

```text
artifacts/curated/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── train.parquet
├── validation.parquet
├── test.parquet
└── manifest.json
```

## 3. Generate option-order robustness examples

```python
!python data/perturb_options.py \
  --input artifacts/curated/test.jsonl \
  --output artifacts/perturbed/test_option_shuffled.jsonl \
  --permutations 2
```

Notebook form:

```python
from data.common import read_jsonl, write_jsonl
from data.perturb_options import create_option_perturbations

records = list(read_jsonl("artifacts/curated/test.jsonl"))
perturbed = create_option_perturbations(
    records,
    permutations_per_example=2,
    seed=42,
)
write_jsonl(perturbed, "artifacts/perturbed/test_option_shuffled.jsonl")
```

## 4. Generate teacher targets through Tinker

Start with 100 records and 3 samples per question:

```python
!python data/generate_teacher_targets.py \
  --input artifacts/issue_release/train_phase1_2000.jsonl \
  --output artifacts/teacher/issue_phase1_teacher.jsonl \
  --teacher-model Qwen/Qwen3.5-9B \
  --renderer qwen3_5_disable_thinking \
  --samples-per-question 5 \
  --max-tokens 192 \
  --concurrency 4 \
  --limit 100
```

After inspecting the output, remove `--limit`. The script appends one completed record at a time and resumes by ID, making it safer for interrupted Colab sessions.

Finalize and audit a completed teacher run:

```bash
python data/finalize_teacher_targets.py \
  --input artifacts/teacher/issue_phase1_teacher.jsonl \
  --release-manifest artifacts/issue_release/manifest.json
```

This checks ID uniqueness and required teacher metadata, writes a Parquet copy, and records parse rate, modal-answer accuracy, mean agreement, and unanimity in a JSON summary and the release manifest.

## Publish to Hugging Face

Publishing is private by default:

```bash
python data/publish_to_huggingface.py \
  --repo-id ritwikraha/calibrate-qwen-curated
```

Add `--public` only after reviewing source-dataset licensing and the generated teacher samples. Re-running the command updates the same dataset repo.

Load each schema-compatible configuration separately:

```python
from datasets import load_dataset

base = load_dataset("ritwikraha/calibrate-qwen-curated", "calibrated_mcq", token=True)
phase1 = load_dataset("ritwikraha/calibrate-qwen-curated", "phase1_2000", token=True)
robustness = load_dataset("ritwikraha/calibrate-qwen-curated", "option_shuffled", token=True)
teacher = load_dataset("ritwikraha/calibrate-qwen-curated", "teacher_phase1", token=True)
```

Teacher metadata includes:

- every raw sample;
- parsing method and validity;
- answer vote counts;
- modal answer;
- agreement-based confidence;
- whether the modal teacher answer matches the gold label.

## Recommended first run

Use only ARC-Challenge, OpenBookQA and CommonsenseQA. Generate targets for 2,000–4,000 training records with three teacher samples each. This is sufficient to validate the entire dataset and Tinker sampling pipeline before scaling.

## Important design choices

- Gold labels are retained separately from teacher outputs.
- Confidence is teacher sample agreement, not an invented verbal percentage.
- Option shuffling is kept as a held-out robustness set by default.
- All records have deterministic IDs, allowing safe resume and joins.
- The data format is Tinker-independent; renderer-specific tokenization belongs in the training scripts.
