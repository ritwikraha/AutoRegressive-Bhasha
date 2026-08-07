---
configs:
- config_name: calibrated_mcq
  default: true
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test_in_domain.parquet
  - split: test_ood
    path: data/test_ood_commonsenseqa.parquet
- config_name: phase1_2000
  data_files:
  - split: train
    path: data/train_phase1_2000.parquet
- config_name: option_shuffled
  data_files:
  - split: test
    path: data/test_option_shuffled.parquet
- config_name: teacher_phase1
  data_files:
  - split: train
    path: teacher/issue_phase1_teacher.parquet
task_categories:
- question-answering
- text-classification
language:
- en
tags:
- calibration
- multiple-choice
- knowledge-distillation
---

# CalibrateQwen Curated MCQ Dataset

This dataset supports confidence calibration and teacher-distillation experiments on multiple-choice questions.

## Data layout

- `data/train.*`: 9,134 in-domain training examples from ARC-Challenge, OpenBookQA, and MMLU.
- `data/train_phase1_2000.*`: deterministic 2,000-example subset for the first experiment phase.
- `data/validation.*`: 1,000 in-domain validation examples.
- `data/test_in_domain.*`: 1,163 in-domain test examples.
- `data/test_ood_commonsenseqa.*`: 3,000 held-out CommonsenseQA examples.
- `data/test_option_shuffled.*`: two deterministic option-order perturbations per in-domain test example.
- `teacher/*.jsonl`: repeated Tinker samples, modal answers, agreement confidence, and parsing diagnostics.

The files are exposed as separate Hugging Face configurations because teacher
and option-shuffled records intentionally have additional columns:

- `calibrated_mcq`: standard train, validation, test, and OOD test splits.
- `phase1_2000`: deterministic 2,000-example training subset.
- `option_shuffled`: robustness test with perturbation metadata.
- `teacher_phase1`: Phase 1 records with repeated teacher samples.

Every base record contains a stable ID, source metadata, normalized question and choices, gold answer index and label, and a rendered prompt. See `data/manifest.json` for counts and provenance.

## Curation

The curation pass removes invalid schemas, duplicate choice text, out-of-range labels, overlong prompts, and duplicate questions across splits. Filtering thresholds and drop counts are recorded in the local curation manifest so the release can be reproduced.

Teacher confidence is measured from repeated answer agreement rather than a model-written confidence percentage. Gold labels and teacher outputs remain separate.

## Source datasets

This is a derived dataset. Users are responsible for following the licenses and terms of ARC-Challenge, OpenBookQA, MMLU, and CommonsenseQA.
