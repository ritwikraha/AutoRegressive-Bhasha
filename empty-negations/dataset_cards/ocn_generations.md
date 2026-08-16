---
license: mit
task_categories:
- text-generation
language:
- en
pretty_name: OCN OSS Model Generations
---

# OCN OSS Model Generations

This dataset contains open-source model generations for prompts designed to elicit or suppress contrastive-negation framing.

## Columns

- prompt metadata from the OCN prompt bank;
- `model_id`: Hugging Face model id;
- `model_family`: model family;
- `model_stage`: base, instruct, or other;
- `decoding`: decoding configuration name;
- `seed`: generation seed;
- `response`: generated answer;
- `created_at`: notebook run timestamp.
- `experiment_id`: experiment cohort identifier;
- `generation_run_id`: generation session timestamp;
- `gpu_name`: accelerator used for generation;
- `precision`: model loading precision;
- `model_revision`: resolved Hugging Face model commit when available.
- `inference_batching`: prompt batching method used for generation.

## Intended Use

Use this dataset to estimate OCN rates by model family, post-training stage, prompt category, persona, and decoding strategy.

## Limitations

Generation settings, model revisions, runtime hardware, and inference batching can affect outputs. The main Gemma 4/Qwen 3.5 run records these fields per row, uses unquantized BF16 weights, and batches up to 24 prompts per generation call. Greedy outputs are deterministic, so the same generated output is reused for both seed-labelled greedy rows.
