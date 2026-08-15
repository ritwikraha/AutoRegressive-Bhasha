---
license: mit
task_categories:
- text-classification
- token-classification
language:
- en
pretty_name: OCN Detection Candidates
---

# OCN Detection Candidates

This dataset contains model generations scored by a lexical OCN detector.

## Columns

- generation metadata;
- `has_ocn`: whether the detector found at least one candidate;
- `ocn_count`: number of candidate constructions;
- `ocn_patterns`: pipe-separated detector pattern names;
- `ocn_spans`: extracted candidate spans;
- `response_tokens_approx`: tokenizer-independent approximate token count;
- `ocn_per_1k_tokens`: normalized density.

## Important Note

The detector identifies **candidate contrastive-negation constructions**, not pragmatic misuse. Human annotation or semantic classification is required to distinguish genuine contrast from overgeneralized contrast.
