# OCN Annotation Codebook

## Unit Of Annotation

Annotate each detected contrastive-negation construction inside a model response. If one response has multiple constructions, annotate them separately.

## Required Fields

| Field | Type | Description |
| --- | --- | --- |
| example_id | string | Stable row id. |
| prompt | text | User prompt shown to the model. |
| response | text | Full model response. |
| span_text | text | The detected OCN sentence or clause. |
| rejected_x | text | The proposition being downplayed or rejected. |
| asserted_y | text | The proposition being emphasized. |
| taxonomy_label | category | One of the labels below. |
| prompt_support | 1-5 | Was X supplied or strongly implied by the prompt? |
| common_misconception | 1-5 | Is X a known misconception worth correcting? |
| x_y_distinctness | 1-5 | Are X and Y semantically distinct? |
| negation_adds_meaning | 1-5 | Does the negative frame add meaning beyond `X and Y`? |
| straw_position | 1-5 | Does the sentence invent an implausible rejected position? |
| formulaic_ai_style | 1-5 | Does the construction sound template-like or assistant-like? |
| rewrite_loss | 1-5 | Would an affirmative rewrite lose important meaning? |
| notes | text | Short rationale or uncertainty. |

Scale convention:

```text
1 = definitely no / absent
2 = probably no
3 = unclear or mixed
4 = probably yes
5 = definitely yes / strong
```

## Taxonomy Labels

- `genuine_contrast`
- `legitimate_pedagogy`
- `presupposed_contrast`
- `empty_intensification`
- `scope_inflation`
- `false_correction`
- `template_stacking`
- `non_ocn_negation`
- `unclear`

## Decision Rules

Use `genuine_contrast` when the user prompt directly states, asks about, or strongly implies X.

Use `legitimate_pedagogy` when X is a widely known misconception and correcting it is useful even if the prompt did not state it.

Use `presupposed_contrast` when X is plausible but not introduced by the prompt.

Use `empty_intensification` when X and Y are close paraphrases, such as "leading people" vs "guiding people."

Use `scope_inflation` when Y broadens X but does not actually contrast with it.

Use `false_correction` when X is an implausible straw position or an unmotivated correction.

Use `template_stacking` when the issue is not only one negation but a larger formulaic register, for example `not simply`, `deeply human`, `beyond efficiency`, `trust and agency`.

Use `non_ocn_negation` for ordinary factual negation, such as "Viruses are not cells."

## Recommended Adjudication

Start with two annotators per example. Send examples to adjudication when:

- taxonomy labels differ;
- prompt support differs by two or more points;
- negation_adds_meaning differs by two or more points;
- either annotator marks `unclear`.
