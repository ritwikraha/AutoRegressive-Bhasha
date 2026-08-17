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

Apply these rules in order and stop at the first clearly satisfied rule:

1. Use `non_ocn_negation` for ordinary factual negation with no rhetorical X-to-Y upgrade.
2. Use `genuine_contrast` when the user prompt, before the model response begins, directly states or strongly implies X. The response's own mention of X is not prompt support.
3. Use `legitimate_pedagogy` when X is a documented, widely known factual misconception and correcting it is useful even if the prompt did not state it. A merely simplistic view is not automatically a common misconception.
4. Use `empty_intensification` when X and Y are close paraphrases, such as "leading people" versus "guiding people."
5. Use `scope_inflation` when Y broadens X but does not actually contrast with it.
6. Use `false_correction` when X is an implausible straw position or an unmotivated correction.
7. Use `template_stacking` when multiple formulaic templates dominate the passage, for example `not simply`, `deeply human`, `beyond efficiency`, `trust and agency`. Do not use this label for one isolated phrase when a more specific semantic relation applies.
8. Use `presupposed_contrast` when X is plausible but not introduced by the prompt.
9. Use `unclear` only when the evidence remains genuinely insufficient or mixed.

## Calibration Boundaries

- Prompt: "Explain photosynthesis." Span: "not only fuels plant growth but also releases oxygen." Label: `scope_inflation`; Y broadens the effects and the prompt did not supply X.
- Prompt: "Some people say a library is only a warehouse for books. Explain its wider role." Span: "not just a warehouse for books; it is a civic learning space." Label: `genuine_contrast`; the prompt supplied X.
- Prompt: "Explain evolution to a beginner." Span: "not a march toward perfection; it is change in inherited traits." Label: `legitimate_pedagogy`; teleological evolution is a recognized misconception.
- Prompt: "Report the server status." Span: "The server is not running." Label: `non_ocn_negation`; this is factual negation.
- Prompt: "Describe leadership." Span: "not just guiding people; it is helping people find direction." Label: `empty_intensification`; X and Y are near paraphrases.
- Prompt: "Describe a museum." Span: "more than just a building; it is a place of memory and interpretation." Label: `presupposed_contrast`; the narrow view is plausible but unprompted.

Notebook `05` includes a separate eight-item held-out boundary set. Report each panel member's accuracy on it, preserve every prediction, and version checkpoints whenever the prompt or codebook changes.

## Recommended Adjudication

Start with two annotators per example. Send examples to adjudication when:

- taxonomy labels differ;
- prompt support differs by two or more points;
- negation_adds_meaning differs by two or more points;
- either annotator marks `unclear`.

For the model-assisted dataset, use two independently prompted open-weight model annotators and preserve their raw outputs. A third model should review every item, not only flagged disagreements, so every final text span and ordinal rating has an explicit adjudicated source. Report model-model agreement separately from human agreement.

For paper validation, draw a blinded audit sample that overrepresents flagged disagreements. Give the same items to two independent human annotators in different random orders, adjudicate using the rules above, and never describe model-panel labels as human annotations before that audit is complete.

## Derived Outcomes

- `strict_misuse`: `empty_intensification`, `scope_inflation`, or `false_correction`.
- `broad_misuse`: strict misuse plus `presupposed_contrast` and `template_stacking`.
- `unsupported_contrast`: `prompt_support <= 2`, excluding `genuine_contrast`, `legitimate_pedagogy`, and `non_ocn_negation`.

Keep all three outcomes separate. The broad definition includes genuinely ambiguous cases and should be reported as a sensitivity analysis, not substituted for the strict primary measure.
