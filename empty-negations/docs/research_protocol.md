# Research Protocol: Overgeneralized Contrastive Negation

## Working Definition

**Overgeneralized Contrastive Negation (OCN)** is a contrastive negation frame whose rejected proposition is unprompted, weakly supported, redundant, or pragmatically unnecessary.

Canonical surface forms include:

- `not just X, but Y`
- `not merely X; it is Y`
- `not only X, but also Y`
- `more than just X`
- `goes beyond X`
- `rather than simply X`
- `not so much X as Y`
- `cannot be reduced to X`

The project distinguishes **occurrence** from **misuse**. Some contrastive negation is legitimate, especially when the prompt explicitly contains a misconception or when the topic has a well-established false belief.

## Central Question

Why do instruction-tuned language models disproportionately use contrastive negation constructions, particularly when the rejected proposition is not supplied by the user?

## Hypotheses

| ID | Hypothesis | Main prediction | First test |
| --- | --- | --- | --- |
| H1 | Pretraining frequency | Base models use OCN in article-like continuations. | Base vs instruct, completion mode. |
| H2 | SFT amplification | Instruct/SFT models sharply exceed base models. | Same-family checkpoints. |
| H3 | Reward preference | OCN variants receive higher reward/judge scores than matched plain variants. | Plain vs OCN pair scoring. |
| H4 | Elaboration scaffold | OCN increases under requests for nuance, depth, and significance. | Prompt-factor sweep. |
| H5 | Discourse glue | OCN connects multiple weakly related facts. | Multi-dimension prompts. |
| H6 | Anti-triviality | Models reject an imagined shallow reading to sound deeper. | Meaning/significance prompts. |
| H7 | Synthetic recursion | Synthetic instruction corpora inherit/amplify OCN-heavy register. | Dataset audit and controlled SFT. |
| H8 | Assistant persona | Assistant framing has higher OCN than other genres/personas. | Persona-conditioned prompts. |
| H9 | Decoding | OCN rate varies systematically with temperature/top-p. | Decoding sweep. |
| H10 | Internal style feature | OCN is predictable before the word `not` appears. | Activation probe. |

## Taxonomy

| Label | Description | Count as misuse? |
| --- | --- | --- |
| genuine_contrast | Prompt or discourse already contains the rejected proposition. | No |
| legitimate_pedagogy | Corrects a common misconception relevant to the topic. | Usually no |
| presupposed_contrast | Invents a plausible but unexpressed simplistic view. | Maybe |
| empty_intensification | X and Y are near paraphrases. | Yes |
| scope_inflation | Y expands or restates implications of X without real contrast. | Often |
| false_correction | Constructs a straw position nobody proposed. | Yes |
| template_stacking | OCN appears inside a larger cluster of formulaic assistant rhetoric. | Contextual |
| non_ocn_negation | Ordinary factual negation. | No |

## Primary Metrics

- **OCN rate:** fraction of responses containing at least one detected OCN.
- **OCN density:** detected OCN constructions per 1,000 response tokens.
- **Unsupported contrast rate:** annotated fraction where X was not supplied or implied by the prompt.
- **Semantic redundancy score:** similarity/entailment between X and Y.
- **Deletion invariance:** whether removing the negation scaffold preserves meaning.
- **False-depth score:** perceived depth minus actual information gain.
- **Template concentration:** share of OCN examples captured by top lexical templates.

## Minimum Viable Paper

1. **Measurement:** base/instruct comparison across several model families.
2. **Prompt triggers:** controlled factorial prompt sweep.
3. **Reward preference:** matched plain/OCN pairs scored by reward models, LLM judges, and humans.
4. **Controlled training:** small SFT or DPO intervention showing style acquisition or suppression.
5. **Mechanistic pilot:** probe whether future OCN is linearly decodable before explicit negation.

## Threats To Validity

- Length confound: OCN often lengthens responses.
- Information confound: the second clause may add real content.
- Topic confound: abstract topics invite contrast more than procedural tasks.
- Detector overreach: lexical matches are not necessarily misuse.
- Base-model comparability: base models may fail direct instruction.
- Judge circularity: LLM judges may share the bias under study.
- English-only scope: cross-lingual conclusions require separate validation.
