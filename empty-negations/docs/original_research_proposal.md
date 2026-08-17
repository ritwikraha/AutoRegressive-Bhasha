## Research proposal: Why do LLMs overuse “not just X, but Y” constructions?

The behavior you describe can be called **contrastive negation framing**:

> “X is not merely A; it is B.”
> “This isn’t just about X—it is also about Y.”
> “The significance lies not only in A, but in B.”

I would avoid calling these “blank negations” in the paper because that term is not established. A useful operational name is:

> **Overgeneralized Contrastive Negation, or OCN**

The key observation is that the negative clause often rejects a position nobody proposed. It creates a rhetorical contrast without supplying genuinely contrasting information.

This is related to a broader finding that LLMs favor particular grammatical and stylistic structures more consistently than humans and exhibit less natural stylistic variation. ([[arXiv](https://arxiv.org/html/2410.16107v1?utm_source=chatgpt.com)][1]) However, I am not aware of research that isolates this exact construction and causally determines whether it originates in pretraining, instruction tuning, preference optimization, decoding, synthetic-data contamination, or internal discourse planning.

That makes it a credible and potentially high-signal research project.

---

# 1. Central research question

> **Why do instruction-tuned language models disproportionately use contrastive negation constructions, particularly when the rejected proposition is unprompted, weak, redundant, or semantically empty?**

The project should distinguish five possible levels of explanation:

1. **Corpus explanation:** the construction is unusually common in influential training domains.
2. **Post-training explanation:** supervised instruction tuning or preference optimization amplifies it.
3. **Reward explanation:** evaluators interpret the construction as nuanced, complete, persuasive, or sophisticated.
4. **Generation explanation:** the construction is a low-risk continuation strategy that helps the model elaborate.
5. **Representation explanation:** models internally formulate answers as corrections or contrasts, even when the user did not provide a proposition to correct.

The goal should not merely be to demonstrate that the phrase occurs frequently. The goal is to establish a **causal decomposition** of where the tendency comes from.

---

# 2. Precise taxonomy of the phenomenon

Before running experiments, divide contrastive negation into subtypes.

## 2.1 Genuine contrastive negation

The rejected proposition is present in the prompt or discourse.

**Prompt**

> Is the French Revolution primarily an economic event?

**Response**

> It was not only an economic event; it also involved political legitimacy and social hierarchy.

This may be pragmatically justified.

## 2.2 Presupposed contrast

The response invents a simplistic interpretation that the user never expressed.

**Prompt**

> Explain the importance of the French Revolution.

**Response**

> The French Revolution was not just about overthrowing a monarchy; it was about redefining citizenship.

The first interpretation is plausible, but the user did not advance it.

## 2.3 Empty intensification

The two sides are not meaningfully contrasted.

> Leadership is not merely about leading people; it is about guiding them toward a goal.

“Leading people” and “guiding them” are nearly paraphrases.

## 2.4 Scope inflation

The second clause expands the scope but does not overturn the first.

> The invention of printing was not just a technological change; it transformed communication.

The second clause is largely an implication of the first.

## 2.5 False correction

The first clause constructs a straw position.

> Climate policy is not about stopping economic growth; it is about building sustainable prosperity.

Nobody in the prompt claimed that it was about stopping growth.

## 2.6 Template stacking

Several rhetorical templates appear together.

> This is not simply a technical challenge. It is a deeply human one—one that extends beyond efficiency and speaks to trust, agency, and the future of work.

This matters because “not just X but Y” may be part of a larger cluster of learned assistant-style rhetoric.

## 2.7 Legitimate pedagogical contrast

The structure is useful because the common misconception is well established.

> A virus is not a cell; it depends on host-cell machinery to reproduce.

These cases must not be counted as failures.

---

# 3. Main hypotheses

## H1: Pretraining-frequency hypothesis

The construction may be common in:

* opinion journalism;
* marketing copy;
* motivational writing;
* speeches;
* academic introductions;
* explanatory essays;
* SEO articles;
* book reviews;
* advocacy writing.

The model may simply learn that explanatory prose frequently uses concessive or corrective framing.

### Prediction

Base models should exhibit the behavior even without instruction tuning, particularly when continuing article-like prose.

### Strong test

Compare base and instruct checkpoints from the same family:

* Qwen base vs instruct;
* Llama base vs instruct;
* Mistral base vs instruct;
* Gemma base vs instruction-tuned;
* OLMo base vs instruct.

If base models already show a high rate, pretraining is a major source.

---

## H2: Supervised fine-tuning hypothesis

Instruction-response datasets may disproportionately contain polished explanatory answers using constructions such as:

* “not only … but also”;
* “it is important to note”;
* “rather than simply”;
* “goes beyond”;
* “at its core”;
* “more than just”.

An instruction-tuned model may therefore learn an “ideal answer register,” not merely task-following.

### Prediction

The rate should rise sharply from base to supervised-fine-tuned checkpoints, even before preference optimization.

### Strong test

Use model families that publish intermediate stages, where possible:

* base;
* SFT;
* DPO or preference-tuned;
* RLHF/RLAIF final model.

---

## H3: Preference-reward hypothesis

Human and model evaluators may reward this framing because it creates the appearance of:

* nuance;
* depth;
* completeness;
* balance;
* rhetorical polish;
* additional information.

There is already substantial evidence that preference evaluators and reward models can favor superficial properties such as response length, and that such biases can propagate into optimized policies. ([[arXiv](https://arxiv.org/abs/2310.10076?utm_source=chatgpt.com)][2])

### Prediction

Given two semantically equivalent responses, reward models will assign higher scores to the one containing contrastive negation.

### Critical matched-pair test

Version A:

> The project improved payment accuracy and gave teams better visibility into discrepancies.

Version B:

> The project was not merely about improving payment accuracy; it also gave teams better visibility into discrepancies.

Keep facts, length, and information as controlled as possible.

Evaluate both using:

* open reward models;
* LLM judges;
* human annotators;
* API judges.

If B wins despite no informational advantage, reward-based stylistic amplification becomes plausible.

---

## H4: Elaboration-scaffold hypothesis

The phrase may function as a convenient planning device.

After generating:

> “X is not just A…”

the model has syntactically committed itself to supplying another point. This provides an easy mechanism for producing a multi-dimensional answer.

It is essentially a **self-created continuation obligation**.

### Prediction

The phrase should become more common when the model is asked to:

* elaborate;
* be comprehensive;
* sound insightful;
* explain significance;
* provide nuance;
* write a compelling introduction.

It should become less common when asked to:

* state only literal facts;
* use atomic propositions;
* avoid rhetorical framing;
* produce one-clause sentences;
* answer in a table.

### Mechanistic possibility

The construction may reduce local uncertainty because “not just X, but…” strongly predicts an upcoming expansion. Measure token entropy before and after the construction.

---

## H5: Contrast-as-coherence hypothesis

Models may use contrastive discourse relations to make independently generated points appear connected.

Suppose the model retrieves two facts:

* X involved technological change.
* X changed institutions.

A human writer might construct an explicit causal relation. The model can cheaply connect them as:

> “It was not only technological; it was institutional.”

Thus, negation may serve as **discourse glue**.

### Prediction

OCN should increase when the prompt requires synthesis across multiple dimensions, especially when the dimensions have weak causal relations.

---

## H6: Anti-triviality hypothesis

Post-trained models may learn that simply answering the obvious interpretation is judged shallow. They therefore reject an imagined “surface interpretation” and present a “deeper interpretation.”

The underlying latent template may be:

1. Identify the obvious answer.
2. Down-rank it as incomplete.
3. Present a supposedly deeper answer.

### Prediction

The construction will occur particularly often in prompts containing:

* “meaning”;
* “importance”;
* “significance”;
* “impact”;
* “what this shows”;
* “why it matters”;
* “deeper”;
* “subtle”.

---

## H7: Synthetic-data recursion hypothesis

New models are increasingly trained or fine-tuned on synthetic text. If earlier assistant models overused these constructions, later models may inherit and amplify them.

### Prediction

Models trained on more recent synthetic instruction corpora may use OCN more than:

* base models;
* older models;
* models trained on more human-authored instruction data.

This will be difficult to prove for closed models but can be tested experimentally through controlled fine-tuning.

---

## H8: Assistant-persona hypothesis

The behavior may be tied specifically to the “helpful assistant” role rather than general language modeling.

### Prediction

The same instruct model should show different rates under:

* assistant;
* newspaper reporter;
* technical manual;
* court transcript;
* terse analyst;
* casual human;
* encyclopedia entry;
* dialogue character.

If assistant framing produces the highest rate, the tendency is persona-conditioned.

---

## H9: Decoding hypothesis

Sampling settings may influence stylistic template reuse.

### Prediction

OCN could increase under low-temperature decoding because highly conventional rhetorical templates have high probability. Alternatively, it could increase at high temperature because the model elaborates more. This should be measured rather than assumed.

Test:

* greedy;
* temperature 0.2, 0.5, 0.8, 1.1;
* top-p 0.8, 0.9, 0.95, 1.0;
* repetition penalty;
* min-p;
* beam search where supported.

---

## H10: Internal negation-feature hypothesis

The stylistic construction may correspond to an identifiable activation direction or circuit.

Recent work suggests that models can encode and process negation through multiple mechanisms, including constructing representations of negative phrases and suppressing concepts associated with the negated material. ([[arXiv](https://arxiv.org/html/2605.03052v1?utm_source=chatgpt.com)][3]) Separate work indicates that stylistic properties can sometimes be controlled through activation-space directions. ([[arXiv](https://arxiv.org/html/2603.03324v1?utm_source=chatgpt.com)][4])

However, your phenomenon is not ordinary semantic negation. It is **rhetorical negation**, so it may have a separable representation.

### Prediction

Activations before OCN constructions should be linearly distinguishable from activations before ordinary affirmative elaboration.

---

# 4. Core experimental design

Build the project in five layers.

## Study A: Establish and quantify the phenomenon

### Dataset

Create a prompt bank of approximately **3,000 prompts**, stratified across 12 categories:

| Category                  | Example                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Factual explanation       | Explain photosynthesis.                                    |
| Historical significance   | Why was the printing press important?                      |
| Concept definition        | What is leadership?                                        |
| Personal advice           | How should someone handle career uncertainty?              |
| Product description       | Describe the value of a budgeting app.                     |
| Literary analysis         | What does the green light represent in *The Great Gatsby*? |
| Scientific interpretation | Why is CRISPR important?                                   |
| Business writing          | Explain the impact of a process improvement.               |
| Moral questions           | What makes an apology sincere?                             |
| Summarization             | Summarize this passage.                                    |
| Procedural instructions   | Explain how to reset a router.                             |
| Creative writing          | Write an introduction about cities.                        |

For every semantic topic, generate prompt variants:

1. neutral;
2. “brief”;
3. “comprehensive”;
4. “deep and nuanced”;
5. “plain factual language”;
6. “do not use contrastive framing”;
7. explicit misconception present;
8. no misconception present.

This gives a controlled factorial dataset.

### Models

Run at least:

* Qwen2.5 or Qwen3 base and instruct at 1.5B/3B/7B scale;
* Llama 3.1 or 3.2 base and instruct;
* Mistral 7B base and instruct;
* Gemma base and instruction-tuned;
* OLMo base and instruct, because training information is comparatively transparent;
* two to four closed APIs for external validity.

Use 3–5 random seeds per prompt for sampled decoding.

A practical first pass is:

* 1,000 prompts;
* 6 open checkpoints;
* 3 samples each;
* around 18,000 generations.

Scale later to 100,000+ generations.

---

# 5. Automated detection system

Do not rely only on exact string matching.

## 5.1 Lexical detector

Search constructions including:

```text
not just X but Y
not merely X but Y
not only X but also Y
isn't simply
doesn't merely
goes beyond
more than just
rather than simply
not so much X as Y
not X alone
far from being merely
cannot be reduced to
is not simply a matter of
```

## 5.2 Syntactic detector

Use dependency parsing or constituency parsing to identify:

* negative marker;
* first complement;
* adversative or additive conjunction;
* second complement.

## 5.3 Semantic classifier

Train a small classifier to label:

* genuine contrast;
* presupposed contrast;
* empty contrast;
* redundant contrast;
* false correction;
* non-OCN negation.

Start with 2,000 manually annotated sentences.

Good classifier options:

* DeBERTa-v3-base;
* ModernBERT;
* small instruction model with constrained JSON output;
* embedding model plus logistic regression.

## 5.4 Clause-relation scoring

For each construction, compute:

### Semantic similarity

Embed X and Y. Very high similarity may indicate empty contrast.

### Entailment

Use an NLI model to test:

* Does X entail Y?
* Does Y entail X?
* Are they contradictory?
* Are they merely additive?

### Prompt support

Determine whether the prompt introduced or implied X.

A contrast is more suspicious when:

[
P(X\text{ asserted or implied by prompt}) \approx 0
]

but the answer frames X as something needing correction.

## 5.5 Human annotation

Have annotators answer:

1. Was the rejected idea present in the prompt?
2. Is it a common misconception relevant to the question?
3. Are X and Y genuinely distinct?
4. Does negation add meaning?
5. Could the sentence be rewritten affirmatively without losing information?
6. Does the sentence create a straw position?
7. Does it sound formulaic or machine-like?

Use a 5-point scale rather than binary labels.

Measure Krippendorff’s alpha or Fleiss’ kappa.

---

# 6. Primary metrics

## OCN frequency

[
\text{OCN Rate}
===============

\frac{\text{Responses containing OCN}}{\text{Total responses}}
]

## OCN density

[
\text{OCN Density}
==================

\frac{\text{OCN constructions}}{1{,}000\text{ generated tokens}}
]

## Unsupported contrast rate

[
\text{UCR}
==========

\frac{\text{OCNs where rejected proposition is unsupported}}
{\text{All OCNs}}
]

## Semantic redundancy score

Use cosine similarity and bidirectional entailment between X and Y.

## Deletion invariance

Delete the negative scaffold and rewrite:

> “X is not just A; it is B.”

as:

> “X is A and B.”

Then test whether meaning, correctness, or informativeness changes.

If human raters judge the two versions equivalent, the negation was probably rhetorically unnecessary.

## False-depth score

Ask raters to judge:

* perceived depth;
* actual additional information;
* clarity;
* naturalness.

Define:

[
\text{False Depth}
==================

## \text{Perceived Depth}

\text{Information Gain}
]

This may become the most interesting metric in the paper.

## Template concentration

Measure how much of a model’s OCN output is captured by its top 10 or top 50 lexical templates.

A high concentration would support the idea that this is a memorized assistant register.

---

# 7. Base-versus-instruct causal study

This is the most important first experiment.

For each model family, use exactly the same prompts and decoding parameters.

Compare:

[
\Delta_{\text{SFT}}
===================

## \text{OCN Rate}_{\text{instruct}}

\text{OCN Rate}_{\text{base}}
]

Control for the fact that base models may not naturally answer instructions. Use three evaluation modes:

1. direct instruction;
2. few-shot completion;
3. article continuation.

### Interpretation

* **High in base and instruct:** likely pretraining-driven.
* **Low in base, high in instruct:** likely instruction-tuning-driven.
* **Moderate in SFT, highest after preference tuning:** likely reward amplification.
* **Only high in assistant chat templates:** role or formatting effect.

Also test the same instruct model with and without the official chat template. Chat templates themselves may prime a particular register.

---

# 8. Preference and reward-model experiment

This experiment can make the project substantially more novel.

## 8.1 Construct controlled pairs

For 1,000 answers, produce four versions:

### A. Plain affirmative

> The policy affects emissions, industrial investment, and energy security.

### B. OCN version

> The policy is not just about emissions; it also affects industrial investment and energy security.

### C. Empty OCN version

> The policy is not merely about reducing emissions; it is about lowering the amount of carbon released.

### D. Genuine contrast version

> Although the prompt describes the policy as an emissions measure, it also affects industrial investment and energy security.

Versions should be matched on:

* facts;
* approximate token length;
* grammatical quality;
* tone;
* number of propositions.

## 8.2 Score with reward models

Use several open reward models rather than one.

Measure:

[
\Delta R = R(\text{OCN}) - R(\text{plain})
]

Run a regression:

[
R =
\beta_0
+\beta_1 \text{OCN}
+\beta_2 \text{Length}
+\beta_3 \text{Correctness}
+\beta_4 \text{Information Count}
+\beta_5 \text{Fluency}
+\epsilon
]

The crucial coefficient is (\beta_1).

## 8.3 LLM judge experiment

Randomize:

* response order;
* labels A/B;
* answer length;
* wording;
* evaluator model.

Ask judges separately about:

* correctness;
* depth;
* clarity;
* naturalness;
* preference;
* professionalism.

This lets you identify the exact dimension on which OCN is rewarded.

## 8.4 Human experiment

Recruit 50–150 participants through Prolific or a smaller convenience sample.

Test two conditions:

* authorship undisclosed;
* participants told one response may be AI-generated.

This can reveal whether OCN increases perceived sophistication but also increases perceived artificiality.

---

# 9. Prompt-factor experiment

Use a factorial design.

## Independent variables

### Requested depth

* brief;
* normal;
* detailed;
* nuanced;
* profound.

### Audience

* child;
* general reader;
* executive;
* academic;
* domain expert.

### Genre

* encyclopedia;
* marketing;
* academic;
* social media;
* technical documentation;
* casual conversation.

### Epistemic framing

* explain;
* evaluate;
* interpret;
* summarize;
* define;
* argue;
* reflect.

### Contrast availability

* explicit misconception;
* implicit common misconception;
* no misconception;
* explicitly prohibit invented contrasts.

### Length target

* 30;
* 75;
* 150;
* 300 words.

Fit a mixed-effects logistic regression:

[
\operatorname{logit}(P(\text{OCN}))
===================================

\beta_0
+\beta_1\text{Depth}
+\beta_2\text{Genre}
+\beta_3\text{Length}
+\beta_4\text{Model}
+\beta_5\text{Instruction Stage}
+u_{\text{Topic}}
]

This determines which communicative conditions trigger the style.

---

# 10. Decoding analysis

For a subset of 500 prompts, generate 20 samples per configuration.

Measure OCN as a function of:

* temperature;
* top-p;
* top-k;
* repetition penalty;
* output length;
* seed.

Also inspect token probabilities.

For prompts that frequently produce OCN, record the probability of tokens such as:

```text
not
isn't
doesn't
merely
simply
only
beyond
rather
```

at the relevant generation position.

### Key question

Is OCN caused by:

* one unusually probable token such as “not”;
* a highly probable multi-token phrase;
* later planning that already encodes the complete construction?

Use teacher forcing to obtain phrase-level log probabilities for:

> “X is important because…”

versus

> “X is not just important because…”

A low surprisal for the latter would support template conventionalization.

---

# 11. Corpus-origin investigation

Exact pretraining corpora are unavailable for many models, but this can still be investigated.

## 11.1 Human corpora

Estimate OCN frequency in:

* Wikipedia;
* news;
* academic papers;
* Reddit;
* books;
* corporate websites;
* advertising;
* Medium-style essays;
* speeches;
* Stack Exchange;
* instruction datasets;
* model-generated datasets.

Normalize by tokens and sentence count.

## 11.2 Domain-matching analysis

Train a classifier predicting source domain from sentences containing OCN.

Ask whether model-generated OCN resembles:

* marketing language;
* opinion columns;
* educational explanations;
* corporate leadership prose;
* assistant responses.

## 11.3 Nearest-neighbour retrieval

For generated phrases, search large public corpora for close matches using:

* MinHash;
* BM25;
* sentence embeddings;
* suffix arrays where feasible.

Do not claim memorization merely from similarity. Instead measure whether model outputs cluster around highly conventional human templates.

## 11.4 Fine-tuning corpus audit

Inspect open instruction datasets such as:

* UltraChat;
* OpenAssistant;
* Dolly;
* FLAN-style mixtures;
* synthetic instruction datasets;
* preference datasets.

Compute OCN rates separately in:

* prompts;
* chosen answers;
* rejected answers;
* human answers;
* synthetic answers.

The most decisive signal would be:

[
P(\text{OCN}\mid \text{chosen})

>

P(\text{OCN}\mid \text{rejected})
]

after controlling for answer length and quality.

That would directly indicate preference-selection pressure.

---

# 12. Controlled synthetic training experiment

This is likely the cleanest causal experiment you can run with Colab-scale compute.

## Base model

Use a 0.5B–1.5B model such as:

* Qwen 0.5B or 1.5B;
* SmolLM;
* TinyLlama;
* OLMo small checkpoint.

## Create three SFT datasets

Use the same 20,000–50,000 prompts.

### Dataset Plain

All answers use direct affirmative prose.

### Dataset OCN

Insert one semantically justified OCN construction into a large percentage of answers.

### Dataset Empty-OCN

Insert rhetorically impressive but semantically redundant OCN constructions.

Fine-tune three LoRA adapters with identical hyperparameters.

### Evaluate

Test on entirely new topics and prompts.

Measure:

* OCN frequency;
* generalization across domains;
* response quality;
* reward score;
* perceived depth;
* whether the construction appears even when prohibited.

This determines how easily the tendency is acquired.

---

# 13. Controlled preference-optimization experiment

Starting from the same SFT model, create preference pairs.

## Condition 1: OCN preferred

Chosen answers contain OCN; rejected answers are plain.

## Condition 2: Plain preferred

Chosen answers are direct; rejected answers contain unnecessary OCN.

## Condition 3: Content-only preference

Pairs differ in correctness but not style.

Train with DPO or a lightweight preference method.

### Important result

Measure whether merely 2,000–10,000 preference pairs can alter OCN rates on unrelated topics.

If a small amount of preference optimization strongly changes the rate, that supports the hypothesis that stylistic quirks can emerge as reward-model shortcuts.

---

# 14. Mechanistic interpretability study

Do this only after behavioral results identify a stable model and trigger set.

Use a 1B–3B open model initially.

## 14.1 Contrastive activation dataset

Construct matched prefixes:

### OCN trajectory

> The significance of the policy is not merely…

### Plain trajectory

> The significance of the policy includes…

Match them for topic, position, and approximate syntax.

Collect residual-stream activations at:

* final prompt token;
* first token of the answer;
* token before “not”;
* “not” token;
* “just/merely” token;
* “but” token;
* beginning of Y.

## 14.2 Linear probes

Train probes to predict whether the completion will contain OCN.

Questions:

* At which layer does OCN become predictable?
* Is it predictable before the word “not” appears?
* Does the representation generalize across topics and templates?
* Does it distinguish rhetorical negation from factual negation?

If OCN is predictable at the final prompt token, it suggests high-level planning rather than local word association.

## 14.3 Activation patching

Take:

* a prompt that normally triggers OCN;
* a matched prompt that produces direct prose.

Patch activations across layers and token positions.

Measure whether patching changes:

[
P(\text{“not” or OCN continuation})
]

This can locate causally relevant layers.

## 14.4 Steering vector

Construct:

[
v_{\text{OCN}}
==============

## \mathbb{E}[h_{\text{OCN}}]

\mathbb{E}[h_{\text{plain}}]
]

Add and subtract the vector during generation.

Test whether:

* positive steering increases OCN;
* negative steering suppresses OCN;
* factual negation remains intact;
* answer quality changes;
* other rhetorical templates move simultaneously.

If steering affects “goes beyond,” “not merely,” and “more than just” together, you may have found a broader **rhetorical profundity direction**, not a negation-specific direction.

## 14.5 Attention-head analysis

Inspect whether particular heads:

* attend from “not” to X;
* attend from “but” to X;
* carry the first clause into the second;
* predict additive expansion.

Recent mechanistic work on semantic negation provides methods such as activation patching, ablation, logit-lens inspection, and head-level analysis that can be adapted here. ([[arXiv](https://arxiv.org/pdf/2605.03052?utm_source=chatgpt.com)][5])

## 14.6 Causal ablation

Ablate candidate heads or MLP components and measure:

* OCN reduction;
* factual negation accuracy;
* general fluency;
* contrastive reasoning;
* other discourse markers.

The ideal result is a selective intervention that reduces rhetorical negation without breaking semantic negation.

---

# 15. Is the model using OCN to think?

A particularly interesting experiment is to separate internal planning from final phrasing.

Ask models to generate:

1. a private outline followed by an answer;
2. an answer directly;
3. a structured list followed by prose;
4. JSON facts followed by prose;
5. one proposition per line;
6. an answer rewritten from its own factual outline.

### Hypothesis

OCN will be lower when content planning is externally structured. This would suggest the template helps the model organize multiple ideas during generation.

Compare:

[
P(\text{OCN}\mid \text{direct generation})
]

against:

[
P(\text{OCN}\mid \text{fact plan supplied})
]

You can also use hidden chain-of-thought alternatives such as concise bullet planning, without needing to inspect proprietary reasoning traces.

---

# 16. Counterfactual rewriting experiment

For every OCN response, create three rewrites:

1. remove negation;
2. replace it with an additive conjunction;
3. replace it with a precise causal or categorical relationship.

Example:

**Original**

> The reform was not merely administrative; it reshaped the relationship between citizens and the state.

**Additive**

> The reform changed administration and reshaped the relationship between citizens and the state.

**Causal**

> By changing administrative authority, the reform reshaped the relationship between citizens and the state.

Have evaluators compare:

* factuality;
* clarity;
* depth;
* concision;
* persuasiveness;
* naturalness.

This reveals whether OCN is useful or merely masks an underspecified relationship.

My expectation is that causal rewrites will often be clearer, while OCN versions may score higher on perceived polish.

---

# 17. Cross-lingual experiment

Generate equivalent responses in:

* English;
* Hindi;
* Bengali;
* Spanish;
* French;
* German;
* Chinese.

Ask whether the construction:

* appears directly in each language;
* appears as translationese;
* is strongest in English;
* transfers through multilingual instruction tuning.

For Hindi, examine patterns such as:

> “यह सिर्फ़ X नहीं है, बल्कि Y भी है।”

If English-centric multilingual models show the same rate across languages, this may indicate transferred assistant style rather than language-specific natural usage.

---

# 18. Longitudinal and model-family study

Compare models released across several years, where accessible.

Questions:

* Has OCN increased over successive assistant generations?
* Do newer preference-tuned models use it more?
* Do reasoning models use it less in factual tasks but more in reflective explanations?
* Do distilled models inherit the teacher’s rate?
* Does a student amplify the teacher’s most common rhetorical templates?

A teacher–student distillation experiment would be especially useful:

1. generate SFT data from an OCN-heavy teacher;
2. train a small student;
3. compare teacher and student rates;
4. filter OCN from the synthetic corpus;
5. retrain and compare.

This tests stylistic inheritance through synthetic data.

---

# 19. Recommended statistical analysis

Use a mixed-effects model rather than reporting raw percentages alone.

For response (i):

[
\text{OCN}_i \sim
\text{Model Stage}
+\text{Model Size}
+\text{Prompt Type}
+\text{Genre}
+\text{Requested Length}
+\text{Temperature}
+\text{Assistant Persona}
+(1|\text{Topic})
+(1|\text{Prompt Template})
]

For reward-model scores:

[
R_i \sim
\text{OCN Type}
+\text{Length}
+\text{Information Count}
+\text{Correctness}
+\text{Fluency}
+(1|\text{Question})
+(1|\text{Reward Model})
]

Report:

* odds ratios;
* bootstrap confidence intervals;
* effect sizes;
* correction for multiple comparisons;
* inter-annotator agreement;
* model-family-specific effects.

Avoid treating each generated answer as fully independent when several answers come from the same prompt.

---

# 20. Minimum viable paper

A strong first paper does not need every experiment above.

## Paper-sized core

### Experiment 1: Measurement

Compare base and instruct versions across four model families and 1,000 prompts.

### Experiment 2: Prompt triggers

Test depth, genre, length, misconception presence, and persona.

### Experiment 3: Reward preference

Use controlled OCN/plain pairs with open reward models, API judges, and humans.

### Experiment 4: Controlled DPO

Show that preferring OCN in a small preference dataset causes out-of-domain OCN proliferation.

### Experiment 5: Early mechanistic result

Show that future OCN generation can be decoded from middle- or late-layer activations before the construction begins.

That would support a paper with a claim resembling:

> Contrastive negation is not solely a lexical habit inherited from pretraining. Instruction and preference tuning amplify it because it functions as a rewarded signal of elaboration and perceived nuance, and models plan the structure before emitting explicit negation.

That claim must remain provisional until the experiments support it.

---

# 21. Colab-Pro implementation plan

## Compute-friendly stack

* `transformers`
* `vllm` where available
* `accelerate`
* `peft`
* `trl`
* `bitsandbytes`
* `sentence-transformers`
* `spaCy`
* `statsmodels`
* `scikit-learn`
* `TransformerLens` or `nnsight`
* `datasets`

## Phase 1: One-week pilot

Use:

* Qwen 1.5B base;
* Qwen 1.5B instruct;
* Mistral 7B instruct in 4-bit;
* one API model;
* 300 prompts;
* 3 samples each.

Manually label 500 outputs.

Deliverables:

* regex detector;
* first taxonomy;
* base/instruct OCN rates;
* prompt-category heatmap;
* 50 representative examples.

## Phase 2: Main behavioral dataset

Scale to:

* 3,000 prompts;
* 8–12 checkpoints;
* approximately 100,000 outputs.

Train the semantic classifier and run mixed-effects analysis.

## Phase 3: Reward study

Create 2,000 controlled answer sets and evaluate them with:

* three open reward models;
* two API judges;
* human raters on a subset.

## Phase 4: Training intervention

LoRA fine-tune a 0.5B–1.5B model under:

* plain SFT;
* OCN SFT;
* plain-preferred DPO;
* OCN-preferred DPO.

## Phase 5: Mechanistic study

Use the strongest small model from Phase 4 because controlled adapters provide clean causal contrasts.

---

# 22. Approximate cost

Open-model generation and LoRA training should be manageable with Colab Pro, particularly at 0.5B–3B scale.

A reasonable API budget:

| Use                        |        Approximate scope |
| -------------------------- | -----------------------: |
| External model generations |   5,000–10,000 responses |
| LLM classification         | 5,000 difficult examples |
| Pairwise judging           |       5,000–15,000 pairs |
| Adjudication               |    1,000 ambiguous cases |

To control cost:

* use local regex and classifiers first;
* send only uncertain examples to API models;
* cache all outputs;
* use multiple API judges only for the core matched-pair set;
* reserve human annotation for 1,000–2,000 carefully selected cases.

---

# 23. Expected outcomes and their interpretations

## Outcome A: Base and instruct models are equally high

Likely cause: pretraining-domain frequency or basic next-token dynamics.

Next step: corpus matching and decoding analysis.

## Outcome B: Instruct models are much higher

Likely cause: supervised instruction data or assistant persona.

Next step: audit instruction datasets and compare chat templates.

## Outcome C: Preference-tuned models are highest

Likely cause: reward-model or evaluator preference.

Next step: controlled reward and DPO studies.

## Outcome D: OCN appears mostly in “deep,” “nuanced,” and “significance” prompts

Likely cause: learned rhetorical depth heuristic.

Next step: false-depth rating study.

## Outcome E: OCN is predictable before generation and steerable

Likely cause: planned high-level style representation.

Next step: cross-model steering and circuit localization.

## Outcome F: OCN disappears when a factual plan is supplied

Likely cause: discourse-planning scaffold.

Next step: compare planning formats and relation specification.

## Outcome G: Reward models prefer OCN while humans do not

Likely cause: alignment-induced reward hacking.

This would be the most important and publishable result.

## Outcome H: Humans also prefer OCN in blind evaluation

Then the behavior may be an exaggerated model version of a genuine human preference rather than a purely artificial defect.

The research should report that honestly.

---

# 24. Main threats to validity

### Length confound

OCN versions are usually longer. Match length or include length as a covariate.

### Information confound

The second clause may add a genuinely new proposition. Keep proposition count controlled.

### Topic confound

Abstract topics naturally invite contrast more than procedural topics. Use topic-level random effects.

### Base-model comparability

Base models may not follow instructions. Evaluate both instruction completion and natural document continuation.

### Detector overreach

Not every “not only” sentence is undesirable. Separate occurrence from pragmatic misuse.

### API opacity

Closed models cannot establish training-stage causality. Use them only for external validation.

### Judge circularity

An LLM judge may share the same stylistic bias being studied. Include humans and objective linguistic metrics.

### English-only conclusions

Do not generalize globally without cross-lingual tests.

---

# 25. Potential paper contributions

A strong final project could contribute:

1. **A formal definition and taxonomy** of overgeneralized contrastive negation.
2. **A benchmark** measuring supported, unsupported, redundant, and false contrast.
3. **A cross-stage analysis** of base, SFT, and preference-tuned models.
4. **Evidence about reward-model preference** for rhetorical negation.
5. **A controlled causal training study** showing how the tendency is learned.
6. **A mechanistic analysis** of when the model commits to the construction.
7. **A mitigation method** that suppresses empty contrast without harming legitimate negation.

A suitable title:

> **Not Just a Figure of Speech: Tracing Overgeneralized Contrastive Negation in Instruction-Tuned Language Models**

Alternative:

> **The Strawman in the Sentence: Why Language Models Invent Contrasts**

Or more technical:

> **From Pretraining to Preference Optimization: A Causal Study of Contrastive Negation in LLM-Generated Text**

The highest-signal version of the project is the combination of **base-versus-instruct comparison, controlled reward-model pairs, small-scale DPO intervention, and pre-token activation probing**. Together, these can move the work from a stylistic observation to a causal account.

[1]: https://arxiv.org/html/2410.16107v1?utm_source=chatgpt.com "Do LLMs write like humans? Variation in grammatical and ..."
[2]: https://arxiv.org/abs/2310.10076?utm_source=chatgpt.com "Verbosity Bias in Preference Labeling by Large Language Models"
[3]: https://arxiv.org/html/2605.03052v1?utm_source=chatgpt.com "How Language Models Process Negation"
[4]: https://arxiv.org/html/2603.03324v1?utm_source=chatgpt.com "Controlling Chat Style in Language Models via Single ..."
[5]: https://arxiv.org/pdf/2605.03052?utm_source=chatgpt.com "How Language Models Process Negation"
