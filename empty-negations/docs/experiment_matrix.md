# Experiment Matrix

## Phase 1: Local Pilot

Goal: verify that the detector, prompt bank, and analysis workflow work before large-scale generation.

| Component | Recommended size |
| --- | ---: |
| Prompts | 120-300 |
| Models | 2-4 |
| Samples per prompt | 3 |
| Generations | 720-3,600 |
| Manual labels | 200-500 |

Suggested models:

- one base checkpoint;
- one instruct checkpoint from the same family;
- one stronger instruct model;
- optionally one closed API model for external validity.

## Phase 2: Behavioral Study

| Factor | Levels |
| --- | --- |
| Stage | base, instruct, preference-tuned if available |
| Prompt mode | direct instruction, few-shot completion, article continuation |
| Requested depth | brief, normal, detailed, nuanced |
| Genre/persona | assistant, encyclopedia, technical manual, marketing, analyst, casual human |
| Contrast availability | explicit misconception, common misconception, no misconception, prohibit invented contrast |
| Length target | 30, 75, 150, 300 words |
| Decoding | greedy, temp 0.2, temp 0.7, temp 1.0 |

Primary model:

```text
OCN ~ stage + prompt_mode + depth + genre + contrast_availability
    + length_target + temperature + (1|topic) + (1|prompt_template)
```

## Phase 3: Reward Preference

Construct matched answer sets:

- A: plain affirmative;
- B: pragmatically valid OCN;
- C: empty/redundant OCN;
- D: explicit genuine contrast.

Score dimensions separately:

- correctness;
- clarity;
- depth;
- naturalness;
- professionalism;
- overall preference.

Critical coefficient:

```text
reward_score ~ ocn_type + length + information_count + fluency + (1|question)
```

## Phase 4: Controlled Training

Use a 0.5B-1.5B model with LoRA adapters.

SFT conditions:

- plain answers;
- justified OCN answers;
- empty OCN answers.

DPO conditions:

- OCN preferred;
- plain preferred;
- content-only preference.

Measure whether stylistic preference transfers to unseen topics and whether explicit anti-OCN instructions still fail.

## Phase 5: Mechanistic Pilot

Use matched prefixes:

- OCN trajectory: `The significance of X is not merely...`
- plain trajectory: `The significance of X includes...`

Probe residual-stream activations for future OCN generation at:

- final prompt token;
- first answer token;
- token before `not`;
- `not`;
- `merely/just`;
- `but`;
- beginning of Y.

Questions:

- Is OCN predictable before lexical negation?
- Does the signal generalize across topics?
- Does it distinguish rhetorical negation from factual negation?
- Can activation steering reduce OCN without harming legitimate negation?
