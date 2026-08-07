# CalibrateQwen

## Know When We Do Not Know: On-Policy Distillation of Calibrated Reasoning into Small Language Models

We built CalibrateQwen to study whether a compact language model can acquire useful uncertainty estimates together with multiple-choice reasoning ability. We train a Qwen student to return an answer, a concise justification, a confidence value, and an abstention decision. We compare hard-label supervised fine-tuning, fixed teacher-completion training, off-policy soft-target distillation, and on-policy distillation. We evaluate accuracy and structured-output reliability together with calibration, error detection, selective prediction, domain transfer, and option-order robustness.

We designed this repository as an executable research artifact. We keep data construction, Tinker training, sampling, metrics, plots, and Colab orchestration in separate modules with explicit schemas. We published the curated dataset as a private Hugging Face dataset at [`ritwikraha/calibrate-qwen-curated`](https://huggingface.co/datasets/ritwikraha/calibrate-qwen-curated). We verified every published configuration through a complete Hugging Face load.

## Research question

We ask one central question:

> Can on-policy distillation transfer answer accuracy, calibrated uncertainty, and useful selective-prediction behavior from a stronger teacher into a compact Qwen student?

We operationalize useful uncertainty through observable behavior. A useful confidence score separates correct and incorrect predictions, supports low risk at reduced coverage, remains stable under option permutations, and generalizes from the in-domain mixture to a held-out source.

We test four hypotheses:

1. We expect hard-label SFT to improve structured-output validity and answer accuracy while encouraging confidence concentration near one.
2. We expect fixed teacher completions to transfer concise rationales and agreement-derived confidence targets.
3. We expect off-policy soft targets to transfer a richer token distribution than one-hot completion labels.
4. We expect on-policy distillation to improve supervision on trajectories sampled from the student's own distribution.

## Verified state of the project

We completed and verified the dataset layer before implementing training. The experiment-ready release contains:

| Split | Records | Purpose |
|---|---:|---|
| Train | 9,134 | In-domain training mixture |
| Phase 1 train | 2,000 | Economical pipeline and SFT validation |
| Validation | 1,000 | Model selection and calibration decisions |
| In-domain test | 1,163 | Held-out ARC, OpenBookQA, and MMLU evaluation |
| OOD test | 3,000 | Held-out CommonsenseQA evaluation |
| Option-shuffled test | 2,326 | Two deterministic permutations per in-domain example |

We generated teacher targets for all 2,000 Phase 1 records with `Qwen/Qwen3.5-9B`. We requested five independent samples per question, producing 10,000 teacher samples. We verified the following teacher-target statistics:

| Statistic | Value |
|---|---:|
| Unique records | 2,000 |
| Requested samples | 10,000 |
| Valid parsed samples | 9,914 |
| Parse rate | 99.14% |
| Modal teacher accuracy | 89.80% |
| Mean agreement confidence | 95.79% |
| Unanimous records | 1,724 |
| Unanimous rate | 86.20% |

We use these statistics as data diagnostics. The teacher accuracy value measures agreement between the modal teacher answer and the public gold label. The agreement value measures concentration across repeated teacher samples. A confidently incorrect teacher answer remains valuable because it exposes a failure mode that a calibration method should identify.

## System architecture

We organize the project as a sequence of typed artifacts:

```text
Public MCQ datasets
        |
        v
Normalized source records
        |
        v
Curated and deduplicated splits
        |
        +---------------------> OOD and option-shuffled tests
        |
        v
Repeated teacher sampling
        |
        v
Conversation targets and prompt-only datasets
        |
        +----------+------------+----------------+
        |          |            |                |
        v          v            v                v
Hard SFT   Teacher SFT   Off-policy soft   On-policy KL
        |          |            |                |
        +----------+------------+----------------+
                           |
                           v
                 Structured model sampling
                           |
                           v
          Calibration and selective-prediction report
```

We keep normalized records independent of the Tinker renderer. We apply renderer-specific tokenization at training and sampling time. This boundary preserves dataset portability and gives every model family control over its own chat template and stop sequences.

## Data design

We train on ARC-Challenge, OpenBookQA, and selected MMLU examples. We reserve CommonsenseQA as an out-of-domain source. We retain held-out examples from each training source for in-domain evaluation. We create deterministic option permutations for robustness analysis.

Every base record follows this schema:

```json
{
  "id": "65e031c0d44323dc",
  "source": "openbookqa",
  "source_split": "train",
  "source_id": "source-row-id",
  "subject": "science_reasoning",
  "question": "Which option best explains the observation?",
  "choices": ["choice A", "choice B", "choice C", "choice D"],
  "answer_index": 1,
  "answer_label": "B",
  "prompt": "Question: ...",
  "metadata": {},
  "split": "train"
}
```

We assign stable IDs from normalized content and source identity. Stable IDs support deterministic sampling, cross-file joins, interrupted-run resume, and parent-child relationships for perturbations.

### Curation rules

We run curation before paid teacher sampling. We validate required fields, question length, choice count, choice length, answer-index range, prompt length, and duplicate choice text. We normalize whitespace, regenerate answer labels, and regenerate formatted prompts. We remove duplicate questions across splits. We apply source limits and split limits with deterministic seeded sampling. We record every rejection reason in the curation manifest.

Our issue-aligned curation pass retained 9,134 training records, 1,000 validation records, and 1,163 in-domain test records. The pass removed malformed choice sets, very short questions, and long questions that exceeded the configured context budget. The manifest at `artifacts/issue_curated/manifest.json` contains the exact drop counts.

### Hugging Face configurations

We expose schema-compatible files as separate configurations:

| Configuration | Splits | Schema |
|---|---|---|
| `calibrated_mcq` | `train`, `validation`, `test`, `test_ood` | Standard MCQ record |
| `phase1_2000` | `train` | Standard MCQ record |
| `option_shuffled` | `test` | MCQ record plus parent and perturbation metadata |
| `teacher_phase1` | `train` | MCQ record plus repeated teacher metadata |

We separated these configurations because Hugging Face applies one feature schema across every split inside a configuration. The perturbation and teacher columns represent intentional schema extensions.

## Teacher uncertainty targets

We estimate teacher confidence from repeated sampling. For question $x$, teacher samples produce answer labels $a_1, \ldots, a_K$. We define the modal teacher answer as

$$
\hat{a}_T = \operatorname{mode}(a_1, \ldots, a_K)
$$

and agreement confidence as

$$
c_T(x) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[a_k = \hat{a}_T].
$$

For five samples `C, C, C, B, C`, we assign a modal answer of `C` and confidence `0.8`. We retain every raw sample, parsing method, answer count, modal answer, representative justification, and correctness indicator. This record-level audit trail supports later analyses of teacher disagreement and dangerous teacher overconfidence.

## Structured output contract

We train the numeric-confidence variants to emit one JSON object:

```json
{
  "answer": "C",
  "confidence": 0.8,
  "justification": "Option C follows from the causal relation stated in the prompt.",
  "abstain": false
}
```

We constrain `answer` to the option labels present in the prompt. We constrain numeric confidence to the closed interval from zero to one. We constrain justification to one concise sentence. We represent abstention as a JSON boolean.

We implement three confidence representations:

| Representation | Output | Evaluation source |
|---|---|---|
| Numeric | `"confidence": 0.73` | Parsed scalar |
| Bucket | `"confidence": "medium"` | Bucket midpoint |
| Implicit | Confidence field omitted | Normalized answer-token probability |

We map low, medium, and high buckets to `0.25`, `0.65`, and `0.90` during evaluation. We compute implicit confidence by scoring each candidate answer label under the same model and normalizing candidate sequence log probabilities with a softmax.

## Training methods

### Base model

We evaluate `Qwen/Qwen3.5-4B` before training. This run establishes initial answer accuracy, JSON validity, calibration, selective prediction, and robustness. We use the same sampling and parsing implementation for every trained checkpoint.

### Hard-label SFT

We convert each gold record into a structured target with the gold answer, confidence `1.0`, a deterministic short justification, and `abstain: false`. This baseline isolates the effect of supervised structured-output training.

We optimize last-assistant-message cross entropy through the Tinker supervised recipe. This mask matches the Qwen3.5 renderer's sequence-extension behavior. We use LoRA rank 32, a batch size of 32, a peak learning rate of `2e-4`, linear decay, and one epoch for the Phase 1 run.

### Teacher-completion SFT

We convert each repeated-sampling teacher record into a structured completion. The target answer is the modal teacher answer. The target confidence is teacher agreement. The target justification is the representative justification from a sample that selected the modal answer. We set the target abstention flag from a held-out threshold.

This method transfers one fixed completion per prompt. It provides a strong and economical distillation baseline.

### Off-policy soft-target distillation

We use the same fixed teacher-completion sequences and query the teacher distribution at every supervised token position. Tinker retains the highest-probability teacher tokens and trains the student against the resulting soft targets. We configure 20 teacher targets per position and bound concurrent teacher requests.

For target sequence $y$ and prompt $x$, this method approximates token-level cross entropy under the teacher distribution:

$$
\mathcal{L}_{\text{off}} = -\sum_t\sum_{v \in \mathcal{V}_K}
p_T(v \mid x, y_{<t})\log p_S(v \mid x, y_{<t}),
$$

where $\mathcal{V}_K$ contains the retained teacher targets at position $t$.

### On-policy distillation

We sample trajectories from the current student. We evaluate the same sampled tokens under the teacher. We use the reverse KL signal on the student's occupied distribution:

$$
\operatorname{KL}(p_S \Vert p_T)
= \mathbb{E}_{y \sim p_S}\left[\log p_S(y \mid x) - \log p_T(y \mid x)\right].
$$

The Tinker recipe converts the negative reverse KL into token-level advantages and applies an importance-sampling update. We use two rollouts per prompt, 16 prompt groups per batch, 192 output tokens, and a KL coefficient of one. We support initialization from the final SFT checkpoint, which gives us both a pure on-policy run and an SFT followed by on-policy run.

## Evaluation protocol

We evaluate every method on the same in-domain, OOD, and option-shuffled records. We preserve raw completions and parsing diagnostics. We treat schema-invalid outputs as incorrect predictions with confidence zero for threshold analysis. This convention rewards methods that satisfy the deployable output contract.

### Accuracy

We compute exact multiple-choice accuracy:

$$
\operatorname{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{y}_i = y_i].
$$

### Expected calibration error

We partition predictions into ten equal-width confidence bins. For bin $B_m$, we compare empirical accuracy and mean confidence:

$$
\operatorname{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\right|.
$$

We save the count, confidence, and accuracy of every bin for reliability diagrams.

### Brier score

For verbal confidence, we compute a binary correctness Brier score:

$$
\operatorname{BS}_{\text{correct}} = \frac{1}{N}\sum_i(c_i-z_i)^2,
$$

where $z_i$ indicates answer correctness. For normalized option probabilities, we compute the multiclass Brier score:

$$
\operatorname{BS}_{\text{multi}} = \frac{1}{N}\sum_i\sum_k(p_{ik}-y_{ik})^2.
$$

### Negative log-likelihood

We compute multiclass NLL from the normalized probability assigned to the gold answer. We also compute binary correctness NLL for verbal confidence. These values expose severe penalties for high-confidence errors.

### Selective prediction

For threshold $\gamma$, we answer when confidence satisfies $c_i \geq \gamma$. We define coverage as the answered fraction and risk as the error rate among answered records:

$$
\operatorname{Coverage}(\gamma) = \frac{|\{i:c_i \geq \gamma\}|}{N},
$$

$$
\operatorname{Risk}(\gamma) =
\frac{\sum_{i:c_i \geq \gamma}\mathbf{1}[\hat{y}_i \neq y_i]}
{|\{i:c_i \geq \gamma\}|}.
$$

We report thresholds `0.50`, `0.60`, `0.70`, `0.80`, and `0.90`. We sort all predictions by confidence to form the full risk-coverage curve. We report area under this curve as AURC. We also report mean confidence on errors, which directly measures overconfidence severity.

### Model-directed abstention

We evaluate the emitted `abstain` field separately from threshold sweeps. We report model-directed coverage, selective accuracy, and risk. This distinction lets us compare an explicit learned policy with a post-hoc threshold selected on validation data.

### Format validity

We report valid JSON rate and full schema-validity rate. The parser requires an allowed answer label, a concise nonempty justification, a boolean abstention value, and the confidence field required by the active representation.

### Option-order robustness

Every perturbed record stores its parent ID and new-to-old choice order. We join base and perturbed predictions by parent ID. We compare selected choice text across permutations because answer labels move when options move. We report perturbed accuracy, accuracy delta, choice-text consistency, and choice-flip rate.

## Repository layout

```text
calibrate_qwen/
|-- configs/
|   |-- data_config.yaml
|   |-- issue_data_config.yaml
|   `-- experiment_config.yaml
|-- data/
|   |-- build_dataset.py
|   |-- curate_dataset.py
|   |-- perturb_options.py
|   |-- generate_teacher_targets.py
|   |-- finalize_teacher_targets.py
|   `-- publish_to_huggingface.py
|-- training/
|   |-- common.py
|   |-- prepare_training_data.py
|   |-- train_sft.py
|   |-- train_off_policy.py
|   `-- train_on_policy.py
|-- evaluation/
|   |-- parsing.py
|   |-- sample_model.py
|   |-- calibrate.py
|   |-- metrics.py
|   |-- robustness.py
|   |-- evaluate_predictions.py
|   |-- plot_results.py
|   `-- compare_runs.py
|-- notebooks/
|   |-- 00_setup_and_data.ipynb
|   |-- 01_train_sft.ipynb
|   |-- 02_train_off_policy.ipynb
|   |-- 03_train_on_policy.ipynb
|   `-- 04_evaluate_and_report.ipynb
|-- tests/
|-- DATASET_CARD.md
|-- README_COLAB.md
|-- README.md
`-- requirements.txt
```

## Colab execution

We store secrets in the Colab Secrets panel with these names:

```text
TINKER_API_KEY
HF_TOKEN
WANDB_API_KEY
```

We load them inside each notebook with:

```python
import os
from google.colab import userdata

os.environ["TINKER_API_KEY"] = userdata.get("TINKER_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

The notebooks contain zero literal credentials. We keep each paid run in a dedicated execution cell so we can inspect parameters before submission.

We run the notebooks in this order:

1. [`00_setup_and_data.ipynb`](https://colab.research.google.com/github/ritwikraha/AutoRegressive-Bhasha/blob/main/calibrate_qwen/notebooks/00_setup_and_data.ipynb) validates Hub access and prepares conversation files.
2. [`01_train_sft.ipynb`](https://colab.research.google.com/github/ritwikraha/AutoRegressive-Bhasha/blob/main/calibrate_qwen/notebooks/01_train_sft.ipynb) runs hard-label or teacher-completion SFT.
3. [`02_train_off_policy.ipynb`](https://colab.research.google.com/github/ritwikraha/AutoRegressive-Bhasha/blob/main/calibrate_qwen/notebooks/02_train_off_policy.ipynb) runs fixed-sequence soft-target distillation.
4. [`03_train_on_policy.ipynb`](https://colab.research.google.com/github/ritwikraha/AutoRegressive-Bhasha/blob/main/calibrate_qwen/notebooks/03_train_on_policy.ipynb) runs pure or SFT-initialized on-policy distillation.
5. [`04_evaluate_and_report.ipynb`](https://colab.research.google.com/github/ritwikraha/AutoRegressive-Bhasha/blob/main/calibrate_qwen/notebooks/04_evaluate_and_report.ipynb) samples checkpoints, computes metrics, and renders figures.

Each training notebook exposes `MAX_STEPS` or an equivalent field. We set it to three for a paid smoke test. We set it to `None` for the complete configured run after the smoke test produces a checkpoint and finite loss values.

## Python execution

We can call every notebook operation through imports. The following example prepares teacher conversations and launches SFT:

```python
from training.prepare_training_data import prepare_training_file
from training.train_sft import SFTConfig, train_sft

prepare_training_file(
    output_path="artifacts/training/teacher_numeric.jsonl",
    variant="teacher",
    confidence_format="numeric",
    abstention_threshold=0.60,
)

config = SFTConfig(
    conversation_file="artifacts/training/teacher_numeric.jsonl",
    log_path="artifacts/runs/teacher_sft",
    model_name="Qwen/Qwen3.5-4B",
    max_steps=3,
)

await train_sft(config)
```

We sample a base model or checkpoint through the same interface:

```python
from evaluation.sample_model import SamplingConfig, sample_dataset

await sample_dataset(
    SamplingConfig(
        output_path="results/predictions/student_test.jsonl",
        split="test",
        model_name="Qwen/Qwen3.5-4B",
        checkpoint_path="tinker://YOUR_CHECKPOINT_PATH",
        score_options=True,
    )
)
```

We compute and plot metrics with:

```python
from evaluation.evaluate_predictions import evaluate_file
from evaluation.plot_results import plot_metrics

metrics = evaluate_file(
    predictions_path="results/predictions/student_test.jsonl",
    output_path="results/metrics/student_test.json",
    confidence_source="verbal",
)
plot_metrics(
    "results/metrics/student_test.json",
    "results/figures",
    "student_test",
)
```

We fit temperature and an abstention threshold on validation predictions, then apply the fitted calibration object to each test split. The calibration artifact stores the temperature, validation NLL, selected threshold, achieved validation coverage, selective accuracy, and risk. We reuse this frozen object for in-domain, OOD, and robustness evaluation.

We publish a final sampler checkpoint as a private Hugging Face PEFT adapter with the Tinker CLI:

```bash
python -m tinker.cli checkpoint push-hf \
  tinker://RUN_ID/sampler_weights/final \
  --repo ritwikraha/calibrate-qwen-student
```

## Experiment matrix

We use this first complete matrix:

| Run | Method | Training examples | Primary comparison |
|---|---|---:|---|
| A | Base Qwen3.5-4B | 0 | Pretraining baseline |
| B | Hard-label SFT | 2,000 then 9,134 | Accuracy and format baseline |
| C | Teacher-completion SFT | 2,000 then 9,134 | Fixed completion transfer |
| D | Off-policy soft targets | 2,000 then 9,134 | Token-distribution transfer |
| E | On-policy KL | 4,000 | Student-distribution supervision |
| F | Teacher SFT followed by on-policy KL | 2,000 plus 4,000 | Combined recipe |

We run numeric, bucket, and implicit confidence ablations on the strongest fixed-data method and strongest on-policy method. We select abstention thresholds on validation data. We preserve the final test splits for one evaluation pass per finalized configuration.

## Reproducibility and artifact retention

We keep source code, YAML configuration, notebooks, and documentation in Git. We keep derived datasets in Hugging Face. We keep Tinker checkpoint paths and local training logs under a run-specific directory. We keep sampled predictions, metric JSON, CSV summaries, and figures under `results/`.

The Tinker supervised loop writes `checkpoints.jsonl` into each run directory. The final entry contains the server-side state path and sampler weights. We pass the state path into later training stages through `load_checkpoint_path`. We pass a sampler-compatible checkpoint path into evaluation through `checkpoint_path`.

We pin `tinker==0.24.0` and `tinker-cookbook==0.5.3` in this revision. We verified the recipes against those installed APIs. We use `Qwen/Qwen3.5-4B` as the student and `Qwen/Qwen3.5-9B` as the teacher. We use `qwen3_5_disable_thinking` for short structured responses.

## Cost controls

We control paid compute through staged execution. We start with three training steps and 25 evaluation records. We inspect schema validity, loss values, checkpoint creation, and request completion. We then run the 2,000-record Phase 1 experiment. We advance to the full training mixture after the Phase 1 comparison produces stable metrics.

We cap generated responses at 192 tokens. We use concise one-sentence justifications. We use two on-policy trajectories per prompt. We use a 4B student and a 9B teacher. We resume sampling by stable record ID and training through Tinker checkpoint logs. These controls make failures observable early and preserve completed paid work.

## Interpretation plan

We treat accuracy and calibration as separate axes. A method can increase accuracy while increasing confidence on residual errors. A method can improve AURC while leaving top-line accuracy nearly unchanged. We therefore prioritize a vector of outcomes:

```text
accuracy
expected calibration error
multiclass Brier score
multiclass negative log-likelihood
area under the risk-coverage curve
mean confidence on errors
80 percent coverage accuracy
format validity
OOD accuracy and ECE
option-permutation consistency
```

We regard an on-policy result as high signal when it improves error ranking, AURC, and error confidence across in-domain and OOD data. We also inspect absolute accuracy and format validity to ensure that calibration gains correspond to a usable model.

## Current boundaries

The present release studies English multiple-choice tasks with two to six options. The teacher confidence target uses five-sample agreement, which quantizes confidence in increments of `0.2`. The Phase 1 teacher set contains 2,000 records. The full 9,134-record mixture currently carries gold labels, while teacher metadata covers the Phase 1 subset. We can extend teacher sampling after the Phase 1 experiment establishes the strongest recipe.

Our option probability calculation scores candidate answer labels under a dedicated answer-only prompt. This produces a normalized ranking across available labels. It represents a conditional answer distribution under that scoring prompt. We report its protocol explicitly alongside verbal confidence.

## Deliverables

We designed the repository to produce these deliverables:

- a reproducible curated MCQ dataset;
- repeated teacher samples and agreement confidence;
- LoRA checkpoints for each student recipe;
- raw structured predictions for every evaluation split;
- in-domain, OOD, and robustness metric JSON;
- reliability and risk-coverage figures;
- a machine-readable experiment matrix;
- a technical report with data, methods, results, costs, and limitations.

We have completed the data release, teacher target generation, training code, evaluation code, and Colab orchestration. The next experimental stage runs the paid model matrix and fills the result tables with measured checkpoint outcomes.
