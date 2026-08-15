from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOK_DIR / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


COMMON_BOOTSTRAP = r"""
from pathlib import Path
import os, sys, json, subprocess, textwrap

def find_repo_root():
    try:
        import google.colab  # type: ignore  # noqa: F401
        from google.colab import drive  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except Exception:
        pass

    candidates = [
        Path.cwd(),
        Path("/content/empty-negations"),
        Path("/content/drive/MyDrive/ocn_empty_negations"),
        Path("/content/drive/MyDrive/AutoRegressive-Bhasha/empty-negations"),
    ]
    for candidate in candidates:
        if (candidate / "src/ocn").exists():
            return candidate
    repo_url = os.environ.get("OCN_REPO_URL", "")
    if repo_url:
        target = Path("/content/empty-negations")
        if not target.exists():
            subprocess.run(["git", "clone", repo_url, str(target)], check=True)
        return target
    raise FileNotFoundError(
        "Could not find the empty-negations repo. Run this notebook from the repo, "
        "copy it to /content/drive/MyDrive/ocn_empty_negations, or set OCN_REPO_URL."
    )

REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))
print("Repo:", REPO_ROOT)
"""


def setup_notebook() -> list[dict]:
    return [
        md(
            """
            # 00 - Colab Setup And Research Configuration

            Run this first. It installs dependencies, mounts Google Drive, reads your Colab secrets, logs in to Hugging Face and W&B, and writes a shared run configuration for the later notebooks.

            Required Colab secrets:

            - `HF_WRITE_ACCESS`
            - `WANDB_KEY`

            Recommended runtime: Colab Pro GPU. A100 is ideal, L4 works for the default Qwen 0.5B/1.5B runs.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import sys, subprocess

            requirements = REPO_ROOT / "requirements-colab.txt"
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
            print("Installed:", requirements)
            """
        ),
        code(
            r"""
            from huggingface_hub import HfApi
            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, utc_timestamp

            HF_TOKEN = login_huggingface("HF_WRITE_ACCESS")
            api = HfApi(token=HF_TOKEN)
            HF_OWNER = api.whoami()["name"]

            RUN_ID = utc_timestamp()
            PROJECT_NAME = "ocn_empty_negations"
            HF_DATASET_PREFIX = "ocn-empty-negations"

            paths = make_colab_paths(PROJECT_NAME)
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"setup-{RUN_ID}",
                config={"hf_owner": HF_OWNER, "run_id": RUN_ID},
            )

            CONFIG = {
                "run_id": RUN_ID,
                "hf_owner": HF_OWNER,
                "hf_private": False,
                "hf_prompt_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-prompts",
                "hf_generation_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-generations",
                "hf_detection_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-detection",
                "hf_reward_pairs_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-reward-pairs",
                "hf_reward_scores_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-reward-scores",
                "drive_project_root": str(paths.project_root),
                "drive_data_root": str(paths.data_root),
                "drive_figure_root": str(paths.figure_root),
                "default_prompt_limit": 96,
                "default_seeds": [1, 2],
            }

            config_path = paths.project_root / "ocn_colab_config.json"
            config_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
            print(json.dumps(CONFIG, indent=2))

            run.finish()
            """
        ),
    ]


def prompts_notebook() -> list[dict]:
    return [
        md(
            """
            # 01 - Create And Publish Prompt Dataset

            This notebook builds the controlled prompt bank for the OCN study, saves it to Google Drive, logs summary plots to W&B, and publishes it to Hugging Face.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import matplotlib.pyplot as plt
            import seaborn as sns
            import wandb

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe
            from ocn.prompt_factory import build_prompt_dataset

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"prompts-{config['run_id']}", config=config)
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            prompts = build_prompt_dataset()
            prompts.head()
            """
        ),
        code(
            r"""
            local_csv = save_dataframe(prompts, Path(config["drive_data_root"]) / "ocn_prompts.csv")
            local_parquet = save_dataframe(prompts, Path(config["drive_data_root"]) / "ocn_prompts.parquet")

            repo_url = publish_dataframe_to_hf(
                prompts,
                repo_id=config["hf_prompt_repo"],
                split="train",
                private=config["hf_private"],
                card_path=REPO_ROOT / "dataset_cards/ocn_prompts.md",
                commit_message=f"Publish OCN prompts {config['run_id']}",
            )
            print("Saved:", local_csv)
            print("Published:", repo_url)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            prompts["category"].value_counts().sort_values().plot(kind="barh", ax=axes[0], color="#4c78a8")
            axes[0].set_title("Prompts by category")
            axes[0].set_xlabel("count")
            prompts["variant"].value_counts().sort_values().plot(kind="barh", ax=axes[1], color="#f58518")
            axes[1].set_title("Prompts by variant")
            axes[1].set_xlabel("count")
            plt.tight_layout()

            figure_path = Path(config["drive_figure_root"]) / "01_prompt_dataset_counts.png"
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")
            wandb.log({
                "prompt_count": len(prompts),
                "category_count": prompts["category"].nunique(),
                "variant_count": prompts["variant"].nunique(),
                "prompt_dataset_counts": wandb.Image(str(figure_path)),
                "prompt_table": wandb.Table(dataframe=prompts.head(200)),
            })
            run.finish()
            figure_path
            """
        ),
    ]


def generation_notebook() -> list[dict]:
    return [
        md(
            """
            # 02 - Generate OSS Model Responses

            This notebook loads the prompt dataset, runs open-source causal language models, autosaves partial generations to Google Drive, logs progress to W&B, and publishes the combined generations dataset to Hugging Face.

            Default models are Qwen 0.5B/1.5B base and instruct checkpoints. Gemma 2B models are included as optional entries because they may require accepting the model license on Hugging Face.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import gc, json
            from pathlib import Path
            import pandas as pd
            import torch
            import wandb
            from datasets import load_dataset

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe
            from ocn.generation import (
                DecodingSpec,
                ModelSpec,
                generation_rows,
                load_text_generation_model,
            )

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"generate-{config['run_id']}", config=config)
            """
        ),
        code(
            r"""
            prompts = load_dataset(config["hf_prompt_repo"], split="train").to_pandas()

            RUN_MODE = "pilot"  # change to "main" for the full prompt bank
            PROMPT_LIMIT = config["default_prompt_limit"] if RUN_MODE == "pilot" else None
            if PROMPT_LIMIT:
                prompts = prompts.sample(n=min(PROMPT_LIMIT, len(prompts)), random_state=13).sort_values("prompt_id")

            MODEL_SPECS = [
                ModelSpec("Qwen/Qwen2.5-0.5B", "qwen", "base", False),
                ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "qwen", "instruct", True),
                ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "qwen", "instruct", True),
                # Uncomment after accepting Gemma terms on Hugging Face:
                # ModelSpec("google/gemma-2-2b", "gemma", "base", False),
                # ModelSpec("google/gemma-2-2b-it", "gemma", "instruct", True),
            ]

            DECODINGS = [
                DecodingSpec("greedy", temperature=0.0, top_p=1.0, max_new_tokens=160),
                DecodingSpec("normal_temp", temperature=0.7, top_p=0.95, max_new_tokens=180),
            ]

            SEEDS = config["default_seeds"]
            QUANTIZE_4BIT = True
            print("Prompts:", len(prompts), "Models:", len(MODEL_SPECS), "Decodings:", len(DECODINGS), "Seeds:", SEEDS)
            """
        ),
        code(
            r"""
            all_parts = []
            generation_dir = Path(config["drive_data_root"]) / "generations_parts"
            generation_dir.mkdir(parents=True, exist_ok=True)

            for model_spec in MODEL_SPECS:
                print(f"\nLoading {model_spec.model_id}")
                tokenizer, model = load_text_generation_model(model_spec.model_id, quantize_4bit=QUANTIZE_4BIT)
                for decoding in DECODINGS:
                    print(f"Generating: {model_spec.model_id} / {decoding.name}")
                    rows = generation_rows(
                        prompts=prompts,
                        model_spec=model_spec,
                        decoding=decoding,
                        tokenizer=tokenizer,
                        model=model,
                        seeds=SEEDS,
                    )
                    part = pd.DataFrame(rows)
                    part_path = generation_dir / f"{model_spec.family}_{model_spec.stage}_{decoding.name}.csv"
                    save_dataframe(part, part_path)
                    all_parts.append(part)

                    combined = pd.concat(all_parts, ignore_index=True)
                    combined_path = save_dataframe(combined, Path(config["drive_data_root"]) / "ocn_generations.csv")
                    repo_url = publish_dataframe_to_hf(
                        combined,
                        repo_id=config["hf_generation_repo"],
                        split="train",
                        private=config["hf_private"],
                        card_path=REPO_ROOT / "dataset_cards/ocn_generations.md",
                        commit_message=f"Update OCN generations {config['run_id']}",
                    )
                    wandb.log({
                        "generated_rows": len(combined),
                        "last_part_rows": len(part),
                        "model_id": model_spec.model_id,
                        "decoding": decoding.name,
                    })
                    print("Autosaved:", combined_path)
                    print("Autopublished:", repo_url)

                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            run.finish()
            """
        ),
    ]


def detection_notebook() -> list[dict]:
    return [
        md(
            """
            # 03 - Detect OCN And Publish Candidate Dataset

            This notebook loads generations, applies the lexical OCN detector, saves scored rows to Drive, publishes them to Hugging Face, and logs charts to W&B.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            import wandb
            from datasets import load_dataset

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe
            from ocn.detectors import OCNDetector
            from ocn.metrics import detection_summary, grouped_ocn_rates, top_patterns

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"detect-{config['run_id']}", config=config)
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            generations = load_dataset(config["hf_generation_repo"], split="train").to_pandas()
            scored = OCNDetector().annotate_rows(generations, text_column="response")
            summary = detection_summary(scored)
            summary
            """
        ),
        code(
            r"""
            scored_path = save_dataframe(scored, Path(config["drive_data_root"]) / "ocn_detection.csv")
            repo_url = publish_dataframe_to_hf(
                scored,
                repo_id=config["hf_detection_repo"],
                split="train",
                private=config["hf_private"],
                card_path=REPO_ROOT / "dataset_cards/ocn_detection.md",
                commit_message=f"Publish OCN detection {config['run_id']}",
            )
            print("Saved:", scored_path)
            print("Published:", repo_url)
            """
        ),
        code(
            r"""
            model_rates = grouped_ocn_rates(scored, ["model_id", "model_stage", "decoding"])
            category_rates = grouped_ocn_rates(scored, ["category", "variant"])
            pattern_counts = top_patterns(scored, 20)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.barplot(data=model_rates, y="model_id", x="ocn_rate", hue="decoding", ax=axes[0])
            axes[0].set_title("Candidate OCN rate by model")
            axes[0].set_xlim(0, 1)
            sns.barplot(data=category_rates.head(20), y="category", x="ocn_rate", hue="variant", ax=axes[1])
            axes[1].set_title("Top category/variant OCN rates")
            axes[1].set_xlim(0, 1)
            plt.tight_layout()
            fig_path = Path(config["drive_figure_root"]) / "03_ocn_rates.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")

            wandb.log({
                **summary.to_dict(),
                "model_rates": wandb.Table(dataframe=model_rates),
                "category_rates": wandb.Table(dataframe=category_rates),
                "top_patterns": wandb.Table(dataframe=pattern_counts),
                "ocn_rate_chart": wandb.Image(str(fig_path)),
            })
            run.finish()
            fig_path
            """
        ),
    ]


def reward_pairs_notebook() -> list[dict]:
    return [
        md(
            """
            # 04 - Create Reward Preference Pair Dataset

            This notebook creates matched plain/OCN response variants, verifies detector separation, saves to Drive, and publishes the pair dataset to Hugging Face.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import wandb

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe
            from ocn.detectors import OCNDetector
            from ocn.reward_pairs import PropositionSet, starter_proposition_sets, variants_to_frame

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"reward-pairs-{config['run_id']}", config=config)
            """
        ),
        code(
            r"""
            extra_items = [
                PropositionSet("r004", "remote work policy", "The policy", "improves hiring flexibility", "changes coordination costs across teams"),
                PropositionSet("r005", "API design", "The design", "makes endpoints easier to use", "reduces integration errors for developers"),
                PropositionSet("r006", "data privacy", "The program", "protects customer information", "strengthens trust with users and regulators"),
                PropositionSet("r007", "process improvement", "The change", "reduces manual review time", "gives teams clearer visibility into bottlenecks"),
                PropositionSet("r008", "CRISPR", "The technique", "edits targeted genetic sequences", "creates new possibilities for biological research"),
                PropositionSet("r009", "budgeting app", "The app", "tracks spending", "helps users notice patterns before they become problems"),
                PropositionSet("r010", "printing press", "The invention", "increased copying speed", "changed how knowledge circulated across institutions"),
                PropositionSet("r011", "leadership", "Leadership", "coordinates group action", "helps people make decisions under uncertainty"),
                PropositionSet("r012", "sincere apology", "A sincere apology", "acknowledges harm", "creates conditions for repair"),
            ]

            pairs = variants_to_frame(starter_proposition_sets() + extra_items, shuffle=True, seed=42)
            scored_pairs = OCNDetector().annotate_rows(pairs, text_column="response")
            scored_pairs.groupby("variant_type")[["has_ocn", "ocn_count"]].mean()
            """
        ),
        code(
            r"""
            pair_path = save_dataframe(scored_pairs, Path(config["drive_data_root"]) / "ocn_reward_pairs.csv")
            repo_url = publish_dataframe_to_hf(
                scored_pairs,
                repo_id=config["hf_reward_pairs_repo"],
                split="train",
                private=config["hf_private"],
                card_path=REPO_ROOT / "dataset_cards/ocn_reward_pairs.md",
                commit_message=f"Publish OCN reward pairs {config['run_id']}",
            )
            wandb.log({
                "reward_pair_rows": len(scored_pairs),
                "reward_pairs": wandb.Table(dataframe=scored_pairs),
            })
            run.finish()
            print("Saved:", pair_path)
            print("Published:", repo_url)
            """
        ),
    ]


def analysis_notebook() -> list[dict]:
    return [
        md(
            """
            # 05 - Analysis And Reporting

            This notebook loads the detection dataset, computes model/category/persona effects, saves report tables and plots to Google Drive, and logs them to W&B.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            import statsmodels.formula.api as smf
            import wandb
            from datasets import load_dataset

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, save_dataframe
            from ocn.metrics import detection_summary, grouped_ocn_rates, top_patterns

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"analysis-{config['run_id']}", config=config)
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            df = load_dataset(config["hf_detection_repo"], split="train").to_pandas()
            summary = detection_summary(df)
            model_rates = grouped_ocn_rates(df, ["model_id", "model_stage", "decoding"])
            prompt_rates = grouped_ocn_rates(df, ["category", "variant", "persona"])
            save_dataframe(model_rates, Path(config["drive_data_root"]) / "report_model_rates.csv")
            save_dataframe(prompt_rates, Path(config["drive_data_root"]) / "report_prompt_rates.csv")
            summary
            """
        ),
        code(
            r"""
            regression_df = df.copy()
            regression_df["has_ocn_int"] = regression_df["has_ocn"].astype(int)
            formula = "has_ocn_int ~ C(model_stage) + C(model_family) + C(decoding) + C(variant) + C(persona) + length_target"
            model = smf.logit(formula, data=regression_df).fit(disp=False)
            report_path = Path(config["drive_data_root"]) / "logit_model_summary.txt"
            report_path.write_text(model.summary().as_text(), encoding="utf-8")
            print(model.summary())
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            sns.barplot(data=model_rates, y="model_id", x="ocn_rate", hue="decoding", ax=axes[0, 0])
            axes[0, 0].set_title("OCN rate by model and decoding")
            axes[0, 0].set_xlim(0, 1)

            variant_rates = grouped_ocn_rates(df, ["variant"])
            sns.barplot(data=variant_rates, y="variant", x="ocn_rate", ax=axes[0, 1], color="#f58518")
            axes[0, 1].set_title("OCN rate by prompt variant")
            axes[0, 1].set_xlim(0, 1)

            persona_rates = grouped_ocn_rates(df, ["persona"])
            sns.barplot(data=persona_rates, y="persona", x="ocn_rate", ax=axes[1, 0], color="#54a24b")
            axes[1, 0].set_title("OCN rate by persona")
            axes[1, 0].set_xlim(0, 1)

            patterns = top_patterns(df, 12)
            sns.barplot(data=patterns, y="pattern", x="count", ax=axes[1, 1], color="#b279a2")
            axes[1, 1].set_title("Top detector patterns")
            plt.tight_layout()

            fig_path = Path(config["drive_figure_root"]) / "05_analysis_dashboard.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            wandb.log({
                **summary.to_dict(),
                "analysis_dashboard": wandb.Image(str(fig_path)),
                "model_rates": wandb.Table(dataframe=model_rates),
                "prompt_rates": wandb.Table(dataframe=prompt_rates),
                "logit_summary": model.summary().as_text(),
            })
            run.finish()
            fig_path
            """
        ),
    ]


def reward_scoring_notebook() -> list[dict]:
    return [
        md(
            """
            # 06 - Optional Reward Model Scoring

            This notebook scores matched reward-pair responses with an open reward model, saves scores to Drive, logs preference deltas to W&B, and publishes the score dataset to Hugging Face.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import pandas as pd
            import torch
            import wandb
            from datasets import load_dataset
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"reward-score-{config['run_id']}", config=config)
            """
        ),
        code(
            r"""
            pairs = load_dataset(config["hf_reward_pairs_repo"], split="train").to_pandas()
            REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"

            tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID, device_map="auto")
            model.eval()

            def score_text(text: str) -> float:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    out = model(**inputs)
                return float(out.logits.squeeze().detach().cpu())

            pairs["reward_model_id"] = REWARD_MODEL_ID
            pairs["reward_score"] = [score_text(text) for text in pairs["response"]]
            pairs.head()
            """
        ),
        code(
            r"""
            scores_path = save_dataframe(pairs, Path(config["drive_data_root"]) / "ocn_reward_scores.csv")
            repo_url = publish_dataframe_to_hf(
                pairs,
                repo_id=config["hf_reward_scores_repo"],
                split="train",
                private=config["hf_private"],
                commit_message=f"Publish OCN reward scores {config['run_id']}",
            )

            deltas = (
                pairs.pivot_table(index="question_id", columns="variant_type", values="reward_score", aggfunc="mean")
                .assign(
                    justified_minus_plain=lambda x: x.get("justified_ocn") - x.get("plain"),
                    empty_minus_plain=lambda x: x.get("empty_ocn") - x.get("plain"),
                )
                .reset_index()
            )
            wandb.log({
                "reward_scores": wandb.Table(dataframe=pairs),
                "reward_deltas": wandb.Table(dataframe=deltas),
                "mean_justified_minus_plain": float(deltas["justified_minus_plain"].mean()),
                "mean_empty_minus_plain": float(deltas["empty_minus_plain"].mean()),
            })
            run.finish()
            print("Saved:", scores_path)
            print("Published:", repo_url)
            deltas.describe()
            """
        ),
    ]


def lora_notebook() -> list[dict]:
    return [
        md(
            """
            # 07 - Optional LoRA Style Intervention

            This is a compact controlled-training notebook. It builds tiny plain-vs-OCN SFT datasets from the reward-pair table, trains a LoRA adapter on Qwen 0.5B, saves outputs to Drive, and logs training to W&B.

            Treat this as a pilot intervention, not a paper-scale DPO experiment.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import pandas as pd
            import torch
            import wandb
            from datasets import Dataset, load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
            from trl import SFTTrainer

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            login_huggingface("HF_WRITE_ACCESS")
            run = login_wandb(project="ocn-empty-negations", name=f"lora-sft-{config['run_id']}", config=config)
            """
        ),
        code(
            r"""
            pairs = load_dataset(config["hf_reward_pairs_repo"], split="train").to_pandas()

            CONDITION = "ocn"  # use "plain" for the counter-condition
            if CONDITION == "plain":
                train_df = pairs[pairs["variant_type"].eq("plain")].copy()
            else:
                train_df = pairs[pairs["variant_type"].isin(["justified_ocn", "empty_ocn"])].copy()

            train_df["text"] = "Prompt: Explain the topic clearly.\nAnswer: " + train_df["response"]
            train_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
            train_ds
            """
        ),
        code(
            r"""
            BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            )

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            output_dir = Path(config["drive_project_root"]) / f"adapters/qwen05_{CONDITION}_{config['run_id']}"
            args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=3,
                logging_steps=1,
                save_strategy="epoch",
                report_to=["wandb"],
                bf16=True,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                dataset_text_field="text",
                peft_config=peft_config,
                args=args,
                max_seq_length=512,
            )
            trainer.train()
            trainer.save_model(str(output_dir))
            run.finish()
            output_dir
            """
        ),
    ]


def main() -> None:
    write_notebook("00_colab_setup_and_config.ipynb", setup_notebook())
    write_notebook("01_create_prompt_dataset.ipynb", prompts_notebook())
    write_notebook("02_generate_oss_model_responses.ipynb", generation_notebook())
    write_notebook("03_detect_and_publish_ocn_dataset.ipynb", detection_notebook())
    write_notebook("04_create_reward_pair_dataset.ipynb", reward_pairs_notebook())
    write_notebook("05_analysis_and_reporting.ipynb", analysis_notebook())
    write_notebook("06_optional_reward_model_scoring.ipynb", reward_scoring_notebook())
    write_notebook("07_optional_lora_style_intervention.ipynb", lora_notebook())


if __name__ == "__main__":
    main()
