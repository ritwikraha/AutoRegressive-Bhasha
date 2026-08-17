from __future__ import annotations

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def write_notebook(name: str, cells: list[dict], gpu_type: str = "A100") -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": gpu_type, "provenance": []},
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

DEFAULT_REPO_URL = "https://github.com/ritwikraha/AutoRegressive-Bhasha.git"

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
        Path("/content/AutoRegressive-Bhasha/empty-negations"),
        Path("/content/drive/MyDrive/ocn_empty_negations"),
        Path("/content/drive/MyDrive/AutoRegressive-Bhasha/empty-negations"),
    ]
    for candidate in candidates:
        if (candidate / "src/ocn").exists():
            return candidate
    repo_url = os.environ.get("OCN_REPO_URL", DEFAULT_REPO_URL)
    target = Path("/content/AutoRegressive-Bhasha")
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)

    cloned_candidates = [target / "empty-negations", target]
    for candidate in cloned_candidates:
        if (candidate / "src/ocn").exists():
            return candidate

    raise FileNotFoundError(
        f"Cloned {repo_url}, but could not find src/ocn. "
        "Set OCN_REPO_URL to a repository containing empty-negations/src/ocn."
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

            Notebook `00` itself can run on CPU. Before the main generation run in notebook `02`, start a fresh A100 runtime and rerun this setup so the pinned model dependencies are installed before Transformers is imported.
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
                "hf_main_generation_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-generations-main-gemma4-qwen35",
                "hf_main_detection_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-detection-main-gemma4-qwen35",
                "hf_main_reward_pairs_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-reward-pairs-main-gemma4-qwen35",
                "hf_main_reward_scores_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-reward-scores-main-gemma4-qwen35",
                "hf_main_semantic_repo": f"{HF_OWNER}/{HF_DATASET_PREFIX}-semantic-main-gemma4-qwen35",
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
            _ = login_huggingface("HF_WRITE_ACCESS")
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

            This is the main OCN generation experiment. It runs matched base/post-trained pairs from Gemma 4 E2B and Qwen 3.5 2B over the full prompt bank, autosaves resumable chunks to Google Drive, logs progress to W&B, and publishes to a dedicated Hugging Face dataset.

            Required runtime: an A100 GPU. The notebook stops before generation if Colab assigned a different accelerator. Both Gemma 4 checkpoints are public Apache 2.0 models and do not require a separate Hugging Face access request.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import gc, json
            from importlib.metadata import version
            from pathlib import Path
            from packaging.version import Version
            import pandas as pd
            import torch
            import wandb
            from datasets import load_dataset

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe, utc_timestamp
            from ocn.generation import (
                DecodingSpec,
                ModelSpec,
                generation_rows,
                load_text_generation_model,
            )
            from ocn.prompt_factory import slugify

            if Version(version("transformers")) < Version("5.14.1"):
                raise RuntimeError(
                    "Gemma 4 requires transformers>=5.14.1. Start a fresh Colab runtime, "
                    "run the updated notebook 00, then return to notebook 02."
                )

            if not torch.cuda.is_available():
                raise RuntimeError("No GPU detected. Select Runtime > Change runtime type > A100 GPU.")
            GPU_NAME = torch.cuda.get_device_name(0)
            if "A100" not in GPU_NAME.upper():
                raise RuntimeError(
                    f"This main run requires an A100, but Colab assigned {GPU_NAME}. "
                    "Reconnect with Runtime > Change runtime type > A100 GPU."
                )
            print("Verified accelerator:", GPU_NAME)

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            """
        ),
        code(
            r"""
            prompts = load_dataset(config["hf_prompt_repo"], split="train").to_pandas()

            EXPERIMENT_ID = "main_gemma4_qwen35"
            GENERATION_RUN_ID = utc_timestamp()
            MAIN_GENERATION_REPO = config.get(
                "hf_main_generation_repo",
                f"{config['hf_owner']}/ocn-empty-negations-generations-main-gemma4-qwen35",
            )

            MODEL_SPECS = [
                ModelSpec("google/gemma-4-E2B", "gemma4", "base", False, "multimodal_lm"),
                ModelSpec("google/gemma-4-E2B-it", "gemma4", "instruct", True, "multimodal_lm"),
                ModelSpec("Qwen/Qwen3.5-2B-Base", "qwen3.5", "base", False, "multimodal_lm"),
                ModelSpec("Qwen/Qwen3.5-2B", "qwen3.5", "instruct", True, "multimodal_lm"),
            ]

            DECODINGS = [
                DecodingSpec("greedy", temperature=0.0, top_p=1.0, max_new_tokens=160),
                DecodingSpec("normal_temp", temperature=0.7, top_p=0.95, max_new_tokens=180),
            ]

            SEEDS = config["default_seeds"]
            PROMPT_BATCH_SIZE = 24
            # All four models fit sequentially in BF16 on an A100. Do not enable
            # quantization for the main study because it changes the model itself.
            QUANTIZE_4BIT = False
            EXPECTED_ROWS = len(prompts) * len(MODEL_SPECS) * len(DECODINGS) * len(SEEDS)

            experiment_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "generation_run_id": GENERATION_RUN_ID,
                "run_mode": "main",
                "gpu_name": GPU_NAME,
                "precision": "bfloat16",
                "quantize_4bit": QUANTIZE_4BIT,
                "main_generation_repo": MAIN_GENERATION_REPO,
                "models": [spec.model_id for spec in MODEL_SPECS],
                "prompt_count": len(prompts),
                "expected_rows": EXPECTED_ROWS,
                "prompt_batch_size": PROMPT_BATCH_SIZE,
                "inference_batching": "prompt_batch_up_to_24",
                "greedy_seed_reuse": True,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"generate-{EXPERIMENT_ID}-{GENERATION_RUN_ID}",
                config=experiment_config,
            )
            print("Prompts:", len(prompts), "Models:", len(MODEL_SPECS), "Decodings:", len(DECODINGS), "Seeds:", SEEDS)
            print("Expected rows:", EXPECTED_ROWS)
            print("Publishing to:", MAIN_GENERATION_REPO)
            """
        ),
        code(
            r"""
            generation_dir = Path(config["drive_data_root"]) / "generation_runs" / EXPERIMENT_ID
            generation_dir.mkdir(parents=True, exist_ok=True)
            expected_part_rows = len(prompts) * len(SEEDS)
            expected_part_keys = {
                (prompt_id, seed)
                for prompt_id in prompts["prompt_id"]
                for seed in SEEDS
            }

            def part_path_for(model_spec, decoding):
                model_slug = slugify(model_spec.model_id)
                return generation_dir / f"{model_slug}_{decoding.name}.csv"

            def load_existing_part(path):
                if not path.exists():
                    return pd.DataFrame()
                part = pd.read_csv(path)
                required = {"prompt_id", "seed", "model_id", "decoding", "response"}
                if not required.issubset(part.columns):
                    print(f"Ignoring incompatible checkpoint: {path.name}")
                    return pd.DataFrame()
                part = part[part["prompt_id"].isin(prompts["prompt_id"])].copy()
                return part.drop_duplicates(["prompt_id", "seed"], keep="last")

            def load_complete_part(path):
                part = load_existing_part(path)
                keys = set(zip(part.get("prompt_id", []), part.get("seed", [])))
                if keys != expected_part_keys:
                    return None
                return part

            def combine_complete_parts():
                frames = []
                for spec in MODEL_SPECS:
                    for decoding_spec in DECODINGS:
                        complete = load_complete_part(part_path_for(spec, decoding_spec))
                        if complete is not None:
                            frames.append(complete)
                if not frames:
                    raise RuntimeError("No completed generation parts were found.")
                return pd.concat(frames, ignore_index=True).sort_values(
                    ["model_id", "decoding", "prompt_id", "seed"]
                ).reset_index(drop=True)

            for model_spec in MODEL_SPECS:
                missing_decodings = [
                    decoding for decoding in DECODINGS
                    if load_complete_part(part_path_for(model_spec, decoding)) is None
                ]
                if not missing_decodings:
                    print(f"Skipping completed model: {model_spec.model_id}")
                    continue

                print(f"\nLoading {model_spec.model_id}")
                try:
                    tokenizer, model = load_text_generation_model(
                        model_spec.model_id,
                        quantize_4bit=QUANTIZE_4BIT,
                        loader_type=model_spec.loader_type,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load {model_spec.model_id}. Check the model ID, "
                        "network connection, installed Transformers version, and available memory."
                    ) from exc

                model_revision = getattr(model.config, "_commit_hash", None)
                for decoding in missing_decodings:
                    print(f"Generating: {model_spec.model_id} / {decoding.name}")
                    part_path = part_path_for(model_spec, decoding)
                    part = load_existing_part(part_path)
                    completed_prompt_ids = {
                        prompt_id
                        for prompt_id, group in part.groupby("prompt_id")
                        if set(group["seed"]) == set(SEEDS)
                    } if not part.empty else set()
                    remaining = prompts[~prompts["prompt_id"].isin(completed_prompt_ids)]

                    for start in range(0, len(remaining), PROMPT_BATCH_SIZE):
                        prompt_batch = remaining.iloc[start : start + PROMPT_BATCH_SIZE]
                        rows = generation_rows(
                            prompts=prompt_batch,
                            model_spec=model_spec,
                            decoding=decoding,
                            tokenizer=tokenizer,
                            model=model,
                            seeds=SEEDS,
                        )
                        new_rows = pd.DataFrame(rows)
                        new_rows["experiment_id"] = EXPERIMENT_ID
                        new_rows["generation_run_id"] = GENERATION_RUN_ID
                        new_rows["gpu_name"] = GPU_NAME
                        new_rows["precision"] = "bfloat16"
                        new_rows["model_revision"] = model_revision
                        new_rows["inference_batching"] = "prompt_batch_up_to_24"
                        part = pd.concat([part, new_rows], ignore_index=True).drop_duplicates(
                            ["prompt_id", "seed"], keep="last"
                        )
                        save_dataframe(part, part_path)
                        wandb.log({
                            "checkpoint_rows": len(part),
                            "checkpoint_fraction": len(part) / expected_part_rows,
                            "model_id": model_spec.model_id,
                            "decoding": decoding.name,
                        })
                        print(f"Checkpointed {len(part)}/{expected_part_rows}: {part_path.name}")

                    part = load_complete_part(part_path)
                    if part is None:
                        raise RuntimeError(f"Generation checkpoint is incomplete: {part_path}")

                    combined = combine_complete_parts()
                    combined_path = save_dataframe(
                        combined,
                        Path(config["drive_data_root"]) / "ocn_generations_main_gemma4_qwen35.csv",
                    )
                    repo_url = publish_dataframe_to_hf(
                        combined,
                        repo_id=MAIN_GENERATION_REPO,
                        split="train",
                        private=config["hf_private"],
                        card_path=REPO_ROOT / "dataset_cards/ocn_generations.md",
                        commit_message=f"Update {EXPERIMENT_ID} generations {GENERATION_RUN_ID}",
                    )
                    wandb.log({
                        "generated_rows": len(combined),
                        "expected_rows": EXPECTED_ROWS,
                        "completion_fraction": len(combined) / EXPECTED_ROWS,
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

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe, utc_timestamp
            from ocn.detectors import OCNDetector
            from ocn.metrics import detection_summary, grouped_ocn_rates, top_patterns

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            DETECTION_RUN_ID = utc_timestamp()
            MAIN_GENERATION_REPO = config.get(
                "hf_main_generation_repo",
                f"{config['hf_owner']}/ocn-empty-negations-generations-main-gemma4-qwen35",
            )
            MAIN_DETECTION_REPO = config.get(
                "hf_main_detection_repo",
                f"{config['hf_owner']}/ocn-empty-negations-detection-main-gemma4-qwen35",
            )
            detection_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "detection_run_id": DETECTION_RUN_ID,
                "source_repo": MAIN_GENERATION_REPO,
                "output_repo": MAIN_DETECTION_REPO,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"detect-{EXPERIMENT_ID}-{DETECTION_RUN_ID}",
                config=detection_config,
            )
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            generations = load_dataset(MAIN_GENERATION_REPO, split="train").to_pandas()
            scored = OCNDetector().annotate_rows(generations, text_column="response")
            summary = detection_summary(scored)
            summary
            """
        ),
        code(
            r"""
            scored_path = save_dataframe(
                scored,
                Path(config["drive_data_root"]) / "ocn_detection_main_gemma4_qwen35.csv",
            )
            repo_url = publish_dataframe_to_hf(
                scored,
                repo_id=MAIN_DETECTION_REPO,
                split="train",
                private=config["hf_private"],
                card_path=REPO_ROOT / "dataset_cards/ocn_detection.md",
                commit_message=f"Publish {EXPERIMENT_ID} detection {DETECTION_RUN_ID}",
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
            fig_path = Path(config["drive_figure_root"]) / "03_ocn_rates_main_gemma4_qwen35.png"
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
            # 04 - Create Counterfactual Reward Preference Pairs

            This notebook loads the main detection dataset, rewrites every unique lexical OCN candidate into direct affirmative prose with Qwen 3.5 2B, filters for detector separation and content retention, saves resumable artifacts to Drive, publishes valid blinded pairs to Hugging Face, and logs quality diagnostics to W&B.

            Use an L4 GPU. Set `OCN_REWARD_PAIR_LIMIT` only for a pilot; leave it unset for the complete run.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import gc
            import json
            import os
            from pathlib import Path
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            import torch
            import wandb
            from datasets import load_dataset

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe, utc_timestamp
            from ocn.generation import DecodingSpec, generate_batch, load_text_generation_model
            from ocn.reward_pairs import build_counterfactual_pair_frame, make_plain_rewrite_prompt, make_plain_rewrite_retry_prompt, normalize_plain_rewrite, select_best_plain_rewrites, select_unique_ocn_candidates

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            REWARD_PAIR_RUN_ID = utc_timestamp()
            MAIN_DETECTION_REPO = config.get(
                "hf_main_detection_repo",
                f"{config['hf_owner']}/ocn-empty-negations-detection-main-gemma4-qwen35",
            )
            MAIN_REWARD_PAIR_REPO = config.get(
                "hf_main_reward_pairs_repo",
                f"{config['hf_owner']}/ocn-empty-negations-reward-pairs-main-gemma4-qwen35",
            )
            REWRITE_MODEL_ID = "Qwen/Qwen3.5-2B"
            FALLBACK_REWRITE_MODEL_ID = "google/gemma-4-E2B-it"
            REWRITE_BATCH_SIZE = int(os.environ.get("OCN_REWRITE_BATCH_SIZE", "12"))
            candidate_limit = int(os.environ.get("OCN_REWARD_PAIR_LIMIT", "0"))
            MAX_CANDIDATES = candidate_limit or None
            MAX_QWEN_REWRITE_ATTEMPTS = 3
            MAX_REWRITE_ATTEMPTS = 4
            QUANTIZE_4BIT = True

            if not torch.cuda.is_available():
                raise RuntimeError("Notebook 04 requires a GPU runtime; select an L4 in Colab.")
            GPU_NAME = torch.cuda.get_device_name(0)
            print("GPU:", GPU_NAME)

            reward_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "reward_pair_run_id": REWARD_PAIR_RUN_ID,
                "source_repo": MAIN_DETECTION_REPO,
                "output_repo": MAIN_REWARD_PAIR_REPO,
                "rewrite_model_id": REWRITE_MODEL_ID,
                "fallback_rewrite_model_id": FALLBACK_REWRITE_MODEL_ID,
                "rewrite_batch_size": REWRITE_BATCH_SIZE,
                "max_candidates": MAX_CANDIDATES,
                "max_rewrite_attempts": MAX_REWRITE_ATTEMPTS,
                "quantize_4bit": QUANTIZE_4BIT,
                "gpu_name": GPU_NAME,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"reward-pairs-{EXPERIMENT_ID}-{REWARD_PAIR_RUN_ID}",
                config=reward_config,
            )
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            detections = load_dataset(MAIN_DETECTION_REPO, split="train").to_pandas()
            candidates = select_unique_ocn_candidates(detections, limit=MAX_CANDIDATES)
            print("Detection rows:", len(detections))
            print("Lexical OCN rows:", int(detections["has_ocn"].sum()))
            print("Unique rewrite candidates:", len(candidates))
            display(
                candidates.groupby(["model_id", "decoding"], dropna=False)
                .size()
                .rename("candidates")
                .reset_index()
            )
            """
        ),
        code(
            r"""
            checkpoint_path = Path(config["drive_data_root"]) / "ocn_reward_pair_rewrites_main_gemma4_qwen35.csv"
            if checkpoint_path.exists():
                checkpoint = pd.read_csv(
                    checkpoint_path,
                    dtype={"source_candidate_id": str},
                )
            else:
                checkpoint = pd.DataFrame(
                    columns=[
                        "source_candidate_id",
                        "plain_response",
                        "rewrite_model_id",
                        "rewrite_run_id",
                        "rewrite_attempt",
                        "rewrite_prompt_version",
                    ]
                )
            if "rewrite_attempt" not in checkpoint.columns:
                checkpoint["rewrite_attempt"] = 1
            if "rewrite_prompt_version" not in checkpoint.columns:
                checkpoint["rewrite_prompt_version"] = "v1"
            checkpoint["rewrite_attempt"] = checkpoint["rewrite_attempt"].fillna(1).astype(int)
            checkpoint = checkpoint.drop_duplicates(
                ["source_candidate_id", "rewrite_attempt"], keep="last"
            )

            completed_ids = set(
                checkpoint.loc[
                    checkpoint["plain_response"].fillna("").str.strip().ne(""),
                    "source_candidate_id",
                ]
            )
            pending = candidates[~candidates["source_candidate_id"].isin(completed_ids)].copy()
            print("Recovered rewrites:", len(completed_ids & set(candidates["source_candidate_id"])))
            print("Pending rewrites:", len(pending))

            processor = None
            model = None
            loaded_rewrite_model_id = None
            decoding = DecodingSpec(
                name="counterfactual_rewrite_greedy",
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=240,
            )

            def ensure_rewriter_loaded(model_id):
                global processor, model, loaded_rewrite_model_id
                if model is not None and loaded_rewrite_model_id != model_id:
                    model = None
                    processor = None
                    loaded_rewrite_model_id = None
                    gc.collect()
                    torch.cuda.empty_cache()
                if model is None:
                    processor, model = load_text_generation_model(
                        model_id,
                        quantize_4bit=QUANTIZE_4BIT,
                        loader_type="multimodal_lm",
                    )
                    loaded_rewrite_model_id = model_id

            def generate_attempt(targets, prompt_builder, attempt, prompt_version, model_id):
                global checkpoint
                if targets.empty:
                    return
                ensure_rewriter_loaded(model_id)
                for start in range(0, len(targets), REWRITE_BATCH_SIZE):
                    batch = targets.iloc[start : start + REWRITE_BATCH_SIZE]
                    records = batch.to_dict("records")
                    generated = generate_batch(
                        tokenizer=processor,
                        model=model,
                        prompts=[prompt_builder(record) for record in records],
                        decoding=decoding,
                        use_chat_template=True,
                        seed=42,
                    )
                    new_rows = pd.DataFrame(
                        {
                            "source_candidate_id": batch["source_candidate_id"].tolist(),
                            "plain_response": [normalize_plain_rewrite(text) for text in generated],
                            "rewrite_model_id": model_id,
                            "rewrite_run_id": REWARD_PAIR_RUN_ID,
                            "rewrite_attempt": attempt,
                            "rewrite_prompt_version": prompt_version,
                        }
                    )
                    checkpoint = (
                        pd.concat([checkpoint, new_rows], ignore_index=True)
                        .drop_duplicates(
                            ["source_candidate_id", "rewrite_attempt"], keep="last"
                        )
                    )
                    save_dataframe(checkpoint, checkpoint_path)
                    completed = min(start + len(batch), len(targets))
                    wandb.log({
                        "rewrite_attempt": attempt,
                        "rewrite_attempt_progress": completed,
                        "rewrite_attempt_total": len(targets),
                    })
                    print(f"Attempt {attempt}: rewritten {completed}/{len(targets)}")

            generate_attempt(
                pending,
                lambda record: make_plain_rewrite_prompt(record["response"]),
                attempt=1,
                prompt_version="v1",
                model_id=REWRITE_MODEL_ID,
            )

            for attempt in range(2, MAX_QWEN_REWRITE_ATTEMPTS + 1):
                prior_checkpoint = checkpoint[checkpoint["rewrite_attempt"] < attempt]
                attempt_records = candidates.merge(
                    prior_checkpoint,
                    on="source_candidate_id",
                    how="inner",
                    validate="one_to_many",
                )
                best_so_far, _ = select_best_plain_rewrites(attempt_records)
                retry_targets = best_so_far[~best_so_far["quality_pass"]].copy()
                completed_attempt_ids = set(
                    checkpoint.loc[
                        checkpoint["rewrite_attempt"].eq(attempt),
                        "source_candidate_id",
                    ]
                )
                pending_retry_targets = retry_targets[
                    ~retry_targets["source_candidate_id"].isin(completed_attempt_ids)
                ]
                print(
                    f"Attempt {attempt}: {len(retry_targets)} eligible, "
                    f"{len(pending_retry_targets)} pending"
                )
                if retry_targets.empty:
                    break
                generate_attempt(
                    pending_retry_targets,
                    lambda record: make_plain_rewrite_retry_prompt(
                        record["response"], record["plain_response"]
                    ),
                    attempt=attempt,
                    prompt_version="v2_strict_retry",
                    model_id=REWRITE_MODEL_ID,
                )

            prior_checkpoint = checkpoint[
                checkpoint["rewrite_attempt"] < MAX_REWRITE_ATTEMPTS
            ]
            attempt_records = candidates.merge(
                prior_checkpoint,
                on="source_candidate_id",
                how="inner",
                validate="one_to_many",
            )
            best_so_far, _ = select_best_plain_rewrites(attempt_records)
            fallback_targets = best_so_far[~best_so_far["quality_pass"]].copy()
            completed_fallback_ids = set(
                checkpoint.loc[
                    checkpoint["rewrite_attempt"].eq(MAX_REWRITE_ATTEMPTS),
                    "source_candidate_id",
                ]
            )
            pending_fallback_targets = fallback_targets[
                ~fallback_targets["source_candidate_id"].isin(completed_fallback_ids)
            ]
            print(
                f"Attempt {MAX_REWRITE_ATTEMPTS} ({FALLBACK_REWRITE_MODEL_ID}): "
                f"{len(fallback_targets)} eligible, {len(pending_fallback_targets)} pending"
            )
            generate_attempt(
                pending_fallback_targets,
                lambda record: make_plain_rewrite_retry_prompt(
                    record["response"], record["plain_response"]
                ),
                attempt=MAX_REWRITE_ATTEMPTS,
                prompt_version="v3_gemma_fallback",
                model_id=FALLBACK_REWRITE_MODEL_ID,
            )

            if model is not None:
                model = None
                processor = None
                loaded_rewrite_model_id = None
                gc.collect()
                torch.cuda.empty_cache()

            attempt_records = candidates.merge(
                checkpoint,
                on="source_candidate_id",
                how="inner",
                validate="one_to_many",
            )
            rewrites, rewrite_attempt_quality = select_best_plain_rewrites(attempt_records)
            assert len(rewrites) == len(candidates), "Not all candidates have a checkpointed rewrite."
            print("Checkpoint:", checkpoint_path)
            print("Rewrite attempts:", len(rewrite_attempt_quality))
            """
        ),
        code(
            r"""
            pairs, rewrite_quality = build_counterfactual_pair_frame(rewrites, seed=42)
            valid_pair_count = int(rewrite_quality["quality_pass"].sum())
            rejected = rewrite_quality[~rewrite_quality["quality_pass"]].copy()

            if valid_pair_count == 0:
                raise RuntimeError("No rewrites passed the counterfactual-pair quality controls.")
            if not pairs.groupby("pair_id").size().eq(2).all():
                raise AssertionError("Every published reward pair must contain exactly two variants.")
            separation = pairs.groupby("variant_type")["has_ocn"].mean()
            if separation.get("candidate_ocn") != 1.0 or separation.get("plain_rewrite") != 0.0:
                raise AssertionError("Published pairs do not have perfect lexical detector separation.")

            pair_path = save_dataframe(
                pairs,
                Path(config["drive_data_root"]) / "ocn_reward_pairs_main_gemma4_qwen35.csv",
            )
            quality_path = save_dataframe(
                rewrite_quality,
                Path(config["drive_data_root"]) / "ocn_reward_pair_rewrite_quality_main_gemma4_qwen35.csv",
            )
            attempt_quality_path = save_dataframe(
                rewrite_attempt_quality,
                Path(config["drive_data_root"]) / "ocn_reward_pair_rewrite_attempt_quality_main_gemma4_qwen35.csv",
            )
            rejected_path = save_dataframe(
                rejected,
                Path(config["drive_data_root"]) / "ocn_reward_pair_rewrite_rejects_main_gemma4_qwen35.csv",
            )
            repo_url = publish_dataframe_to_hf(
                pairs,
                repo_id=MAIN_REWARD_PAIR_REPO,
                split="train",
                private=config["hf_private"],
                card_path=REPO_ROOT / "dataset_cards/ocn_reward_pairs.md",
                commit_message=f"Publish {EXPERIMENT_ID} reward pairs {REWARD_PAIR_RUN_ID}",
            )

            model_counts = (
                rewrite_quality.groupby("model_id", dropna=False)
                .agg(candidates=("source_candidate_id", "size"), accepted=("quality_pass", "sum"))
                .reset_index()
            )
            attempt_yield = (
                rewrite_attempt_quality.groupby("rewrite_attempt", dropna=False)
                .agg(
                    attempts=("source_candidate_id", "size"),
                    passing=("quality_pass", "sum"),
                )
                .reset_index()
            )
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            model_plot = model_counts.melt(
                id_vars="model_id",
                value_vars=["candidates", "accepted"],
                var_name="status",
                value_name="count",
            )
            sns.barplot(data=model_plot, y="model_id", x="count", hue="status", ax=axes[0])
            axes[0].set_title("Counterfactual rewrite yield by source model")
            sns.histplot(
                data=rewrite_quality,
                x="length_ratio",
                hue="quality_pass",
                bins=30,
                multiple="layer",
                ax=axes[1],
            )
            axes[1].set_title("Plain/source token-length ratio")
            plt.tight_layout()
            figure_path = Path(config["drive_figure_root"]) / "04_reward_pair_quality_main_gemma4_qwen35.png"
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")

            wandb.log({
                "source_detection_rows": len(detections),
                "lexical_ocn_rows": int(detections["has_ocn"].sum()),
                "unique_rewrite_candidates": len(candidates),
                "accepted_pairs": valid_pair_count,
                "acceptance_rate": valid_pair_count / max(len(candidates), 1),
                "reward_pair_rows": len(pairs),
                "rewrite_attempt_rows": len(rewrite_attempt_quality),
                "max_rewrite_attempt": int(rewrite_attempt_quality["rewrite_attempt"].max()),
                "mean_content_overlap": float(rewrite_quality["content_overlap"].mean()),
                "mean_length_ratio": float(rewrite_quality["length_ratio"].mean()),
                "model_yield": wandb.Table(dataframe=model_counts),
                "attempt_yield": wandb.Table(dataframe=attempt_yield),
                "rewrite_quality": wandb.Table(dataframe=rewrite_quality),
                "reward_pairs": wandb.Table(dataframe=pairs),
                "reward_pair_quality_chart": wandb.Image(str(figure_path)),
            })
            run.finish()
            print("Saved:", pair_path)
            print("Quality audit:", quality_path)
            print("Attempt audit:", attempt_quality_path)
            print("Rejected rewrites:", rejected_path)
            print("Figure:", figure_path)
            print("Published:", repo_url)
            print("Accepted pairs:", valid_pair_count)
            separation
            """
        ),
    ]


def semantic_annotation_notebook() -> list[dict]:
    return [
        md(
            """
            # 05 - Sample, Annotate, And Adjudicate OCN Semantics

            This notebook turns lexical detector matches into a semantic research dataset. It collapses duplicate greedy generations, samples responses within every model-by-decoding stratum, annotates every detected span with two independent open-weight models, and has a third open-weight model adjudicate every item. Checkpoints are resumable on Google Drive, final splits are published to Hugging Face, and agreement and prevalence diagnostics are logged to W&B.

            Required runtime: an A100 GPU. The three models run sequentially in BF16, so this notebook does not use bitsandbytes or 4-bit quantization. The resulting labels are model-assisted annotations, not human gold labels. A blinded 100-item, two-annotator human audit packet is created for paper validation.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import gc
            import json
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            import wandb
            from datasets import Dataset, load_dataset
            from huggingface_hub import HfApi

            from ocn.annotation import (
                ANNOTATION_FIELDS,
                ANNOTATION_PROMPT_VERSION,
                add_semantic_outcomes,
                agreement_summary,
                annotation_calibration_frame,
                build_span_population,
                compare_annotations,
                finalize_adjudications,
                make_adjudication_prompt,
                make_annotation_packet,
                make_annotation_prompt,
                parse_annotation_json,
                stratified_response_sample,
                validate_annotation_frame,
                weighted_rate,
            )
            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, save_dataframe, utc_timestamp
            from ocn.generation import DecodingSpec, generate_batch, load_text_generation_model
            from ocn.metrics import weighted_group_bootstrap

            if not torch.cuda.is_available():
                raise RuntimeError("Notebook 05 requires an A100 GPU runtime in Colab.")
            GPU_NAME = torch.cuda.get_device_name(0)
            if "A100" not in GPU_NAME.upper():
                raise RuntimeError(
                    f"Notebook 05 requires an A100, but Colab assigned {GPU_NAME}. "
                    "Reconnect with Runtime > Change runtime type > A100 GPU."
                )
            print("Verified accelerator:", GPU_NAME)

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            HF_TOKEN = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            ANNOTATION_RUN_ID = utc_timestamp()
            MAIN_DETECTION_REPO = config.get(
                "hf_main_detection_repo",
                f"{config['hf_owner']}/ocn-empty-negations-detection-main-gemma4-qwen35",
            )
            MAIN_SEMANTIC_REPO = config.get(
                "hf_main_semantic_repo",
                f"{config['hf_owner']}/ocn-empty-negations-semantic-main-gemma4-qwen35",
            )

            TARGET_RESPONSES_PER_STRATUM = int(os.environ.get("OCN_ANNOTATION_STRATUM_N", "50"))
            SAMPLE_SEED = 20260817
            HUMAN_AUDIT_N = 100
            HUMAN_AUDIT_SEED = 20260818
            BATCH_SIZE = int(os.environ.get("OCN_ANNOTATION_BATCH_SIZE", "8"))
            MAX_PARSE_ATTEMPTS = 3
            ADJUDICATE_ALL = True
            PROMPT_VERSION = ANNOTATION_PROMPT_VERSION

            ANNOTATOR_SPECS = [
                {
                    "annotator_id": "annotator_a",
                    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                    "loader_type": "causal_lm",
                },
                {
                    "annotator_id": "annotator_b",
                    "model_id": "allenai/OLMo-2-1124-13B-Instruct",
                    "loader_type": "causal_lm",
                },
            ]
            ADJUDICATOR_SPEC = {
                "annotator_id": "adjudicator",
                "model_id": "Qwen/Qwen3-14B",
                "loader_type": "causal_lm",
            }
            decoding = DecodingSpec(
                "semantic_annotation_greedy",
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=384,
            )

            annotation_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "annotation_run_id": ANNOTATION_RUN_ID,
                "source_repo": MAIN_DETECTION_REPO,
                "output_repo": MAIN_SEMANTIC_REPO,
                "gpu_name": GPU_NAME,
                "precision": "bfloat16",
                "quantize_4bit": False,
                "target_responses_per_stratum": TARGET_RESPONSES_PER_STRATUM,
                "sample_seed": SAMPLE_SEED,
                "batch_size": BATCH_SIZE,
                "max_parse_attempts": MAX_PARSE_ATTEMPTS,
                "annotator_models": [spec["model_id"] for spec in ANNOTATOR_SPECS],
                "adjudicator_model": ADJUDICATOR_SPEC["model_id"],
                "adjudicate_all": ADJUDICATE_ALL,
                "annotation_prompt_version": PROMPT_VERSION,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"semantic-annotation-{EXPERIMENT_ID}-{ANNOTATION_RUN_ID}",
                config=annotation_config,
            )
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            detections = load_dataset(MAIN_DETECTION_REPO, split="train").to_pandas()
            population = build_span_population(detections)
            sample = stratified_response_sample(
                population,
                target_responses_per_stratum=TARGET_RESPONSES_PER_STRATUM,
                strata=("model_id", "decoding"),
                seed=SAMPLE_SEED,
            )
            calibration = annotation_calibration_frame()

            population_path = save_dataframe(
                population,
                Path(config["drive_data_root"]) / "ocn_semantic_span_population_main_gemma4_qwen35.csv",
            )
            sample_path = save_dataframe(
                sample,
                Path(config["drive_data_root"]) / "ocn_semantic_annotation_sample_main_gemma4_qwen35.csv",
            )
            response_allocation = (
                sample.drop_duplicates("response_id")
                .groupby(["model_id", "decoding"], dropna=False)
                .agg(
                    population_responses=("stratum_population_responses", "first"),
                    sampled_responses=("response_id", "size"),
                )
                .reset_index()
            )
            span_allocation = (
                sample.groupby(["model_id", "decoding"], dropna=False)
                .size()
                .rename("sampled_spans")
                .reset_index()
            )
            allocation = response_allocation.merge(
                span_allocation,
                on=["model_id", "decoding"],
                validate="one_to_one",
            )
            print("Detection rows:", len(detections))
            print("Lexical candidate rows:", int(detections["has_ocn"].sum()))
            print("Unique candidate responses:", population["response_id"].nunique())
            print("Candidate spans:", len(population))
            print("Sampled responses:", sample["response_id"].nunique())
            print("Sampled spans:", len(sample))
            print("Held-out calibration cases:", len(calibration))
            display(allocation)
            """
        ),
        code(
            r"""
            checkpoint_root = (
                Path(config["drive_data_root"])
                / "semantic_annotation_runs"
                / EXPERIMENT_ID
                / PROMPT_VERSION
            )
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            def checkpoint_path_for(annotator_id):
                return checkpoint_root / f"{annotator_id}.csv"

            def load_checkpoint(annotator_id):
                path = checkpoint_path_for(annotator_id)
                if not path.exists():
                    return pd.DataFrame()
                frame = pd.read_csv(path, dtype={"example_id": str})
                if "attempt" in frame.columns:
                    frame["attempt"] = frame["attempt"].astype(int)
                parse_ok = frame["parse_ok"].astype(str).str.lower().eq("true")
                recovered = 0
                for index in frame.index[~parse_ok]:
                    try:
                        parsed = parse_annotation_json(str(frame.at[index, "raw_model_output"]))
                    except Exception:
                        continue
                    for field, value in parsed.items():
                        frame.at[index, field] = value
                    frame.at[index, "parse_ok"] = True
                    frame.at[index, "parse_error"] = ""
                    frame.at[index, "reparsed_from_checkpoint"] = True
                    recovered += 1
                if recovered:
                    save_dataframe(frame, path)
                    print(f"{annotator_id}: reparsed {recovered} checkpoint rows")
                return frame

            def completed_annotations(checkpoint, expected_ids):
                if checkpoint.empty or "parse_ok" not in checkpoint.columns:
                    return pd.DataFrame()
                parse_ok = checkpoint["parse_ok"].astype(str).str.lower().eq("true")
                valid = checkpoint[parse_ok].copy()
                valid = valid.sort_values("attempt").drop_duplicates("example_id", keep="last")
                valid = valid[valid["example_id"].isin(expected_ids)]
                return valid

            def run_model_annotations(items, spec, prompt_builder):
                annotator_id = spec["annotator_id"]
                expected_ids = set(items["example_id"])
                checkpoint = load_checkpoint(annotator_id)
                completed = completed_annotations(checkpoint, expected_ids)
                pending_ids = expected_ids - set(completed.get("example_id", []))
                print(f"{annotator_id}: recovered {len(completed)}, pending {len(pending_ids)}")
                if not pending_ids:
                    return validate_annotation_frame(completed, expected_ids)

                print("Loading", spec["model_id"])
                processor, model = load_text_generation_model(
                    spec["model_id"],
                    quantize_4bit=False,
                    loader_type=spec["loader_type"],
                )
                model_revision = getattr(model.config, "_commit_hash", None)

                try:
                    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
                        completed = completed_annotations(checkpoint, expected_ids)
                        pending_ids = expected_ids - set(completed.get("example_id", []))
                        if not pending_ids:
                            break
                        pending = items[items["example_id"].isin(pending_ids)].copy()
                        previous_output = {}
                        if not checkpoint.empty:
                            previous = checkpoint[checkpoint["example_id"].isin(pending_ids)]
                            previous = previous.sort_values("attempt").drop_duplicates("example_id", keep="last")
                            previous_output = dict(zip(previous["example_id"], previous["raw_model_output"]))

                        print(f"{annotator_id}: attempt {attempt}, {len(pending)} pending")
                        records = pending.to_dict("records")
                        for start in range(0, len(records), BATCH_SIZE):
                            batch = records[start : start + BATCH_SIZE]
                            prompts = []
                            for record in batch:
                                base_prompt = prompt_builder(record)
                                invalid = previous_output.get(record["example_id"])
                                if invalid:
                                    base_prompt = (
                                        "Your previous response failed schema validation. Return only one compact "
                                        "JSON object for the original request, with no Markdown. Every rating must "
                                        "be an integer from 1 through 5, never 0. Empty rejected_x or asserted_y is "
                                        "allowed when no proposition exists. Keep notes under 50 words.\n\n"
                                        "ORIGINAL REQUEST:\n"
                                        + base_prompt
                                    )
                                prompts.append(base_prompt)
                            outputs = generate_batch(
                                tokenizer=processor,
                                model=model,
                                prompts=prompts,
                                decoding=decoding,
                                use_chat_template=True,
                                seed=SAMPLE_SEED + attempt,
                            )
                            new_rows = []
                            for record, output in zip(batch, outputs):
                                row = {
                                    "example_id": record["example_id"],
                                    "annotator_id": annotator_id,
                                    "annotator_type": "open_weight_model",
                                    "annotator_model_id": spec["model_id"],
                                    "model_revision": model_revision,
                                    "annotation_run_id": ANNOTATION_RUN_ID,
                                    "attempt": attempt,
                                    "raw_model_output": output,
                                    "parse_ok": False,
                                    "parse_error": "",
                                }
                                try:
                                    row.update(parse_annotation_json(output))
                                    row["parse_ok"] = True
                                except Exception as exc:
                                    row["parse_error"] = str(exc)
                                new_rows.append(row)
                            checkpoint = pd.concat(
                                [checkpoint, pd.DataFrame(new_rows)], ignore_index=True
                            )
                            save_dataframe(checkpoint, checkpoint_path_for(annotator_id))
                            wandb.log({
                                f"{annotator_id}/checkpoint_rows": len(checkpoint),
                                f"{annotator_id}/completed": len(completed_annotations(checkpoint, expected_ids)),
                            })
                            print(
                                f"{annotator_id}: {len(completed_annotations(checkpoint, expected_ids))}/"
                                f"{len(expected_ids)} valid"
                            )
                finally:
                    model = None
                    processor = None
                    gc.collect()
                    torch.cuda.empty_cache()

                completed = completed_annotations(checkpoint, expected_ids)
                if len(completed) != len(expected_ids):
                    failed = sorted(expected_ids - set(completed["example_id"]))
                    raise RuntimeError(
                        f"{annotator_id} still has {len(failed)} parse failures after "
                        f"{MAX_PARSE_ATTEMPTS} attempts. Checkpoint is preserved at "
                        f"{checkpoint_path_for(annotator_id)}."
                    )
                return validate_annotation_frame(completed, expected_ids)
            """
        ),
        code(
            r"""
            panel_items = pd.concat([sample, calibration], ignore_index=True, sort=False)
            sample_ids = set(sample["example_id"])
            calibration_ids = set(calibration["example_id"])
            annotation_frames = {}
            calibration_rows = []
            for spec in ANNOTATOR_SPECS:
                combined_annotations = run_model_annotations(
                    panel_items,
                    spec,
                    make_annotation_prompt,
                )
                annotation_frames[spec["annotator_id"]] = combined_annotations[
                    combined_annotations["example_id"].isin(sample_ids)
                ].copy()
                calibration_annotations = combined_annotations[
                    combined_annotations["example_id"].isin(calibration_ids)
                ].merge(
                    calibration[
                        ["example_id", "prompt", "response", "span_text", "expected_taxonomy_label"]
                    ],
                    on="example_id",
                    validate="one_to_one",
                )
                calibration_annotations["calibration_correct"] = (
                    calibration_annotations["taxonomy_label"]
                    == calibration_annotations["expected_taxonomy_label"]
                )
                calibration_rows.append(calibration_annotations)

            annotation_a = annotation_frames["annotator_a"]
            annotation_b = annotation_frames["annotator_b"]
            comparison = compare_annotations(sample, annotation_a, annotation_b)
            calibration_a = calibration_rows[0]
            calibration_b = calibration_rows[1]
            calibration_comparison = compare_annotations(
                calibration,
                calibration_a,
                calibration_b,
            )
            agreement = agreement_summary(comparison)
            disagreement_count = int(comparison["adjudication_required"].sum())
            calibration_summary = pd.DataFrame([
                {
                    "panel_member": ANNOTATOR_SPECS[0]["annotator_id"],
                    "model_id": ANNOTATOR_SPECS[0]["model_id"],
                    "taxonomy_accuracy": float(calibration_a["calibration_correct"].mean()),
                    "correct": int(calibration_a["calibration_correct"].sum()),
                    "total": len(calibration_a),
                },
                {
                    "panel_member": ANNOTATOR_SPECS[1]["annotator_id"],
                    "model_id": ANNOTATOR_SPECS[1]["model_id"],
                    "taxonomy_accuracy": float(calibration_b["calibration_correct"].mean()),
                    "correct": int(calibration_b["calibration_correct"].sum()),
                    "total": len(calibration_b),
                },
            ])
            print("Flagged disagreements:", disagreement_count, "/", len(comparison))
            display(calibration_summary)
            display(agreement)
            """
        ),
        code(
            r"""
            main_adjudication_items = comparison if ADJUDICATE_ALL else comparison[
                comparison["adjudication_required"]
            ]
            adjudication_items = pd.concat(
                [main_adjudication_items, calibration_comparison],
                ignore_index=True,
                sort=False,
            )
            combined_adjudications = run_model_annotations(
                adjudication_items,
                ADJUDICATOR_SPEC,
                make_adjudication_prompt,
            )
            adjudications = combined_adjudications[
                combined_adjudications["example_id"].isin(sample_ids)
            ].copy()
            calibration_adjudications = combined_adjudications[
                combined_adjudications["example_id"].isin(calibration_ids)
            ].merge(
                calibration[
                    ["example_id", "prompt", "response", "span_text", "expected_taxonomy_label"]
                ],
                on="example_id",
                validate="one_to_one",
            )
            calibration_adjudications["calibration_correct"] = (
                calibration_adjudications["taxonomy_label"]
                == calibration_adjudications["expected_taxonomy_label"]
            )
            calibration_summary = pd.concat(
                [
                    calibration_summary,
                    pd.DataFrame([{
                        "panel_member": ADJUDICATOR_SPEC["annotator_id"],
                        "model_id": ADJUDICATOR_SPEC["model_id"],
                        "taxonomy_accuracy": float(
                            calibration_adjudications["calibration_correct"].mean()
                        ),
                        "correct": int(calibration_adjudications["calibration_correct"].sum()),
                        "total": len(calibration_adjudications),
                    }]),
                ],
                ignore_index=True,
            )
            calibration_detail = pd.concat(
                [
                    calibration_a.assign(panel_member="annotator_a"),
                    calibration_b.assign(panel_member="annotator_b"),
                    calibration_adjudications.assign(panel_member="adjudicator"),
                ],
                ignore_index=True,
                sort=False,
            )
            calibration_detail["annotation_prompt_version"] = PROMPT_VERSION
            final = finalize_adjudications(
                comparison,
                adjudications,
                adjudicate_all=ADJUDICATE_ALL,
            )
            final["annotator_a_model_id"] = ANNOTATOR_SPECS[0]["model_id"]
            final["annotator_b_model_id"] = ANNOTATOR_SPECS[1]["model_id"]
            final["adjudicator_model_id"] = ADJUDICATOR_SPEC["model_id"]
            final["annotation_run_id"] = ANNOTATION_RUN_ID
            final["annotation_prompt_version"] = PROMPT_VERSION

            semantic_rate_rows = []
            for key, group in final.groupby(
                ["model_id", "model_stage", "decoding"], dropna=False
            ):
                semantic_rate_rows.append(
                    {
                        "model_id": key[0],
                        "model_stage": key[1],
                        "decoding": key[2],
                        "sampled_spans": len(group),
                        "weighted_strict_misuse_rate": weighted_rate(group, "strict_misuse"),
                        "weighted_broad_misuse_rate": weighted_rate(group, "broad_misuse"),
                        "weighted_unsupported_contrast_rate": weighted_rate(group, "unsupported_contrast"),
                    }
                )
            semantic_rates = pd.DataFrame(semantic_rate_rows)
            semantic_intervals = weighted_group_bootstrap(
                final,
                outcomes=["strict_misuse", "broad_misuse", "unsupported_contrast"],
                group_columns=["model_id", "model_stage", "decoding"],
                cluster_column="response_id",
                weight_column="sample_weight",
                n_boot=2000,
                seed=20260819,
            )
            display(semantic_rates)
            display(calibration_summary)
            """
        ),
        code(
            r"""
            final_path = save_dataframe(
                final,
                Path(config["drive_data_root"])
                / f"ocn_semantic_adjudicated_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            agreement_path = save_dataframe(
                agreement,
                Path(config["drive_data_root"])
                / f"ocn_semantic_agreement_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            annotation_a_path = save_dataframe(
                annotation_a,
                Path(config["drive_data_root"])
                / f"ocn_semantic_annotations_a_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            annotation_b_path = save_dataframe(
                annotation_b,
                Path(config["drive_data_root"])
                / f"ocn_semantic_annotations_b_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            adjudication_path = save_dataframe(
                adjudications,
                Path(config["drive_data_root"])
                / f"ocn_semantic_adjudications_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            calibration_path = save_dataframe(
                calibration_summary,
                Path(config["drive_data_root"])
                / f"ocn_semantic_calibration_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            calibration_detail_path = save_dataframe(
                calibration_detail,
                Path(config["drive_data_root"])
                / f"ocn_semantic_calibration_detail_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            semantic_intervals_path = save_dataframe(
                semantic_intervals,
                Path(config["drive_data_root"])
                / f"ocn_semantic_intervals_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )

            audit_n = min(HUMAN_AUDIT_N, len(final))
            disagreement_pool = final[final["adjudication_required"]]
            priority_n = min(len(disagreement_pool), audit_n // 2)
            priority = disagreement_pool.sample(n=priority_n, random_state=HUMAN_AUDIT_SEED)
            remainder = final[~final["example_id"].isin(priority["example_id"])]
            remainder = remainder.sample(
                n=audit_n - priority_n,
                random_state=HUMAN_AUDIT_SEED + 1,
            )
            human_audit_source = pd.concat([priority, remainder], ignore_index=True)
            human_audit_a = make_annotation_packet(
                human_audit_source,
                packet_id="human_audit_a",
                seed=HUMAN_AUDIT_SEED + 2,
            )
            human_audit_b = make_annotation_packet(
                human_audit_source,
                packet_id="human_audit_b",
                seed=HUMAN_AUDIT_SEED + 3,
            )
            human_audit_a_path = save_dataframe(
                human_audit_a,
                Path(config["drive_data_root"])
                / f"ocn_semantic_human_audit_a_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )
            human_audit_b_path = save_dataframe(
                human_audit_b,
                Path(config["drive_data_root"])
                / f"ocn_semantic_human_audit_b_main_gemma4_qwen35_{PROMPT_VERSION}.csv",
            )

            public_audit = human_audit_source[
                ["example_id", "prompt", "response", "span_text", "model_id", "decoding"]
            ].copy()
            public_audit["annotation_prompt_version"] = PROMPT_VERSION
            hub_configs = {
                "sample": sample,
                "annotator_a": annotation_a,
                "annotator_b": annotation_b,
                "adjudicated": final,
                "agreement": agreement,
                "human_audit": public_audit,
                "calibration": calibration_detail,
            }
            for config_name, frame in hub_configs.items():
                Dataset.from_pandas(frame, preserve_index=False).push_to_hub(
                    MAIN_SEMANTIC_REPO,
                    config_name=config_name,
                    split="train",
                    private=config["hf_private"],
                    token=HF_TOKEN,
                    commit_message=(
                        f"Publish {config_name} semantic annotations {ANNOTATION_RUN_ID}"
                    ),
                )
            HfApi(token=HF_TOKEN).upload_file(
                path_or_fileobj=str(REPO_ROOT / "dataset_cards/ocn_semantic.md"),
                path_in_repo="README.md",
                repo_id=MAIN_SEMANTIC_REPO,
                repo_type="dataset",
                commit_message="Add semantic annotation dataset card",
            )

            fig, axes = plt.subplots(1, 3, figsize=(19, 6))
            final["taxonomy_label"].value_counts().sort_values().plot(
                kind="barh", ax=axes[0], color="#4c78a8"
            )
            axes[0].set_title("Adjudicated taxonomy")
            agreement.plot(
                x="field", y="cohen_kappa", kind="bar", ax=axes[1], color="#f58518", legend=False
            )
            axes[1].set_title("Inter-model agreement")
            axes[1].set_ylim(-0.1, 1)
            axes[1].tick_params(axis="x", rotation=60)
            rate_plot = semantic_rates.melt(
                id_vars=["model_id", "model_stage", "decoding", "sampled_spans"],
                value_vars=[
                    "weighted_strict_misuse_rate",
                    "weighted_broad_misuse_rate",
                    "weighted_unsupported_contrast_rate",
                ],
                var_name="metric",
                value_name="rate",
            )
            sns.barplot(data=rate_plot, y="model_id", x="rate", hue="metric", ax=axes[2])
            axes[2].set_title("Weighted semantic rates")
            axes[2].set_xlim(0, 1)
            plt.tight_layout()
            figure_path = (
                Path(config["drive_figure_root"])
                / f"05_semantic_annotation_main_gemma4_qwen35_{PROMPT_VERSION}.png"
            )
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")

            wandb.log({
                "population_responses": population["response_id"].nunique(),
                "population_spans": len(population),
                "sample_responses": sample["response_id"].nunique(),
                "sample_spans": len(sample),
                "flagged_disagreements": disagreement_count,
                "flagged_disagreement_rate": disagreement_count / len(comparison),
                "weighted_strict_misuse_rate": weighted_rate(final, "strict_misuse"),
                "weighted_broad_misuse_rate": weighted_rate(final, "broad_misuse"),
                "weighted_unsupported_contrast_rate": weighted_rate(final, "unsupported_contrast"),
                "sampling_allocation": wandb.Table(dataframe=allocation),
                "agreement": wandb.Table(dataframe=agreement),
                "semantic_rates": wandb.Table(dataframe=semantic_rates),
                "semantic_intervals": wandb.Table(dataframe=semantic_intervals),
                "calibration": wandb.Table(dataframe=calibration_summary),
                "calibration_detail": wandb.Table(dataframe=calibration_detail),
                "adjudicated_sample": wandb.Table(dataframe=final),
                "semantic_annotation_dashboard": wandb.Image(str(figure_path)),
            })
            run.finish()

            print("Saved sample:", sample_path)
            print("Saved final annotations:", final_path)
            print("Saved agreement:", agreement_path)
            print("Saved calibration:", calibration_path)
            print("Saved calibration detail:", calibration_detail_path)
            print("Saved semantic intervals:", semantic_intervals_path)
            print("Human audit packets:", human_audit_a_path, human_audit_b_path)
            print("Published:", f"https://huggingface.co/datasets/{MAIN_SEMANTIC_REPO}")
            print("Figure:", figure_path)
            """
        ),
    ]


def analysis_notebook() -> list[dict]:
    return [
        md(
            """
            # 06 - Analysis And Reporting

            This notebook produces the paper-facing analysis while keeping lexical detection and model-panel semantics separate. It removes deterministic greedy seed duplicates, reports interval estimates, fits a prompt-clustered binomial model, and treats the two annotators plus adjudicator as a sensitivity analysis rather than human ground truth.

            A GPU is not required. Run notebook `05` first so the latest calibrated semantic configurations are available on Hugging Face.
            """
        ),
        code(COMMON_BOOTSTRAP),
        code(
            r"""
            import json
            from pathlib import Path
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            import wandb
            from datasets import load_dataset

            from ocn.annotation import add_semantic_outcomes
            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, save_dataframe, utc_timestamp
            from ocn.metrics import (
                deduplicate_greedy_seed_reuse,
                detection_summary,
                grouped_ocn_rates_with_ci,
                top_patterns,
                weighted_group_bootstrap,
            )

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            ANALYSIS_VERSION = "v2_robust"
            ANALYSIS_RUN_ID = utc_timestamp()
            MAIN_DETECTION_REPO = config.get(
                "hf_main_detection_repo",
                f"{config['hf_owner']}/ocn-empty-negations-detection-main-gemma4-qwen35",
            )
            MAIN_SEMANTIC_REPO = config.get(
                "hf_main_semantic_repo",
                f"{config['hf_owner']}/ocn-empty-negations-semantic-main-gemma4-qwen35",
            )
            analysis_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "analysis_run_id": ANALYSIS_RUN_ID,
                "analysis_version": ANALYSIS_VERSION,
                "detection_repo": MAIN_DETECTION_REPO,
                "semantic_repo": MAIN_SEMANTIC_REPO,
                "greedy_seed_deduplication": True,
                "regression_covariance": "prompt_clustered",
                "semantic_bootstrap_replicates": 2000,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"analysis-{EXPERIMENT_ID}-{ANALYSIS_RUN_ID}",
                config=analysis_config,
            )
            sns.set_theme(style="whitegrid")
            """
        ),
        code(
            r"""
            raw_df = load_dataset(MAIN_DETECTION_REPO, split="train").to_pandas()
            df = deduplicate_greedy_seed_reuse(raw_df)
            semantics = load_dataset(
                MAIN_SEMANTIC_REPO, "adjudicated", split="train"
            ).to_pandas()
            calibration = load_dataset(
                MAIN_SEMANTIC_REPO, "calibration", split="train"
            ).to_pandas()

            duplicate_audit = pd.DataFrame([{
                "raw_rows": len(raw_df),
                "analysis_rows": len(df),
                "removed_greedy_seed_duplicates": len(raw_df) - len(df),
                "raw_greedy_rows": int(raw_df["decoding"].eq("greedy").sum()),
                "analysis_greedy_rows": int(df["decoding"].eq("greedy").sum()),
            }])
            summary = detection_summary(df)
            model_rates = grouped_ocn_rates_with_ci(
                df,
                ["model_id", "model_stage", "model_family", "decoding"],
                cluster_column="prompt_id",
                n_boot=2000,
                seed=20260821,
            )
            variant_rates = grouped_ocn_rates_with_ci(
                df, ["variant"], cluster_column="prompt_id", n_boot=2000, seed=20260822
            )
            persona_rates = grouped_ocn_rates_with_ci(
                df, ["persona"], cluster_column="prompt_id", n_boot=2000, seed=20260823
            )
            category_rates = grouped_ocn_rates_with_ci(
                df, ["category"], cluster_column="prompt_id", n_boot=2000, seed=20260824
            )
            patterns = top_patterns(df, 12)

            output_root = Path(config["drive_data_root"]) / "analysis" / EXPERIMENT_ID / ANALYSIS_VERSION
            output_root.mkdir(parents=True, exist_ok=True)
            save_dataframe(duplicate_audit, output_root / "deduplication_audit.csv")
            save_dataframe(model_rates, output_root / "lexical_model_rates.csv")
            save_dataframe(variant_rates, output_root / "lexical_variant_rates.csv")
            save_dataframe(persona_rates, output_root / "lexical_persona_rates.csv")
            save_dataframe(category_rates, output_root / "lexical_category_rates.csv")
            save_dataframe(patterns, output_root / "lexical_patterns.csv")
            display(duplicate_audit)
            display(summary.to_frame("value"))
            """
        ),
        code(
            r"""
            panel_frames = []
            panel_specs = [
                ("annotator_a", "a_taxonomy_label", "a_prompt_support"),
                ("annotator_b", "b_taxonomy_label", "b_prompt_support"),
                ("adjudicator", "taxonomy_label", "prompt_support"),
            ]
            for panel_source, taxonomy_column, support_column in panel_specs:
                panel = add_semantic_outcomes(
                    semantics,
                    taxonomy_column=taxonomy_column,
                    prompt_support_column=support_column,
                )
                panel["panel_source"] = panel_source
                panel_frames.append(panel)
            semantic_sensitivity_rows = pd.concat(panel_frames, ignore_index=True, sort=False)

            semantic_overall = weighted_group_bootstrap(
                semantic_sensitivity_rows,
                outcomes=["strict_misuse", "broad_misuse", "unsupported_contrast"],
                group_columns=["panel_source"],
                cluster_column="response_id",
                weight_column="sample_weight",
                n_boot=2000,
                seed=20260819,
            )
            semantic_by_model = weighted_group_bootstrap(
                semantic_sensitivity_rows,
                outcomes=["strict_misuse", "broad_misuse", "unsupported_contrast"],
                group_columns=["panel_source", "model_id", "model_stage", "decoding"],
                cluster_column="response_id",
                weight_column="sample_weight",
                n_boot=2000,
                seed=20260820,
            )
            calibration_summary = (
                calibration.groupby(["panel_member", "annotator_model_id"], dropna=False)
                .agg(
                    calibration_cases=("example_id", "size"),
                    calibration_correct=("calibration_correct", "sum"),
                )
                .reset_index()
            )
            calibration_summary["calibration_accuracy"] = (
                calibration_summary["calibration_correct"]
                / calibration_summary["calibration_cases"]
            )
            save_dataframe(semantic_overall, output_root / "semantic_panel_sensitivity_overall.csv")
            save_dataframe(semantic_by_model, output_root / "semantic_panel_sensitivity_by_model.csv")
            save_dataframe(calibration_summary, output_root / "semantic_panel_calibration.csv")
            display(semantic_overall)
            display(calibration_summary)
            """
        ),
        code(
            r"""
            regression_df = df.copy()
            regression_df["has_ocn_int"] = regression_df["has_ocn"].astype(int)
            formula = (
                "has_ocn_int ~ C(model_stage) * C(model_family) + C(decoding) + "
                "C(variant) + C(persona) + C(category) + length_target"
            )
            model = smf.glm(
                formula,
                data=regression_df,
                family=sm.families.Binomial(),
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": regression_df["prompt_id"]},
            )
            confidence = model.conf_int()
            coefficient_table = pd.DataFrame({
                "term": model.params.index,
                "log_odds": model.params.values,
                "clustered_std_error": model.bse.values,
                "p_value": model.pvalues.values,
                "odds_ratio": np.exp(model.params.values),
                "odds_ratio_ci_low": np.exp(confidence[0].values),
                "odds_ratio_ci_high": np.exp(confidence[1].values),
            })
            report_path = output_root / "clustered_binomial_model_summary.txt"
            report_path.write_text(model.summary().as_text(), encoding="utf-8")
            save_dataframe(coefficient_table, output_root / "clustered_binomial_coefficients.csv")
            print(model.summary())
            """
        ),
        code(
            r"""
            def interval_dotplot(frame, label_column, rate_column, low_column, high_column, ax, color):
                plot = frame.sort_values(rate_column).reset_index(drop=True)
                positions = np.arange(len(plot))
                ax.errorbar(
                    plot[rate_column],
                    positions,
                    xerr=np.vstack([
                        plot[rate_column] - plot[low_column],
                        plot[high_column] - plot[rate_column],
                    ]),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=3,
                )
                ax.set_yticks(positions, plot[label_column])
                ax.set_xlim(0, 1)

            fig, axes = plt.subplots(2, 3, figsize=(22, 13))
            model_plot = model_rates.copy()
            model_plot["label"] = (
                model_plot["model_id"].str.split("/").str[-1]
                + " | " + model_plot["decoding"].astype(str)
            )
            interval_dotplot(
                model_plot, "label", "ocn_rate", "ocn_rate_ci_low", "ocn_rate_ci_high",
                axes[0, 0], "#35618f",
            )
            axes[0, 0].set_title("Lexical OCN rate by model and decoding")

            interval_dotplot(
                variant_rates, "variant", "ocn_rate", "ocn_rate_ci_low", "ocn_rate_ci_high",
                axes[0, 1], "#e17829",
            )
            axes[0, 1].set_title("Lexical OCN rate by prompt variant")

            sns.barplot(data=patterns, y="pattern", x="count", ax=axes[0, 2], color="#6f8f5d")
            axes[0, 2].set_title("Top lexical detector patterns")

            semantics["taxonomy_label"].value_counts().sort_values().plot(
                kind="barh", ax=axes[1, 0], color="#bd4d4d"
            )
            axes[1, 0].set_title("Adjudicator taxonomy (model-assisted)")

            sensitivity_plot = semantic_overall.copy()
            interval_dotplot(
                sensitivity_plot,
                "panel_source",
                "strict_misuse_rate",
                "strict_misuse_ci_low",
                "strict_misuse_ci_high",
                axes[1, 1],
                "#71588f",
            )
            axes[1, 1].set_title("Strict misuse sensitivity by panel member")

            calibration_plot = calibration_summary.copy()
            calibration_plot["label"] = calibration_plot["panel_member"]
            sns.barplot(
                data=calibration_plot,
                y="label",
                x="calibration_accuracy",
                ax=axes[1, 2],
                color="#2d8b84",
            )
            axes[1, 2].axvline(1 / 8, color="black", linestyle="--", linewidth=1)
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_title("Held-out calibration accuracy")
            plt.tight_layout()

            figure_root = Path(config["drive_figure_root"]) / "analysis" / EXPERIMENT_ID / ANALYSIS_VERSION
            figure_root.mkdir(parents=True, exist_ok=True)
            fig_path = figure_root / "06_analysis_dashboard.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")

            report_markdown = f'''# OCN Main Analysis ({ANALYSIS_VERSION})

            - Raw generation rows: {len(raw_df):,}
            - Analysis rows after deterministic greedy deduplication: {len(df):,}
            - Removed greedy seed duplicates: {len(raw_df) - len(df):,}
            - Lexical OCN rate: {summary['ocn_rate']:.3f}
            - Prompt-clustered binomial regression: `{formula}`
            - Semantic estimates: response-cluster bootstrap with 2,000 replicates

            Semantic results are model-panel sensitivity estimates. They are not human-gold prevalence estimates and remain provisional until the blinded audit is independently annotated and adjudicated.
            '''
            report_markdown = "\n".join(line.strip() for line in report_markdown.splitlines()).strip() + "\n"
            report_md_path = output_root / "analysis_report.md"
            report_md_path.write_text(report_markdown, encoding="utf-8")

            wandb.log({
                **summary.to_dict(),
                "raw_rows": len(raw_df),
                "analysis_rows": len(df),
                "removed_greedy_seed_duplicates": len(raw_df) - len(df),
                "analysis_dashboard": wandb.Image(str(fig_path)),
                "model_rates": wandb.Table(dataframe=model_rates),
                "variant_rates": wandb.Table(dataframe=variant_rates),
                "persona_rates": wandb.Table(dataframe=persona_rates),
                "category_rates": wandb.Table(dataframe=category_rates),
                "semantic_panel_sensitivity": wandb.Table(dataframe=semantic_overall),
                "semantic_panel_sensitivity_by_model": wandb.Table(dataframe=semantic_by_model),
                "semantic_panel_calibration": wandb.Table(dataframe=calibration_summary),
                "clustered_binomial_coefficients": wandb.Table(dataframe=coefficient_table),
                "clustered_binomial_summary": model.summary().as_text(),
            })
            run.finish()
            print("Saved analysis tables:", output_root)
            print("Saved dashboard:", fig_path)
            print("Saved report:", report_md_path)
            """
        ),
    ]


def reward_scoring_notebook() -> list[dict]:
    return [
        md(
            """
            # 07 - Optional Reward Model Scoring

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

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, publish_dataframe_to_hf, save_dataframe, utc_timestamp

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            REWARD_SCORE_RUN_ID = utc_timestamp()
            MAIN_REWARD_PAIR_REPO = config.get(
                "hf_main_reward_pairs_repo",
                f"{config['hf_owner']}/ocn-empty-negations-reward-pairs-main-gemma4-qwen35",
            )
            MAIN_REWARD_SCORE_REPO = config.get(
                "hf_main_reward_scores_repo",
                f"{config['hf_owner']}/ocn-empty-negations-reward-scores-main-gemma4-qwen35",
            )
            score_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "reward_score_run_id": REWARD_SCORE_RUN_ID,
                "source_repo": MAIN_REWARD_PAIR_REPO,
                "output_repo": MAIN_REWARD_SCORE_REPO,
            }
            run = login_wandb(
                project="ocn-empty-negations",
                name=f"reward-score-{EXPERIMENT_ID}-{REWARD_SCORE_RUN_ID}",
                config=score_config,
            )
            """
        ),
        code(
            r"""
            pairs = load_dataset(MAIN_REWARD_PAIR_REPO, split="train").to_pandas()
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
            scores_path = save_dataframe(
                pairs,
                Path(config["drive_data_root"]) / "ocn_reward_scores_main_gemma4_qwen35.csv",
            )
            repo_url = publish_dataframe_to_hf(
                pairs,
                repo_id=MAIN_REWARD_SCORE_REPO,
                split="train",
                private=config["hf_private"],
                commit_message=f"Publish {EXPERIMENT_ID} reward scores {REWARD_SCORE_RUN_ID}",
            )

            deltas = (
                pairs.pivot_table(index="pair_id", columns="variant_type", values="reward_score", aggfunc="mean")
                .assign(
                    candidate_ocn_minus_plain=lambda x: x["candidate_ocn"] - x["plain_rewrite"],
                )
                .reset_index()
            )
            wandb.log({
                "reward_scores": wandb.Table(dataframe=pairs),
                "reward_deltas": wandb.Table(dataframe=deltas),
                "mean_candidate_ocn_minus_plain": float(deltas["candidate_ocn_minus_plain"].mean()),
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
            # 08 - Optional LoRA Style Intervention

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
            _ = login_huggingface("HF_WRITE_ACCESS")
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
    write_notebook("04_create_reward_pair_dataset.ipynb", reward_pairs_notebook(), gpu_type="L4")
    write_notebook("05_annotation_sampling_and_adjudication.ipynb", semantic_annotation_notebook())
    write_notebook("06_analysis_and_reporting.ipynb", analysis_notebook())
    write_notebook("07_optional_reward_model_scoring.ipynb", reward_scoring_notebook())
    write_notebook("08_optional_lora_style_intervention.ipynb", lora_notebook())


if __name__ == "__main__":
    main()
