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
            REWRITE_BATCH_SIZE = int(os.environ.get("OCN_REWRITE_BATCH_SIZE", "12"))
            candidate_limit = int(os.environ.get("OCN_REWARD_PAIR_LIMIT", "0"))
            MAX_CANDIDATES = candidate_limit or None
            MAX_REWRITE_ATTEMPTS = 3
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
            decoding = DecodingSpec(
                name="counterfactual_rewrite_greedy",
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=240,
            )

            def ensure_rewriter_loaded():
                global processor, model
                if model is None:
                    processor, model = load_text_generation_model(
                        REWRITE_MODEL_ID,
                        quantize_4bit=QUANTIZE_4BIT,
                        loader_type="multimodal_lm",
                    )

            def generate_attempt(targets, prompt_builder, attempt, prompt_version):
                global checkpoint
                if targets.empty:
                    return
                ensure_rewriter_loaded()
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
                            "rewrite_model_id": REWRITE_MODEL_ID,
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
            )

            for attempt in range(2, MAX_REWRITE_ATTEMPTS + 1):
                attempt_records = candidates.merge(
                    checkpoint,
                    on="source_candidate_id",
                    how="inner",
                    validate="one_to_many",
                )
                best_so_far, _ = select_best_plain_rewrites(attempt_records)
                retry_targets = best_so_far[~best_so_far["quality_pass"]].copy()
                print(f"Attempt {attempt}: retrying {len(retry_targets)} failed candidates")
                if retry_targets.empty:
                    break
                generate_attempt(
                    retry_targets,
                    lambda record: make_plain_rewrite_retry_prompt(
                        record["response"], record["plain_response"]
                    ),
                    attempt=attempt,
                    prompt_version="v2_strict_retry",
                )

            if model is not None:
                del model, processor
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

            from ocn.colab_utils import login_huggingface, login_wandb, make_colab_paths, save_dataframe, utc_timestamp
            from ocn.metrics import detection_summary, grouped_ocn_rates, top_patterns

            paths = make_colab_paths()
            config = json.loads((paths.project_root / "ocn_colab_config.json").read_text())
            _ = login_huggingface("HF_WRITE_ACCESS")
            EXPERIMENT_ID = "main_gemma4_qwen35"
            ANALYSIS_RUN_ID = utc_timestamp()
            MAIN_DETECTION_REPO = config.get(
                "hf_main_detection_repo",
                f"{config['hf_owner']}/ocn-empty-negations-detection-main-gemma4-qwen35",
            )
            analysis_config = {
                **config,
                "experiment_id": EXPERIMENT_ID,
                "analysis_run_id": ANALYSIS_RUN_ID,
                "source_repo": MAIN_DETECTION_REPO,
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
            df = load_dataset(MAIN_DETECTION_REPO, split="train").to_pandas()
            summary = detection_summary(df)
            model_rates = grouped_ocn_rates(df, ["model_id", "model_stage", "decoding"])
            prompt_rates = grouped_ocn_rates(df, ["category", "variant", "persona"])
            save_dataframe(model_rates, Path(config["drive_data_root"]) / "report_model_rates_main_gemma4_qwen35.csv")
            save_dataframe(prompt_rates, Path(config["drive_data_root"]) / "report_prompt_rates_main_gemma4_qwen35.csv")
            summary
            """
        ),
        code(
            r"""
            regression_df = df.copy()
            regression_df["has_ocn_int"] = regression_df["has_ocn"].astype(int)
            formula = "has_ocn_int ~ C(model_stage) + C(model_family) + C(decoding) + C(variant) + C(persona) + length_target"
            model = smf.logit(formula, data=regression_df).fit(disp=False)
            report_path = Path(config["drive_data_root"]) / "logit_model_summary_main_gemma4_qwen35.txt"
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

            fig_path = Path(config["drive_figure_root"]) / "05_analysis_dashboard_main_gemma4_qwen35.png"
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
    write_notebook("05_analysis_and_reporting.ipynb", analysis_notebook())
    write_notebook("06_optional_reward_model_scoring.ipynb", reward_scoring_notebook())
    write_notebook("07_optional_lora_style_intervention.ipynb", lora_notebook())


if __name__ == "__main__":
    main()
