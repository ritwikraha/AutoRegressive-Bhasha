from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColabPaths:
    drive_root: Path
    project_root: Path
    artifact_root: Path
    data_root: Path
    figure_root: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if running_in_colab():
        try:
            from google.colab import userdata  # type: ignore

            value = userdata.get(name)
            if value:
                os.environ[name] = value
                return value
        except Exception:
            pass
    return default


def mount_drive() -> Path:
    if running_in_colab():
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            return drive_root

        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return drive_root
    return Path.cwd()


def make_colab_paths(project_name: str = "ocn_empty_negations") -> ColabPaths:
    drive_root = mount_drive()
    project_root = drive_root / project_name
    artifact_root = project_root / "artifacts"
    data_root = artifact_root / "data"
    figure_root = artifact_root / "figures"
    for path in [project_root, artifact_root, data_root, figure_root]:
        path.mkdir(parents=True, exist_ok=True)
    return ColabPaths(
        drive_root=drive_root,
        project_root=project_root,
        artifact_root=artifact_root,
        data_root=data_root,
        figure_root=figure_root,
    )


def login_huggingface(secret_name: str = "HF_WRITE_ACCESS") -> str:
    token = get_secret(secret_name)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token. In Colab, add a secret named {secret_name}."
        )
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    return token


def login_wandb(project: str, secret_name: str = "WANDB_KEY", **init_kwargs: Any):
    token = get_secret(secret_name)
    if not token:
        raise RuntimeError(f"Missing W&B key. In Colab, add a secret named {secret_name}.")
    import wandb

    wandb.login(key=token)
    return wandb.init(project=project, **init_kwargs)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def publish_dataframe_to_hf(
    df: pd.DataFrame,
    repo_id: str,
    split: str = "train",
    private: bool = False,
    card_path: str | Path | None = None,
    commit_message: str | None = None,
) -> str:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset_dict = DatasetDict({split: dataset})
    dataset_dict.push_to_hub(repo_id, private=private, commit_message=commit_message)
    if card_path is not None:
        HfApi().upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    return f"https://huggingface.co/datasets/{repo_id}"
