"""Run fixed-sequence soft-target distillation through Tinker."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from tinker_cookbook.distillation import train_off_policy as recipe
from tinker_cookbook.distillation.datasets import TeacherConfig
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import require_tinker_key


@dataclass(frozen=True)
class OffPolicyConfig:
    conversation_file: str
    log_path: str
    model_name: str = "Qwen/Qwen3.5-4B"
    teacher_model: str = "Qwen/Qwen3.5-9B"
    renderer_name: str = "qwen3_5_disable_thinking"
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_length: int = 1024
    lora_rank: int = 32
    n_teacher_targets: int = 20
    teacher_concurrency: int = 16
    save_every: int = 10
    eval_every: int = 10
    max_steps: int | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    teacher_checkpoint: str | None = None


async def train_off_policy(config: OffPolicyConfig) -> None:
    require_tinker_key()
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=config.renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common,
        file_path=config.conversation_file,
        test_size=0,
        shuffle_seed=42,
    )
    dataset_with_teacher = recipe.DatasetWithTeacher(
        dataset_builder=dataset_builder,
        teacher_config=TeacherConfig(
            base_model=config.teacher_model,
            load_checkpoint_path=config.teacher_checkpoint,
        ),
    )
    run_config = recipe.Config(
        learning_rate=config.learning_rate,
        dataset_configs=[dataset_with_teacher],
        model_name=config.model_name,
        recipe_name="calibrate_qwen_off_policy",
        renderer_name=config.renderer_name,
        lora_rank=config.lora_rank,
        n_teacher_targets=config.n_teacher_targets,
        teacher_concurrency=config.teacher_concurrency,
        batch_size=config.batch_size,
        save_every=config.save_every,
        eval_every=config.eval_every,
        max_steps=config.max_steps,
        load_checkpoint_path=config.load_checkpoint_path,
        log_path=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )
    await recipe.main(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-file", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--renderer-name", default="qwen3_5_disable_thinking")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--n-teacher-targets", type=int, default=20)
    parser.add_argument("--teacher-concurrency", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    args = parser.parse_args()
    asyncio.run(train_off_policy(OffPolicyConfig(**vars(args))))


if __name__ == "__main__":
    main()
