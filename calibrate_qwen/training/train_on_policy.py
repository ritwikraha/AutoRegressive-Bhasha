"""Run on-policy KL distillation on CalibrateQwen prompts through Tinker."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chz
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.distillation import train_on_policy as recipe
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import build_system_prompt, require_tinker_key


@chz.chz
class MCQPromptDatasetBuilder(RLDatasetBuilder):
    repo_id: str
    config_name: str
    split: str
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    max_examples: int | None = None
    max_prompt_tokens: int | None = 1024

    async def __call__(self) -> tuple[PromptOnlyDataset, None]:
        dataset = load_dataset(
            self.repo_id,
            self.config_name,
            split=self.split,
            token=os.environ.get("HF_TOKEN"),
        )
        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))
        prompts = [str(prompt) for prompt in dataset["prompt"]]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_dataset = PromptOnlyDataset(
            prompts=prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            convo_prefix=[{"role": "system", "content": build_system_prompt("numeric")}],
            dataset_name="calibrate_qwen",
        )
        return train_dataset, None


@dataclass(frozen=True)
class OnPolicyConfig:
    log_path: str
    repo_id: str = "ritwikraha/calibrate-qwen-curated"
    config_name: str = "calibrated_mcq"
    split: str = "train"
    model_name: str = "Qwen/Qwen3.5-4B"
    teacher_model: str = "Qwen/Qwen3.5-9B"
    renderer_name: str = "qwen3_5_disable_thinking"
    learning_rate: float = 1e-4
    groups_per_batch: int = 16
    group_size: int = 2
    max_examples: int = 4000
    max_prompt_tokens: int = 1024
    max_tokens: int = 192
    temperature: float = 1.0
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    num_substeps: int = 1
    save_every: int = 10
    eval_every: int = 10
    max_steps: int | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    teacher_checkpoint: str | None = None


async def train_on_policy(config: OnPolicyConfig) -> None:
    require_tinker_key()
    dataset_builder = MCQPromptDatasetBuilder(
        repo_id=config.repo_id,
        config_name=config.config_name,
        split=config.split,
        groups_per_batch=config.groups_per_batch,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=config.renderer_name,
        max_examples=config.max_examples,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=TeacherConfig(
            base_model=config.teacher_model,
            load_checkpoint_path=config.teacher_checkpoint,
        ),
        groups_per_batch=config.groups_per_batch,
    )
    run_config = recipe.Config(
        learning_rate=config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=config.model_name,
        recipe_name="calibrate_qwen_on_policy",
        renderer_name=config.renderer_name,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        lora_rank=config.lora_rank,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
        num_substeps=config.num_substeps,
        log_path=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        eval_every=config.eval_every,
        save_every=config.save_every,
        load_checkpoint_path=config.load_checkpoint_path,
        max_steps=config.max_steps,
    )
    await recipe.main(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--repo-id", default="ritwikraha/calibrate-qwen-curated")
    parser.add_argument("--config-name", default="calibrated_mcq")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--renderer-name", default="qwen3_5_disable_thinking")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--groups-per-batch", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--kl-discount-factor", type=float, default=0.0)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    args = parser.parse_args()
    asyncio.run(train_on_policy(OnPolicyConfig(**vars(args))))


if __name__ == "__main__":
    main()
