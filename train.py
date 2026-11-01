"""Train a language model with TEA or Prefix-TEA."""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig

from data import load_prompts
from rewards import make_reward_function
from trainer import TEATrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tea.yaml")
    parser.add_argument("--output-dir", help="Override the output directory in the YAML file")
    parser.add_argument("--resume-from-checkpoint", help="A checkpoint directory saved by TRL")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    training_args = GRPOConfig(**config["training"])
    set_seed(training_args.seed)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    dataset = load_prompts(config["dataset"], config_path.parent)
    reward = make_reward_function(config["reward_model"], dtype)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = TEATrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=dataset,
        args=training_args,
        **config["algorithm"],
    )
    output_dir = Path(training_args.output_dir)
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
