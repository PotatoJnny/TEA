"""Load prompt-only Hugging Face datasets or a local JSONL file."""

from pathlib import Path

from datasets import load_dataset


def load_prompts(config: dict, config_dir: Path):
    if config.get("file"):
        dataset = load_dataset(
            "json", data_files=str(config_dir / config["file"]), split="train"
        )
    else:
        dataset = load_dataset(
            config["name"], config.get("subset"), split=config.get("split", "train")
        )

    column = config.get("prompt_column", "prompt")
    if column not in dataset.column_names:
        raise ValueError(f"Prompt column {column!r} is missing from {dataset.column_names}")
    if column != "prompt":
        dataset = dataset.rename_column(column, "prompt")
    return dataset.select_columns(["prompt"])
