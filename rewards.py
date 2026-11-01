"""Batched scalar reward-model scoring for TRL."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def reward_text(tokenizer, prompt, completion):
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    messages += (
        [{"role": "assistant", "content": completion}]
        if isinstance(completion, str)
        else list(completion)
    )
    return tokenizer.apply_chat_template(messages, tokenize=False)


def make_reward_function(config: dict, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        config["name"], torch_dtype=dtype
    ).to(config["device"])
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.requires_grad_(False)
    batch_size = config.get("batch_size", 4)
    max_length = config.get("max_length", 512)

    @torch.no_grad()
    def reward_model(prompts, completions, **kwargs):
        texts = [
            reward_text(tokenizer, prompt, completion)
            for prompt, completion in zip(prompts, completions, strict=True)
        ]
        scores = []
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(model.device)
            scores.extend(model(**inputs).logits.squeeze(-1).float().cpu().tolist())
        return scores

    return reward_model
