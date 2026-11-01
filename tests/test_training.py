"""CPU integration checks using local, randomly initialized tiny models."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_prompts
from rewards import make_reward_function, reward_text


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        words = ["<pad>", "<eos>", "<unk>", "user", "assistant", ":", "hello", "world",
                 "explain", "blue", "sky", "write", "a", "note", "red", "green"]
        backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
        )
        cls.tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ' : ' + message['content'] + ' ' }}"
            "{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
        )
        model_config = GPT2Config(
            vocab_size=len(words), n_positions=64, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=1, pad_token_id=0,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )
        cls.policy_dir = cls.root / "policy"
        cls.reward_dir = cls.root / "reward"
        policy = GPT2LMHeadModel(model_config)
        cls.initial_weights = policy.transformer.wte.weight.detach().clone()
        policy.save_pretrained(cls.policy_dir)
        model_config.num_labels = 1
        GPT2ForSequenceClassification(model_config).save_pretrained(cls.reward_dir)
        cls.tokenizer.save_pretrained(cls.policy_dir)
        cls.tokenizer.save_pretrained(cls.reward_dir)
        cls.prompts = [
            [{"role": "user", "content": text}]
            for text in ["hello world", "explain blue sky", "write a note", "red green"]
        ]
        (cls.root / "prompts.jsonl").write_text(
            "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in cls.prompts)
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prompt_loading_preserves_order_and_format(self):
        dataset = Dataset.from_list([
            {"question": "hello", "answer": "world"},
            {"question": "write", "answer": "note"},
            {"question": "red", "answer": "green"},
        ])
        with patch("data.load_dataset", return_value=dataset):
            actual = load_prompts(
                {"name": "local", "prompt_column": "question"}, self.root
            )
        self.assertEqual(actual.column_names, ["prompt"])
        self.assertEqual(list(actual["prompt"]), ["hello", "write", "red"])

    def test_reward_batches_match_individual_scoring(self):
        config = {"name": str(self.reward_dir), "device": "cpu", "batch_size": 2, "max_length": 32}
        reward = make_reward_function(config, torch.float32)
        prompts = self.prompts[:3]
        completions = [[{"role": "assistant", "content": t}] for t in ["blue sky", "note", "red"]]
        batches = reward(prompts, completions)
        singles = [reward([p], [c])[0] for p, c in zip(prompts, completions)]
        torch.testing.assert_close(torch.tensor(batches), torch.tensor(singles))
        self.assertEqual(
            reward_text(self.tokenizer, "hello", "world"),
            reward_text(self.tokenizer, [{"role": "user", "content": "hello"}],
                        [{"role": "assistant", "content": "world"}]),
        )

    def test_train_save_and_resume_both_methods(self):
        environment = {
            **os.environ,
            "HF_HOME": str(self.root / "hf_cache"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        for method in ("tea", "prefix_tea"):
            with self.subTest(method=method):
                config = yaml.safe_load((ROOT / "configs" / f"{method}.yaml").read_text())
                output = self.root / method
                config["model"] = str(self.policy_dir)
                config["dataset"] = {"file": "prompts.jsonl"}
                config["reward_model"].update(name=str(self.reward_dir), device="cpu", max_length=32)
                config["training"].update(
                    output_dir=str(output), max_steps=1, num_generations=24,
                    per_device_train_batch_size=4, gradient_accumulation_steps=6,
                    max_completion_length=8,
                    learning_rate=1e-3, bf16=False, use_cpu=True,
                    save_steps=1, disable_tqdm=True, dataloader_pin_memory=False,
                )
                config_path = self.root / f"{method}.yaml"
                config_path.write_text(yaml.safe_dump(config))
                command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path)]
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                model = AutoModelForCausalLM.from_pretrained(output / "final")
                self.assertFalse(torch.equal(self.initial_weights, model.transformer.wte.weight))
                self.assertTrue(torch.isfinite(model.transformer.wte.weight).all())
                self.assertTrue((output / "final" / "tokenizer.json").exists())
                self.assertTrue((output / "train_config.yaml").exists())
                state = json.loads((output / "trainer_state.json").read_text())
                self.assertEqual(state["global_step"], 1)
                config["training"]["max_steps"] = 2
                config_path.write_text(yaml.safe_dump(config))
                resumed = subprocess.run(
                    command + ["--resume-from-checkpoint", str(output / "checkpoint-1")],
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                state = json.loads((output / "trainer_state.json").read_text())
                self.assertEqual(state["global_step"], 2)

if __name__ == "__main__":
    unittest.main()
