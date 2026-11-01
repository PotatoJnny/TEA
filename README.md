# TEA / Prefix-TEA

Train language models with **TEA (Tail-Extrapolated Advantage)** using PyTorch and Hugging Face TRL.

TEA samples several responses to each prompt, scores them with a reward model, and uses upper-tail reward statistics to construct the policy-training advantages. Prefix-TEA combines estimates from four nested rollout prefixes.

Introduced in **[What should post-training optimize? A test-time scaling law perspective](https://arxiv.org/abs/2605.10716)** — NeurIPS 2026.  
[Muheng Li](https://potatojnny.github.io/), Jian Qian, Wenlong Mou.

## Install

Use Python 3.11 and a CUDA-enabled PyTorch installation.

```bash
git clone https://github.com/PotatoJnny/TEA.git
cd TEA
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The example configuration uses `meta-llama/Llama-3.2-1B-Instruct`. Obtain access to that model on Hugging Face and authenticate before downloading it:

```bash
hf auth login
```

## Train a model

```bash
# TEA
python train.py --config configs/tea.yaml

# Prefix-TEA
python train.py --config configs/prefix_tea.yaml
```

Edit the YAML file to choose your policy, reward model, dataset, and training budget. The supplied configurations are starting examples using Llama-3.2-1B-Instruct, Skywork-Reward-V2-Llama-3.1-8B, and UltraFeedback prompts.

The training loop loads prompts, generates responses, scores them, computes TEA advantages, and updates the policy with a KL penalty. The example uses one training process with bfloat16 and gradient checkpointing. Set `reward_model.device: cuda:1` to place the reward model on a second GPU.

The main settings are:

- `model`: a Hugging Face causal language model or local model directory.
- `reward_model.name`: a sequence-classification model that returns one scalar reward.
- `algorithm.method`: `tea` or `prefix_tea`.
- `algorithm.alpha`: the fraction of rewards used to estimate the upper tail.
- `algorithm.n_target`: the best-of-N budget that controls tail extrapolation.
- `training.num_generations`: the number of sampled responses per prompt.
- `training.max_steps`, `learning_rate`, and `beta`: training duration, learning rate, and KL coefficient.

On one training process, `per_device_train_batch_size * gradient_accumulation_steps` must be divisible by `num_generations`. The example uses `4 * 16 = 64`. Prefix-TEA with `alpha=0.25` needs at least 24 responses per group.

## Use your own data

For a Hugging Face dataset, configure its name, split, and prompt column:

```yaml
dataset:
  name: trl-lib/ultrafeedback-prompt
  split: train
  prompt_column: prompt
```

Add `subset` if the dataset has a named configuration.

For a local JSONL file, create one prompt per line:

```json
{"prompt": [{"role": "user", "content": "Explain why the sky is blue."}]}
{"prompt": [{"role": "user", "content": "Write a short thank-you note."}]}
```

Then replace the `dataset` section with:

```yaml
dataset:
  file: ../my_prompts.jsonl
  prompt_column: prompt
```

Local data paths are relative to the configuration file. Plain text prompts are also supported; message lists use the policy's chat template.

## Save, resume, and use the model

The final model and tokenizer are saved to `outputs/tea/final/`, or `outputs/prefix_tea/final/` for Prefix-TEA. Change the destination with `--output-dir`.

Resumable checkpoints are stored under `checkpoint-*/`. To continue training, set the desired total `max_steps` in the YAML file and run:

```bash
python train.py --config configs/tea.yaml \
  --resume-from-checkpoint outputs/tea/checkpoint-50
```

Load the saved model with Transformers:

```python
from transformers import pipeline

generate = pipeline("text-generation", model="outputs/tea/final", device_map="auto")
response = generate(
    [{"role": "user", "content": "Explain why the sky is blue."}],
    max_new_tokens=128,
    do_sample=True,
    temperature=0.8,
)
print(response[0]["generated_text"][-1]["content"])
```

## Code structure

| File | Purpose |
| --- | --- |
| [tea_advantages.py](tea_advantages.py) | TEA and Prefix-TEA advantage computation |
| [trainer.py](trainer.py) | Apply the advantages in TRL's policy-training loop |
| [rewards.py](rewards.py) | Batch prompt-response scoring with a reward model |
| [data.py](data.py) | Load prompts from a dataset or JSONL file |
| [train.py](train.py) | Load models, train, resume, and save |
| [configs/](configs/) | Editable training examples |

The advantage functions accept rewards of shape `[batch_size, num_generations]` and return coefficients with the same shape, so they can also be used in another policy-gradient trainer:

```python
from tea_advantages import tea_advantages, prefix_tea_advantages

advantages = tea_advantages(rewards, alpha=0.25, n_target=128)
# Or:
advantages = prefix_tea_advantages(rewards, alpha=0.25, n_target=128)
```

## Citation

```bibtex
@inproceedings{li2026posttraining,
  title     = {What should post-training optimize? A test-time scaling law perspective},
  author    = {Li, Muheng and Qian, Jian and Mou, Wenlong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.10716}
}
```

## License

[MIT](LICENSE). Pretrained models and datasets retain their own licenses.
