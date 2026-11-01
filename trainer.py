"""TEA advantages in the existing TRL generation, KL, and optimization loop."""

import torch
from trl import GRPOTrainer

from tea_advantages import prefix_tea_advantages, tea_advantages


class TEATrainer(GRPOTrainer):
    def __init__(self, *args, method="tea", alpha=0.25, n_target=128, **kwargs):
        self.advantage_fn = {
            "tea": tea_advantages,
            "prefix_tea": prefix_tea_advantages,
        }[method]
        self.alpha = alpha
        self.n_target = n_target
        self._group_rewards = None
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        # TRL gathers rewards across processes before promptwise grouping.
        self._group_rewards = rewards
        return rewards

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        rewards = self._group_rewards
        self._group_rewards = None
        weights = self.reward_weights.to(rewards.device)
        grouped = (rewards * weights).nansum(dim=1).view(
            -1, self.num_generations if self.model.training else self.num_generations_eval
        )
        with torch.no_grad():
            advantages = self.advantage_fn(
                grouped, alpha=self.alpha, n_target=self.n_target
            ).reshape(-1)
        local_count = output["advantages"].shape[0]
        start = self.accelerator.process_index * local_count
        output["advantages"] = advantages[start : start + local_count].to(output["advantages"])
        return output
