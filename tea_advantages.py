"""TEA and Prefix-TEA advantage functions.

The functions in this file operate on grouped rewards with shape
``[batch_size, m]`` and return policy-gradient coefficients with the same
shape. They are framework-agnostic: the returned coefficients can be plugged
into any grouped policy-gradient implementation that uses sequence log
probabilities.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

from scipy.integrate import quad

import torch


def _standard_normal_icdf(prob: float) -> float:
    p = torch.tensor(float(prob), dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(p))


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@lru_cache(maxsize=None)
def calculate_c_tilde(alpha: float, n_target: int) -> float:
    """Compute the Gaussian-tail scaling constant used by TEA.

    Args:
        alpha: Upper-tail fraction, with ``0 < alpha < 1``.
        n_target: Target Best-of-N value.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")
    if n_target <= 1:
        raise ValueError(f"n_target must be > 1, got {n_target}")

    # E[max(Z_1, ..., Z_N)] for independent standard Gaussian variables.
    c_n = quad(
        lambda z: z * n_target * _standard_normal_pdf(z)
        * (0.5 * math.erfc(-z / math.sqrt(2.0))) ** (n_target - 1),
        -math.inf, math.inf, epsabs=1e-10,
    )[0]
    z_alpha = _standard_normal_icdf(1.0 - alpha)
    lambda_z = _standard_normal_pdf(z_alpha) / alpha
    delta_z = math.sqrt(1.0 + z_alpha * lambda_z - lambda_z * lambda_z)
    return float((c_n - lambda_z) / delta_z)


def tea_scores(
    rewards: torch.Tensor,
    *,
    alpha: float = 0.25,
    n_target: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return nonnegative upper-tail TEA scores before group centering."""
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [batch_size, m], got {tuple(rewards.shape)}")

    _, m = rewards.shape
    k = max(1, math.ceil(m * alpha))
    c_tilde = calculate_c_tilde(alpha, n_target)

    top_values, top_indices = torch.topk(rewards, k, dim=1)
    threshold = top_values[:, -1:]
    tail_mean = top_values.mean(dim=1, keepdim=True)
    tail_std = top_values.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)

    linear = top_values - threshold
    quadratic = (top_values - tail_mean).square() - (threshold - tail_mean).square()
    raw_top = (linear + (c_tilde / (2.0 * tail_std)) * quadratic) / alpha
    transformed_top = raw_top.clamp_min(0.0)

    scores = torch.zeros_like(rewards)
    scores.scatter_(1, top_indices, transformed_top)
    return scores


def tea_advantages(
    rewards: torch.Tensor,
    *,
    alpha: float = 0.25,
    n_target: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return TEA coefficients centered within each prompt group."""
    scores = tea_scores(rewards, alpha=alpha, n_target=n_target, eps=eps)
    return scores - scores.mean(dim=1, keepdim=True)


def _alpha_fraction(alpha: float) -> Fraction:
    alpha_fraction = Fraction(str(alpha)).limit_denominator(10_000)
    if not (0 < alpha_fraction < 1):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")
    return alpha_fraction


def prefix_lengths_for_prefix_tea(m: int, alpha: float = 0.25) -> list[int]:
    """Return the four prefix lengths used by Prefix-TEA.

    This uses the order-2, four-prefix construction.
    For ``m=64`` and ``alpha=0.25`` it returns
    ``[40, 48, 56, 64]``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    alpha_fraction = _alpha_fraction(alpha)
    q = alpha_fraction.denominator
    lower_bound = math.ceil(m / 2)
    step = (m - lower_bound) // (3 * q)
    if step < 1:
        raise ValueError("m is too small for Prefix-TEA compatible prefixes")

    prefixes = [m - 3 * step * q, m - 2 * step * q, m - step * q, m]
    if any(prefix_len < lower_bound for prefix_len in prefixes):
        raise ValueError(f"prefixes dropped below ceil(m/2): {prefixes}")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"prefixes must be distinct, got {prefixes}")
    if len({prefix_len % q for prefix_len in prefixes}) != 1:
        raise ValueError(f"prefixes must be congruent modulo denominator {q}, got {prefixes}")
    return prefixes


def prefix_weights_for_prefix_tea(prefixes: list[int], m: int) -> list[float]:
    """Return four Prefix-TEA extrapolation weights."""
    if len(prefixes) != 4:
        raise ValueError(f"Prefix-TEA expects exactly four prefixes, got {prefixes}")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"prefixes must be distinct, got {prefixes}")

    z_values = torch.tensor([m / prefix_len for prefix_len in prefixes], dtype=torch.float64)
    z_bar = z_values.mean()
    centered = z_values - z_bar
    s2 = centered.square().sum()
    if float(s2) <= 0.0:
        raise ValueError(f"Prefix-TEA weights are undefined for prefixes {prefixes}")

    weights = 0.25 - (z_bar / s2) * centered
    return [float(weight) for weight in weights]


def _prefix_tea_scores_for_prefix(
    rewards: torch.Tensor,
    *,
    prefix_len: int,
    alpha: float,
    n_target: int,
    eps: float,
) -> torch.Tensor:
    batch_size, m = rewards.shape
    if prefix_len < 1 or prefix_len > m:
        raise ValueError(f"prefix_len must satisfy 1 <= prefix_len <= {m}, got {prefix_len}")

    c_tilde = calculate_c_tilde(alpha, n_target)
    prefix_rewards = rewards[:, :prefix_len]
    k = max(1, math.ceil(alpha * prefix_len))
    top_values, _ = torch.topk(prefix_rewards, k, dim=1)

    threshold = top_values[:, -1:]
    tail_mean = top_values.mean(dim=1, keepdim=True)
    tail_std = top_values.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)

    linear = prefix_rewards - threshold
    quadratic = (prefix_rewards - tail_mean).square() - (threshold - tail_mean).square()
    raw_scores = (linear + (c_tilde / (2.0 * tail_std)) * quadratic) / alpha
    tail_mask = prefix_rewards >= threshold
    prefix_scores = tail_mask.to(prefix_rewards.dtype) * raw_scores.clamp_min(0.0)

    full_scores = torch.zeros((batch_size, m), dtype=rewards.dtype, device=rewards.device)
    full_scores[:, :prefix_len] = prefix_scores
    return full_scores


def prefix_tea_scores(
    rewards: torch.Tensor,
    *,
    alpha: float = 0.25,
    n_target: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return Prefix-TEA scores before group centering."""
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [batch_size, m], got {tuple(rewards.shape)}")

    _, m = rewards.shape
    prefixes = prefix_lengths_for_prefix_tea(m, alpha)
    weights = prefix_weights_for_prefix_tea(prefixes, m)
    scores = torch.zeros_like(rewards)
    for prefix_len, weight in zip(prefixes, weights):
        scores = scores + weight * (m / prefix_len) * _prefix_tea_scores_for_prefix(
            rewards,
            prefix_len=prefix_len,
            alpha=alpha,
            n_target=n_target,
            eps=eps,
        )
    return scores


def prefix_tea_advantages(
    rewards: torch.Tensor,
    *,
    alpha: float = 0.25,
    n_target: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return Prefix-TEA coefficients centered within each prompt group."""
    scores = prefix_tea_scores(rewards, alpha=alpha, n_target=n_target, eps=eps)
    return scores - scores.mean(dim=1, keepdim=True)


def policy_gradient_loss(
    sequence_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    normalizer: str = "num_samples",
) -> torch.Tensor:
    """Minimal grouped policy-gradient loss.

    Args:
        sequence_logprobs: Sequence log probabilities with shape
            ``[batch_size, m]``.
        advantages: Stop-gradient coefficients with the same shape.
        normalizer: ``"num_samples"`` divides by all sampled responses, while
            ``"batch"`` divides only by the number of prompt groups.
    """
    if sequence_logprobs.shape != advantages.shape:
        raise ValueError(
            "sequence_logprobs and advantages must have the same shape, "
            f"got {tuple(sequence_logprobs.shape)} and {tuple(advantages.shape)}"
        )
    weighted = -(advantages.detach() * sequence_logprobs).sum()
    if normalizer == "num_samples":
        return weighted / sequence_logprobs.numel()
    if normalizer == "batch":
        return weighted / sequence_logprobs.shape[0]
    raise ValueError(f"unknown normalizer: {normalizer}")

