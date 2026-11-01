"""CPU sanity checks for the minimal TEA package."""

from __future__ import annotations

import math
import os
import sys

import torch

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from tea_advantages import (  # noqa: E402
    calculate_c_tilde,
    policy_gradient_loss,
    prefix_lengths_for_prefix_tea,
    prefix_tea_advantages,
    prefix_tea_scores,
    prefix_weights_for_prefix_tea,
    tea_advantages,
    tea_scores,
)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"expected {expected}, got {actual}")


def test_tea_shapes_centering_and_nonnegative_scores() -> None:
    rewards = torch.tensor(
        [
            [0.1, 0.5, 1.2, 0.8, 1.8, 2.5],
            [2.0, 0.2, 0.3, 1.9, 1.1, 3.2],
        ],
        dtype=torch.float32,
    )
    scores = tea_scores(rewards, alpha=0.5, n_target=128)
    advantages = tea_advantages(rewards, alpha=0.5, n_target=128)

    if scores.shape != rewards.shape or advantages.shape != rewards.shape:
        raise AssertionError("TEA outputs must match reward shape")
    if not torch.isfinite(scores).all() or not torch.isfinite(advantages).all():
        raise AssertionError("TEA outputs must be finite")
    if scores.min().item() < -1e-6:
        raise AssertionError("TEA raw scores should be nonnegative")
    assert_close(advantages.mean(dim=1), torch.zeros(rewards.shape[0]))


def test_prefix_tea_prefixes_weights_and_centering() -> None:
    rewards = torch.arange(1, 129, dtype=torch.float32).view(2, 64)
    prefixes = prefix_lengths_for_prefix_tea(m=64, alpha=0.25)
    weights = prefix_weights_for_prefix_tea(prefixes, m=64)
    z_values = [64 / prefix_len for prefix_len in prefixes]

    if prefixes != [40, 48, 56, 64]:
        raise AssertionError(f"unexpected Prefix-TEA prefixes: {prefixes}")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise AssertionError(f"Prefix-TEA weights should sum to one: {weights}")
    if abs(sum(weight * z for weight, z in zip(weights, z_values))) > 1e-6:
        raise AssertionError(f"Prefix-TEA weights should cancel first z moment: {weights}")

    scores = prefix_tea_scores(rewards, alpha=0.25, n_target=128)
    advantages = prefix_tea_advantages(rewards, alpha=0.25, n_target=128)
    if scores.shape != rewards.shape or advantages.shape != rewards.shape:
        raise AssertionError("Prefix-TEA outputs must match reward shape")
    if not torch.isfinite(scores).all() or not torch.isfinite(advantages).all():
        raise AssertionError("Prefix-TEA outputs must be finite")
    assert_close(advantages.mean(dim=1), torch.zeros(rewards.shape[0]), tol=1e-5)


def test_loss_has_gradient_flow() -> None:
    torch.manual_seed(1)
    rewards = torch.randn(3, 64)
    sequence_logprobs = torch.randn(3, 64, requires_grad=True)

    advantages = tea_advantages(rewards, alpha=0.25, n_target=128)
    loss = policy_gradient_loss(sequence_logprobs, advantages)
    loss.backward()
    if sequence_logprobs.grad is None:
        raise AssertionError("loss should produce gradients for sequence_logprobs")
    if not torch.isfinite(sequence_logprobs.grad).all():
        raise AssertionError("sequence_logprobs gradients must be finite")


def test_c_tilde_is_finite() -> None:
    value = calculate_c_tilde(alpha=0.25, n_target=128)
    if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
        raise AssertionError("c_tilde should be a finite float")


def test_positive_part_and_alpha_scale() -> None:
    # For these rewards and N=128, the complete raw scores at 5 and 6 are
    # negative; only reward 7 survives, with (7 - 4) / 0.5 = 6.
    rewards = torch.arange(8, dtype=torch.float64).reshape(1, -1)
    expected = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 6.]], dtype=torch.float64)
    assert_close(tea_scores(rewards, alpha=0.5, n_target=128), expected)


def test_prefix_normalization() -> None:
    # A single outlier gives an analytic score for every zero-padded prefix.
    m, alpha = 64, 0.25
    rewards = torch.zeros(1, m, dtype=torch.float64)
    rewards[0, 0] = 1.0
    prefixes = prefix_lengths_for_prefix_tea(m, alpha)
    weights = prefix_weights_for_prefix_tea(prefixes, m)
    c = calculate_c_tilde(alpha, 128)
    expected = 0.0
    for length, weight in zip(prefixes, weights):
        k = math.ceil(alpha * length)
        outlier_score = (1 + c * (k - 2) / (2 * math.sqrt(k - 1))) / alpha
        expected += weight * (m / length) * outlier_score
    actual = prefix_tea_scores(rewards, alpha=alpha)
    assert_close(actual[:, 0], torch.tensor([expected], dtype=torch.float64))
    assert_close(actual[:, 1:], torch.zeros(1, m - 1, dtype=torch.float64))


def test_gaussian_maximum_constant() -> None:
    # E[max(Z1, Z2)] = 1 / sqrt(pi), rather than the median quantile 0.
    alpha = 0.5
    tail_mean = math.sqrt(2 / math.pi)
    tail_std = math.sqrt(1 - 2 / math.pi)
    recovered = calculate_c_tilde(alpha, 2) * tail_std + tail_mean
    if not math.isclose(recovered, 1 / math.sqrt(math.pi), abs_tol=1e-10):
        raise AssertionError(f"Incorrect Gaussian maximum expectation: {recovered}")


def main() -> None:
    test_tea_shapes_centering_and_nonnegative_scores()
    test_prefix_tea_prefixes_weights_and_centering()
    test_loss_has_gradient_flow()
    test_c_tilde_is_finite()
    test_positive_part_and_alpha_scale()
    test_prefix_normalization()
    test_gaussian_maximum_constant()
    print("All TEA / Prefix-TEA checks passed.")


if __name__ == "__main__":
    main()

