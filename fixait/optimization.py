from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def _normalize_positive(
    features: Sequence[str],
    values: Mapping[str, float],
) -> np.ndarray:
    array = np.asarray(
        [max(0.0, float(values.get(feature, 0.0))) for feature in features],
        dtype=float,
    )
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum < 1e-12:
        return np.ones_like(array)
    return (array - minimum) / (maximum - minimum + 1e-12)


def optimize_fei_rank_gradient(
    global_fei: Mapping[str, float],
    drop_impacts: Mapping[str, float],
    *,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    random_state: int = 42,
    reg_lambda: float = 0.10,
    max_weight_change_pct: Optional[float] = None,
) -> dict[str, float]:
    """Regularized notebook rank-gradient FEI optimization.

    Optimization operates on normalized absolute FEI magnitudes, preserves the
    original signs and scale, and optionally projects every update into a
    feature-specific percentage range around its starting magnitude.
    """

    if max_weight_change_pct is not None and not (
        0.0 <= max_weight_change_pct <= 100.0
    ):
        raise ValueError(
            "max_weight_change_pct must be between 0 and 100 or None."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Faithfulness optimization requires PyTorch. Install the "
            "optimizer extra with: pip install 'fixait[optimizer]'"
        ) from exc

    features = [feature for feature in global_fei if feature in drop_impacts]
    if not features:
        raise ValueError("No common FEI and permutation-impact features were found.")

    signs = {
        feature: -1.0 if float(global_fei[feature]) < 0 else 1.0
        for feature in features
    }
    target = _normalize_positive(features, drop_impacts)
    original_magnitudes = np.asarray(
        [abs(float(global_fei[feature])) for feature in features],
        dtype=float,
    )
    magnitude_scale = float(np.max(original_magnitudes))
    if magnitude_scale <= 1e-12:
        return dict(global_fei)
    initial = original_magnitudes / magnitude_scale

    if len(features) < 2 or len(np.unique(target)) <= 1:
        return dict(global_fei)

    torch.manual_seed(random_state)
    rng = np.random.RandomState(random_state)
    target_tensor = torch.tensor(target, dtype=torch.float32)
    reference = torch.tensor(initial, dtype=torch.float32)
    if max_weight_change_pct is None:
        lower_bound = torch.zeros_like(reference)
        upper_bound = torch.ones_like(reference)
    else:
        change_ratio = float(max_weight_change_pct) / 100.0
        lower_bound = reference * max(0.0, 1.0 - change_ratio)
        upper_bound = reference * (1.0 + change_ratio)
    # A zero FEI must remain zero; calibration cannot invent importance.
    upper_bound = torch.where(reference == 0.0, reference, upper_bound)
    weights = torch.nn.Parameter(reference.clone())
    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    pair_count = len(features) * (len(features) - 1) // 2

    def sample_pairs():
        if pair_batch >= pair_count:
            left, right = np.triu_indices(len(features), k=1)
        else:
            left = rng.randint(0, len(features), size=pair_batch)
            right = rng.randint(0, len(features), size=pair_batch)
            mask = left != right
            left, right = left[mask], right[mask]
        return (
            torch.tensor(left, dtype=torch.long),
            torch.tensor(right, dtype=torch.long),
        )

    for _ in range(n_steps):
        optimizer.zero_grad()

        left, right = sample_pairs()
        target_difference = target_tensor[left] - target_tensor[right]
        direction = torch.sign(target_difference)
        informative = direction != 0
        if not torch.any(informative):
            break
        left = left[informative]
        right = right[informative]
        direction = direction[informative]

        logits = tau * direction * (weights[left] - weights[right])
        rank_loss = torch.nn.functional.softplus(-logits).mean()
        regularization = torch.mean((weights - reference) ** 2)
        loss = rank_loss + reg_lambda * regularization
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights.copy_(
                torch.maximum(lower_bound, torch.minimum(weights, upper_bound))
            )

    with torch.no_grad():
        values = weights.detach().cpu().numpy()
    optimized_magnitudes = values.astype(float) * magnitude_scale
    if max_weight_change_pct is not None:
        change_ratio = float(max_weight_change_pct) / 100.0
        optimized_magnitudes = np.clip(
            optimized_magnitudes,
            original_magnitudes * max(0.0, 1.0 - change_ratio),
            original_magnitudes * (1.0 + change_ratio),
        )
    optimized_magnitudes[original_magnitudes == 0.0] = 0.0

    optimized = dict(global_fei)
    for feature, value in zip(features, optimized_magnitudes):
        optimized[feature] = signs[feature] * float(value)
    return optimized


def summarize_fei_weight_change(
    original_fei: Mapping[str, float],
    optimized_fei: Mapping[str, float],
) -> dict[str, float]:
    """Summarize per-feature magnitude changes as percentages."""

    features = [feature for feature in original_fei if feature in optimized_fei]
    if not features:
        return {"mean_weight_change_pct": 0.0, "max_weight_change_pct": 0.0}
    reference = np.asarray(
        [abs(float(original_fei[feature])) for feature in features],
        dtype=float,
    )
    optimized = np.asarray(
        [abs(float(optimized_fei[feature])) for feature in features],
        dtype=float,
    )
    changes = np.zeros_like(reference)
    non_zero = reference > 0.0
    changes[non_zero] = (
        np.abs(optimized[non_zero] - reference[non_zero])
        / reference[non_zero]
        * 100.0
    )
    # Zero starting weights are expected to remain zero.  A non-zero optimized
    # value would signal a contract violation rather than a finite percentage.
    if np.any(~non_zero & (optimized > 0.0)):
        return {
            "mean_weight_change_pct": float("inf"),
            "max_weight_change_pct": float("inf"),
        }
    return {
        "mean_weight_change_pct": float(np.mean(changes)),
        "max_weight_change_pct": float(np.max(changes)),
    }
