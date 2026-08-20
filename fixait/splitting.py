from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def normalize_split_indices(
    n_rows: int,
    split_indices: Mapping[str, Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize caller-supplied train/validation/test row indices.

    The three partitions must be disjoint and together cover every input row.
    ``validation`` may be omitted (or supplied as the legacy alias ``opt``).
    """

    if not isinstance(split_indices, Mapping):
        raise TypeError("split_indices must be a mapping of partition names to row indices.")
    if "train" not in split_indices or "test" not in split_indices:
        raise ValueError("split_indices must contain 'train' and 'test' partitions.")
    if "validation" in split_indices and "opt" in split_indices:
        raise ValueError("Pass either 'validation' or 'opt', not both.")

    def as_indices(name: str, values: Sequence[int]) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError(f"split_indices[{name!r}] must be one-dimensional.")
        if raw.size and not np.issubdtype(raw.dtype, np.integer):
            try:
                numeric = raw.astype(float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"split_indices[{name!r}] must contain integer row positions."
                ) from exc
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
                raise TypeError(
                    f"split_indices[{name!r}] must contain integer row positions."
                )
        result = raw.astype(int, copy=True)
        if result.size and (result.min() < 0 or result.max() >= n_rows):
            raise ValueError(
                f"split_indices[{name!r}] contains positions outside [0, {n_rows - 1}]."
            )
        if np.unique(result).size != result.size:
            raise ValueError(f"split_indices[{name!r}] contains duplicate positions.")
        return result

    validation_values = split_indices.get("validation", split_indices.get("opt", ()))
    train = as_indices("train", split_indices["train"])
    validation = as_indices("validation", validation_values)
    test = as_indices("test", split_indices["test"])
    if train.size == 0 or test.size == 0:
        raise ValueError("The train and test partitions must both be non-empty.")
    combined = np.concatenate([train, validation, test])
    if np.unique(combined).size != combined.size:
        raise ValueError("The supplied train, validation, and test partitions overlap.")
    expected = np.arange(n_rows)
    if combined.size != n_rows or not np.array_equal(np.sort(combined), expected):
        missing = np.setdiff1d(expected, combined)
        raise ValueError(
            "The supplied partitions must cover every input row exactly once; "
            f"missing {missing[:10].tolist()}{'...' if len(missing) > 10 else ''}."
        )
    return train, validation, test


def make_split_indices(
    y: np.ndarray,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the deterministic train/validation/test indices used by fiXAIt."""

    y = np.asarray(y)
    indices = np.arange(len(y))
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    stratification = y if stratify and len(np.unique(y)) > 1 else None
    train_full, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification,
    )

    if validation_size <= 0.0:
        return (
            np.asarray(train_full),
            np.empty(0, dtype=int),
            np.asarray(test),
        )
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1 when enabled.")
    if test_size + validation_size >= 1.0:
        raise ValueError("test_size + validation_size must be less than 1.")

    validation_relative = validation_size / max(1e-12, 1.0 - test_size)
    validation_relative = float(np.clip(validation_relative, 1e-6, 0.999999))
    second_stratification = y[train_full] if stratification is not None else None
    train, validation = train_test_split(
        np.asarray(train_full),
        test_size=validation_relative,
        random_state=random_state + 1,
        stratify=second_stratification,
    )
    return np.asarray(train), np.asarray(validation), np.asarray(test)
