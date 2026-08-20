from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def resolve_baseline(
    X_reference: pd.DataFrame,
    features: Iterable[str],
    *,
    method: str = "median",
    categorical_features: Optional[Iterable[str]] = None,
) -> pd.Series:
    if method not in {"median", "mean", "zero"}:
        raise ValueError("baseline method must be 'median', 'mean', or 'zero'.")

    categorical = set(categorical_features or [])
    values = {}
    for feature in features:
        if feature not in X_reference.columns:
            raise ValueError(f"Feature {feature!r} is missing from X_reference.")
        column = X_reference[feature]
        is_categorical = (
            feature in categorical
            or isinstance(column.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(column.dtype)
            or pd.api.types.is_bool_dtype(column.dtype)
        )
        if is_categorical:
            mode = column.mode(dropna=True)
            if mode.empty:
                raise ValueError(f"Could not determine a categorical baseline for {feature!r}.")
            values[feature] = mode.iloc[0]
        elif method == "median":
            values[feature] = column.median()
        elif method == "mean":
            values[feature] = column.mean()
        else:
            values[feature] = 0.0
    return pd.Series(values)

