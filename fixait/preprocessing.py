from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _sorted_categories(values: Sequence[Any]) -> list[Any]:
    unique = list(pd.unique(pd.Series(values).dropna()))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda value: (type(value).__name__, repr(value)))


@dataclass(frozen=True)
class PreprocessingSummary:
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    ordinal_features: tuple[str, ...]
    imputation_values: Mapping[str, Any]
    categories: Mapping[str, tuple[Any, ...]]


class TabularPreprocessor:
    """Train-only tabular preprocessing that preserves one column per feature."""

    def __init__(
        self,
        *,
        categorical_features: Optional[Iterable[str]] = None,
        ordinal_features: Optional[Iterable[str]] = None,
        ordinal_categories: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> None:
        self.declared_categorical = set(categorical_features or [])
        self.declared_ordinal = set(ordinal_features or [])
        self.ordinal_categories = {
            feature: list(categories)
            for feature, categories in (ordinal_categories or {}).items()
        }
        self.feature_names_: list[str] = []
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.ordinal_features_: list[str] = []
        self.imputation_values_: dict[str, Any] = {}
        self.categories_: dict[str, list[Any]] = {}
        self.category_maps_: dict[str, dict[Any, float]] = {}
        self.fitted_ = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("The preprocessing training data cannot be empty.")
        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate feature names.")

        columns = set(X.columns)
        unknown = sorted((self.declared_categorical | self.declared_ordinal) - columns)
        if unknown:
            raise ValueError(f"Declared feature types contain unknown columns: {unknown}")
        overlap = sorted(self.declared_categorical & self.declared_ordinal)
        if overlap:
            raise ValueError(f"Features cannot be both categorical and ordinal: {overlap}")
        unknown_orders = sorted(set(self.ordinal_categories) - self.declared_ordinal)
        if unknown_orders:
            raise ValueError(
                "ordinal_categories keys must also be listed in ordinal_features: "
                f"{unknown_orders}"
            )

        inferred_categorical = {
            column
            for column in X.columns
            if (
                isinstance(X[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(X[column].dtype)
                or pd.api.types.is_string_dtype(X[column].dtype)
                or pd.api.types.is_bool_dtype(X[column].dtype)
            )
        }
        categorical = (self.declared_categorical | inferred_categorical) - self.declared_ordinal
        ordinal = set(self.declared_ordinal)

        self.feature_names_ = list(X.columns)
        self.categorical_features_ = [c for c in X.columns if c in categorical]
        self.ordinal_features_ = [c for c in X.columns if c in ordinal]
        self.numeric_features_ = [
            c for c in X.columns if c not in categorical and c not in ordinal
        ]

        for feature in self.numeric_features_:
            numeric = pd.to_numeric(X[feature], errors="coerce")
            if numeric.notna().sum() == 0:
                raise ValueError(
                    f"Numeric feature {feature!r} contains no usable numeric values."
                )
            self.imputation_values_[feature] = float(numeric.median())

        for feature in self.categorical_features_ + self.ordinal_features_:
            series = X[feature]
            mode = series.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "__fixait_missing__"
            self.imputation_values_[feature] = fill_value
            filled = series.where(series.notna(), fill_value)
            if feature in self.ordinal_categories:
                categories = list(self.ordinal_categories[feature])
                observed = set(pd.unique(filled))
                missing_from_order = observed - set(categories)
                if missing_from_order:
                    raise ValueError(
                        f"ordinal_categories[{feature!r}] is missing observed values: "
                        f"{sorted(missing_from_order, key=repr)}"
                    )
            else:
                categories = _sorted_categories(filled.tolist())
            if not categories:
                raise ValueError(f"Could not derive categories for {feature!r}.")
            self.categories_[feature] = categories
            self.category_maps_[feature] = {
                category: float(index) for index, category in enumerate(categories)
            }

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        missing = [feature for feature in self.feature_names_ if feature not in X.columns]
        if missing:
            raise ValueError(f"X is missing fitted features: {missing}")

        transformed = pd.DataFrame(index=X.index)
        for feature in self.feature_names_:
            series = X[feature]
            fill_value = self.imputation_values_[feature]
            if feature in self.numeric_features_:
                transformed[feature] = (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(fill_value)
                    .astype(float)
                )
            else:
                filled = series.where(series.notna(), fill_value)
                transformed[feature] = (
                    filled.map(self.category_maps_[feature]).fillna(-1.0).astype(float)
                )
        return transformed[self.feature_names_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def summary(self) -> PreprocessingSummary:
        if not self.fitted_:
            raise RuntimeError("Call fit() before requesting a summary.")
        return PreprocessingSummary(
            numeric_features=tuple(self.numeric_features_),
            categorical_features=tuple(self.categorical_features_),
            ordinal_features=tuple(self.ordinal_features_),
            imputation_values=dict(self.imputation_values_),
            categories={
                feature: tuple(categories)
                for feature, categories in self.categories_.items()
            },
        )

