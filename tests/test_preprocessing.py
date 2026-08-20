from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig, TabularPreprocessor


def test_tabular_preprocessor_preserves_names_and_handles_unknown_values():
    train = pd.DataFrame(
        {
            "amount": [1.0, np.nan, 3.0, 5.0],
            "city": ["ankara", None, "izmir", "ankara"],
            "tier": ["low", "medium", None, "high"],
            "active": [True, False, True, True],
        }
    )
    preprocessor = TabularPreprocessor(
        ordinal_features=["tier"],
        ordinal_categories={"tier": ["low", "medium", "high"]},
    ).fit(train)

    transformed = preprocessor.transform(
        pd.DataFrame(
            {
                "amount": [np.nan],
                "city": ["bursa"],
                "tier": ["high"],
                "active": [False],
            }
        )
    )

    assert list(transformed.columns) == list(train.columns)
    assert transformed.loc[0, "amount"] == 3.0
    assert transformed.loc[0, "city"] == -1.0
    assert transformed.loc[0, "tier"] == 2.0
    assert set(preprocessor.categorical_features_) == {"city", "active"}
    assert preprocessor.ordinal_features_ == ["tier"]


def _mixed_data() -> pd.DataFrame:
    values, target = make_classification(
        n_samples=240,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=19,
    )
    data = pd.DataFrame(values, columns=[f"numeric_{index}" for index in range(5)])
    data["region"] = np.where(
        values[:, 0] > 0.6,
        "north",
        np.where(values[:, 0] < -0.6, "south", "central"),
    )
    data["tier"] = pd.cut(
        values[:, 1],
        bins=[-np.inf, -0.4, 0.7, np.inf],
        labels=["low", "medium", "high"],
    ).astype(object)
    data["active"] = values[:, 2] > 0
    data.loc[data.index[::29], "numeric_0"] = np.nan
    data.loc[data.index[::31], "region"] = None
    data.loc[data.index[::37], "tier"] = None
    data["decision"] = np.where(target == 1, "approve", "reject")
    return data


def test_unified_api_supports_mixed_types_missing_values_and_unseen_categories():
    data = _mixed_data()
    features = data.drop(columns="decision")
    explainer = FiXAIt(
        RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            n_jobs=1,
        ),
        config=FiXAItConfig(
            group_size=4,
            top_k_groups=6,
            n_jobs=1,
            model_n_jobs=1,
        ),
    ).fit(
        data,
        target_column="decision",
        ordinal_features=["tier"],
        ordinal_categories={"tier": ["low", "medium", "high"]},
    )

    unseen = features.iloc[0].copy()
    unseen["region"] = "previously_unseen"
    unseen["numeric_2"] = np.nan
    global_result = explainer.explain_global()
    local_result = explainer.explain_local(unseen)

    assert explainer.preprocessor_ is not None
    assert set(explainer.preprocessor_.feature_names_) == set(features.columns)
    assert set(global_result.selected_features).issubset(features.columns)
    assert set(local_result.selected_features).issubset(features.columns)
    assert global_result.metadata["preprocessing"]["fit_scope"] == "train"
    assert "region" in global_result.metadata["preprocessing"]["categorical_features"]
    assert global_result.metadata["preprocessing"]["ordinal_features"] == ["tier"]
    assert local_result.metadata["preprocessing"]["unknown_category_code"] == -1.0
    assert np.isfinite(list(global_result.global_fei.values())).all()
    assert np.isfinite(list(local_result.local_fei.values())).all()
