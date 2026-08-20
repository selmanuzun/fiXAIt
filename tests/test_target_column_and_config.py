from __future__ import annotations

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pytest

from fixait import CalcFeatureWeight, FiXAIt, FiXAItConfig


def _data(target_name: str = "outcome") -> pd.DataFrame:
    X, y = make_classification(
        n_samples=220,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=17,
    )
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    data[target_name] = np.asarray(["red", "green", "blue"])[y]
    return data


def _model():
    return RandomForestClassifier(
        n_estimators=12,
        max_depth=3,
        random_state=42,
        n_jobs=1,
    )


def test_unified_api_accepts_a_user_defined_target_column():
    data = _data("decision")
    config = FiXAItConfig(
        group_size=4,
        step=1,
        alphas=(0.01, 0.1, 1.0),
        test_size=0.15,
        validation_size=0.20,
        top_k_groups=None,
        sc_metric="jaccard",
        local_baseline="mean",
        n_jobs=1,
        model_n_jobs=1,
    )
    explainer = FiXAIt(_model(), config=config).fit(
        data,
        target_column="decision",
    )

    global_result = explainer.explain_global()
    local_result = explainer.explain_local(data.drop(columns="decision").iloc[0])

    assert explainer.target_column_ == "decision"
    assert global_result.metadata["target_column"] == "decision"
    assert local_result.metadata["target_column"] == "decision"
    assert global_result.global_sc.metric == "jaccard"
    assert local_result.baseline == "mean"
    assert explainer.core_ is not None
    assert explainer.core_.step == 1
    assert explainer.core_.alphas == [0.01, 0.1, 1.0]
    assert explainer.core_.top_k_groups is None


def test_calc_feature_weight_accepts_non_class_target_column():
    data = _data("risk_label")
    encoded = data.copy()
    encoded["risk_label"] = pd.factorize(encoded["risk_label"])[0]
    core = CalcFeatureWeight(
        df=encoded,
        target_column="risk_label",
        model=_model(),
        group_size=4,
        test_size=0.20,
        opt_size=0.20,
        stratify=True,
        random_state=42,
        n_jobs=1,
        model_n_jobs=1,
        verbose=False,
    )

    assert core.target_column == "risk_label"
    assert "risk_label" not in core.data_f.columns
    assert "class" in core.data_f.columns
    assert core.features is not None and core.features[-1] == "class"


def test_original_class_target_remains_supported():
    data = _data("class")
    explainer = FiXAIt(
        _model(),
        config=FiXAItConfig(group_size=4, n_jobs=1, model_n_jobs=1),
    ).fit(data, target_column="class")

    assert explainer.explain_global().metadata["target_column"] == "class"


def test_fit_csv_supports_an_arbitrary_target_and_feature_subset():
    data = _data("decision")
    selected_columns = ["f0", "f1", "f2", "f3", "f4"]
    with TemporaryDirectory() as directory:
        path = f"{directory}/custom_target.csv"
        data.to_csv(path, index=False)
        explainer = FiXAIt(
            _model(),
            config=FiXAItConfig(group_size=4, n_jobs=1, model_n_jobs=1),
        ).fit_csv(
            path,
            target_column="decision",
            usecols=selected_columns,
        )

    assert explainer.target_column_ == "decision"
    assert explainer.feature_names_ == selected_columns
    assert explainer.explain_global().metadata["target_column"] == "decision"


def test_local_faithfulness_requires_at_least_one_perturbation():
    with pytest.raises(ValueError, match="local_faithfulness_runs_per_feature"):
        FiXAItConfig(local_faithfulness_runs_per_feature=0)

    with pytest.raises(
        ValueError,
        match="local_faithfulness_calibration_runs_per_feature",
    ):
        FiXAItConfig(local_faithfulness_calibration_runs_per_feature=0)


def test_local_faithfulness_optimizer_limits_are_validated():
    with pytest.raises(ValueError, match="local_faithfulness_optimizer_steps"):
        FiXAItConfig(local_faithfulness_optimizer_steps=0)

    with pytest.raises(
        ValueError,
        match="local_faithfulness_max_weight_change_pct",
    ):
        FiXAItConfig(local_faithfulness_max_weight_change_pct=101.0)


def test_local_combination_strategy_is_validated():
    with pytest.raises(ValueError, match="local_combination_strategy"):
        FiXAItConfig(local_combination_strategy="unknown")
