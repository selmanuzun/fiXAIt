from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixait import FiXAIt, FiXAItConfig  # noqa: E402


DATASETS = {
    "adult": ("benchmarks/data/adult.csv", "income_label"),
    "boston_housing": ("benchmarks/data/boston_housing.csv", "price_band"),
    "compas": ("benchmarks/data/compas.csv", "risk_level"),
    "german_credit": ("benchmarks/data/german_credit.csv", "creditability"),
    "student": ("benchmarks/data/student.csv", "grade"),
}


MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "decision_tree": lambda: DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=12,
        max_depth=4,
        random_state=42,
        n_jobs=1,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=30,
        max_depth=2,
        random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=700,
        random_state=42,
    ),
}


def run(
    *,
    max_rows: int,
    model_names: list[str],
    dataset_names: list[str],
    optimize_faithfulness: bool = False,
    optimize_local_faithfulness: bool = False,
    optimizer_steps: int = 100,
) -> dict:
    records = []
    started_all = time.perf_counter()
    for dataset_name in dataset_names:
        relative_path, target_name = DATASETS[dataset_name]
        data = pd.read_csv(ROOT / relative_path)
        if len(data) > max_rows:
            data = data.sample(max_rows, random_state=42).reset_index(drop=True)
        data = data.rename(columns={"class": target_name})
        feature_count = data.shape[1] - 1

        for model_name in model_names:
            started = time.perf_counter()
            record = {
                "dataset": dataset_name,
                "model": model_name,
                "target_column": target_name,
                "rows": int(len(data)),
                "features": int(feature_count),
            }
            try:
                explainer = FiXAIt(
                    MODEL_FACTORIES[model_name](),
                    config=FiXAItConfig(
                        group_size=min(7, feature_count),
                        optimize_faithfulness=optimize_faithfulness,
                        optimize_local_faithfulness=optimize_local_faithfulness,
                        faithfulness_runs_per_feature=3,
                        local_faithfulness_runs_per_feature=5,
                        local_faithfulness_calibration_runs_per_feature=5,
                        faithfulness_optimizer_steps=optimizer_steps,
                        local_faithfulness_optimizer_steps=optimizer_steps,
                        top_k_groups=8,
                        n_jobs=1,
                        model_n_jobs=1,
                        random_state=42,
                    ),
                ).fit(data, target_column=target_name)
                global_result = explainer.explain_global()
                local_result = explainer.explain_local(
                    data.drop(columns=target_name).iloc[0]
                )
                optimization = global_result.metadata["optimization"]
                finite_values = [
                    global_result.global_sc.overall,
                    local_result.local_sc.overall,
                    global_result.faithfulness,
                    global_result.fidelity,
                    local_result.fidelity_r2,
                    local_result.fei_fvi_agreement_spearman,
                    local_result.local_faithfulness_spearman,
                ]
                if not np.isfinite(finite_values).all():
                    raise ValueError("A non-finite evaluation value was produced.")
                record.update(
                    {
                        "status": "passed",
                        "selected_features": global_result.selected_features,
                        "optimization_requested": optimization["requested"],
                        "optimization_accepted": optimization["accepted"],
                        "optimization_applied": global_result.optimization_applied,
                        "selected_feature_accuracy": float(
                            global_result.selected_feature_accuracy
                        ),
                        "global_sc": float(global_result.global_sc.overall),
                        "local_sc": float(local_result.local_sc.overall),
                        "local_combination_strategy": local_result.metadata[
                            "combination_strategy"
                        ],
                        "local_combinations": int(
                            local_result.metadata["n_combinations"]
                        ),
                        "local_surrogate_rows": int(
                            local_result.metadata["n_surrogate_rows"]
                        ),
                        "empty_coalition_included": local_result.metadata[
                            "empty_coalition_included"
                        ],
                        "global_faithfulness": float(global_result.faithfulness),
                        "global_fidelity": float(global_result.fidelity),
                        "local_fidelity_r2": float(local_result.fidelity_r2),
                        "local_fei_fvi_agreement_spearman": float(
                            local_result.fei_fvi_agreement_spearman
                        ),
                        "local_faithfulness_spearman": float(
                            local_result.local_faithfulness_spearman
                        ),
                        "local_optimization_requested": local_result.metadata[
                            "optimization"
                        ]["requested"],
                        "local_optimization_accepted": local_result.metadata[
                            "optimization"
                        ]["accepted"],
                        "local_optimization_applied": (
                            local_result.optimization_applied
                        ),
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            record["elapsed_seconds"] = float(time.perf_counter() - started)
            records.append(record)

    passed = sum(record["status"] == "passed" for record in records)
    return {
        "summary": {
            "passed": passed,
            "failed": len(records) - passed,
            "total": len(records),
            "max_rows": max_rows,
            "optimize_faithfulness": optimize_faithfulness,
            "optimize_local_faithfulness": optimize_local_faithfulness,
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds": float(time.perf_counter() - started_all),
        },
        "records": records,
    }


def _parse_names(raw: str, available: dict) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown names: {unknown}. Available: {sorted(available)}")
    return names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the fiXAIt dataset/model regression matrix."
    )
    parser.add_argument("--max-rows", type=int, default=800)
    parser.add_argument("--models", default=",".join(MODEL_FACTORIES))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--optimize-faithfulness", action="store_true")
    parser.add_argument("--optimize-local-faithfulness", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=100)
    arguments = parser.parse_args()

    report = run(
        max_rows=arguments.max_rows,
        model_names=_parse_names(arguments.models, MODEL_FACTORIES),
        dataset_names=_parse_names(arguments.datasets, DATASETS),
        optimize_faithfulness=arguments.optimize_faithfulness,
        optimize_local_faithfulness=arguments.optimize_local_faithfulness,
        optimizer_steps=arguments.optimizer_steps,
    )
    rendered = json.dumps(report, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["summary"]["failed"]:
        raise SystemExit(1)
