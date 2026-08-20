from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


XAI_BENCH_COMMIT = "f0431a7c1628a8b66fd322ce86ddcba0d81201d5"
XAI_BENCH_COMMIT_DATE = "2021-10-12T13:46:17-07:00"
DATASET_SEED = 7
SPLIT_SEEDS = (7, 17, 27, 37, 47)
TEST_SIZE = 0.20

WDBC_BASE_FEATURES = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
)
WDBC_FEATURE_COLUMNS = tuple(
    f"{feature}_{summary}"
    for summary in ("mean", "standard_error", "worst")
    for feature in WDBC_BASE_FEATURES
)
WDBC_RAW_COLUMNS = ("ID", "Diagnosis", *WDBC_FEATURE_COLUMNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path, *, full_precision: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"index": False, "lineterminator": "\n"}
    if full_precision:
        options["float_format"] = "%.17g"
    frame.to_csv(path, **options)


def preserve_raw_inputs(root: Path) -> dict[str, str]:
    raw_root = root / "raw"
    original_hashes: dict[str, str] = {}
    for family in ("real", "generated"):
        source_dir = root / family
        target_dir = raw_root / family
        target_dir.mkdir(parents=True, exist_ok=True)
        # Backward-compatible bootstrap for a new release root. Once a raw
        # snapshot exists, it is immutable and processed outputs are never
        # copied back into it on later runs.
        if not any(path.is_file() for path in target_dir.rglob("*")):
            for source in sorted(source_dir.iterdir()):
                if not source.is_file() or source.suffix.lower() not in {".csv", ".json"}:
                    continue
                shutil.copy2(source, target_dir / source.name)
        for target in sorted(path for path in target_dir.rglob("*") if path.is_file()):
            original_hashes[target.relative_to(root).as_posix()] = sha256(target)
    return original_hashes


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["class"].value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def csv_profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns_including_target": int(frame.shape[1]),
        "feature_count": int(frame.shape[1] - 1),
        "missing_cells": int(frame.isna().sum().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": class_counts(frame),
    }


def source_record(
    *, name: str, uci_id: int, doi: str, citation: str
) -> dict[str, Any]:
    return {
        "name": name,
        "repository": "UCI Machine Learning Repository",
        "uci_dataset_id": uci_id,
        "url": f"https://archive.ics.uci.edu/dataset/{uci_id}",
        "doi": doi,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": citation,
    }


def prepare_real_datasets(root: Path) -> dict[str, dict[str, Any]]:
    raw_dir = root / "raw" / "real"
    out_dir = root / "real"
    prepared: dict[str, dict[str, Any]] = {}

    breast_raw_dir = raw_dir / "breast_cancer_wisconsin_diagnostic"
    breast_data_path = breast_raw_dir / "wdbc.data"
    breast_names_path = breast_raw_dir / "wdbc.names"
    if not breast_data_path.is_file() or not breast_names_path.is_file():
        raise FileNotFoundError(
            "The official UCI wdbc.data and wdbc.names files must exist under "
            "raw/real/breast_cancer_wisconsin_diagnostic/."
        )
    breast = pd.read_csv(breast_data_path, header=None, names=WDBC_RAW_COLUMNS)
    if breast.shape != (569, 32):
        raise ValueError(f"Unexpected WDBC shape: {breast.shape}; expected (569, 32).")
    if not set(breast["Diagnosis"].unique()) <= {"B", "M"}:
        raise ValueError("Unexpected WDBC diagnosis labels; expected only B and M.")
    breast = breast.drop(columns=["ID"])
    breast["class"] = breast.pop("Diagnosis").map({"B": 0, "M": 1}).astype(int)
    breast_duplicate_count = int(breast.duplicated().sum())
    breast = breast.drop_duplicates(keep="first").reset_index(drop=True)
    breast_output_path = out_dir / "breast_cancer_wisconsin_diagnostic.csv"
    write_csv(breast, breast_output_path)
    breast_mapping = {
        "schema_version": "1.0",
        "dataset_id": "breast_cancer_wisconsin_diagnostic",
        "display_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_kind": "real",
        "task": "binary_classification",
        "target": {"column": "class", "labels": {"0": "benign", "1": "malignant"}},
        "source": source_record(
            name="Breast Cancer Wisconsin (Diagnostic)",
            uci_id=17,
            doi="10.24432/C5DW2B",
            citation="Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.",
        ),
        "processing": [
            "Parsed the official UCI wdbc.data file with the 32 fields documented in wdbc.names.",
            "Removed ID because it is a record identifier rather than a predictive attribute.",
            "Mapped diagnosis B to class 0 (benign) and M to class 1 (malignant).",
            "Renamed the three groups of ten measurements with explicit _mean, _standard_error, and _worst suffixes.",
            f"Removed {breast_duplicate_count} exact duplicate rows after identifier removal and target mapping, before any train/test split.",
            "Preserved the original UCI row order of every retained observation.",
        ],
        "excluded_columns": {
            "ID": "Unique record identifier; excluded to prevent identifier memorization and avoid exposing an unnecessary identifier."
        },
        "feature_types": {"numeric": list(WDBC_FEATURE_COLUMNS)},
        "feature_groups": {
            "mean": [f"{feature}_mean" for feature in WDBC_BASE_FEATURES],
            "standard_error": [f"{feature}_standard_error" for feature in WDBC_BASE_FEATURES],
            "worst": [f"{feature}_worst" for feature in WDBC_BASE_FEATURES],
        },
        "raw_source_files": {
            "wdbc.data": {
                "sha256": sha256(breast_data_path),
                "role": "Official UCI data file",
            },
            "wdbc.names": {
                "sha256": sha256(breast_names_path),
                "role": "Official UCI dataset documentation",
            },
        },
        "modeling_requirements": [
            "Apply stratified splits before fitting any scaler, feature selector, or model.",
            "Fit numeric scaling on the training partition only.",
            "Report balanced accuracy, macro-F1, ROC-AUC, and calibration-sensitive metrics in addition to accuracy.",
            "Treat malignant as the positive class for binary discrimination metrics.",
        ],
        "clinical_scope_note": (
            "This is a retrospective machine-learning benchmark derived from digitized fine-needle aspirate images. "
            "It must not be presented as a clinically validated diagnostic system or used for patient care."
        ),
        "profile": csv_profile(breast),
        "exact_duplicate_rows_removed_after_id_exclusion": breast_duplicate_count,
    }
    breast_mapping["file_sha256"] = sha256(breast_output_path)
    json_dump(out_dir / "breast_cancer_wisconsin_diagnostic_mappings.json", breast_mapping)
    prepared["breast_cancer_wisconsin_diagnostic"] = breast_mapping

    credit = pd.read_csv(raw_dir / "default_of_credit_card_clients.csv")
    if "ID" not in credit.columns:
        raise ValueError("The raw credit-default file does not contain 'ID'.")
    credit = credit.drop(columns=["ID"])
    credit_duplicates_before_normalization = int(credit.duplicated().sum())
    credit["EDUCATION"] = credit["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    credit["MARRIAGE"] = credit["MARRIAGE"].replace({0: 3})
    credit_duplicate_count = int(credit.duplicated().sum())
    credit = credit.drop_duplicates(keep="first").reset_index(drop=True)
    write_csv(credit, out_dir / "default_of_credit_card_clients.csv")
    credit_mapping = {
        "schema_version": "1.0",
        "dataset_id": "default_of_credit_card_clients",
        "display_name": "Default of Credit Card Clients",
        "dataset_kind": "real",
        "task": "binary_classification",
        "target": {"column": "class", "labels": {"0": "no_default", "1": "default"}},
        "source": source_record(
            name="Default of Credit Card Clients",
            uci_id=350,
            doi="10.24432/C55S3H",
            citation="Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.",
        ),
        "processing": [
            "Removed ID because it is a unique record identifier rather than a predictive attribute.",
            "Collapsed undocumented EDUCATION codes 0, 5, and 6 into the source-defined 'others' code 4.",
            "Collapsed undocumented MARRIAGE code 0 into the source-defined 'others' code 3.",
            "Preserved repayment-status codes -2 and 0 and explicitly flagged them as observed but undocumented by the UCI variable description.",
            f"Removed {credit_duplicate_count} exact duplicate rows after identifier removal and category normalization, before any train/test split.",
        ],
        "excluded_columns": {"ID": "Unique record identifier; excluded to prevent identifier memorization."},
        "feature_types": {
            "categorical": ["SEX", "EDUCATION", "MARRIAGE"],
            "ordinal": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
            "numeric": [
                "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
                "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
            ],
        },
        "categorical_value_labels": {
            "SEX": {"1": "male", "2": "female"},
            "EDUCATION": {"1": "graduate_school", "2": "university", "3": "high_school", "4": "other_or_undocumented"},
            "MARRIAGE": {"1": "married", "2": "single", "3": "other_or_undocumented"},
        },
        "ordinal_value_notes": {
            "PAY_0_to_PAY_6": {
                "source_documented": {"-1": "pay_duly", "1_to_9": "payment_delay_in_months"},
                "observed_but_not_documented_by_uci": [-2, 0],
                "policy": "Preserve the observed codes without inventing semantic labels; treat them consistently across all splits and methods.",
            }
        },
        "modeling_requirements": [
            "Fit all encoders and scalers on the training partition only.",
            "Treat SEX, EDUCATION, and MARRIAGE as categorical despite their integer storage.",
            "Use stratified splits and report class-sensitive metrics because the default class is imbalanced.",
        ],
        "profile": csv_profile(credit),
        "exact_duplicate_rows_after_id_removal_before_category_normalization": credit_duplicates_before_normalization,
        "exact_duplicate_rows_removed_after_processing": credit_duplicate_count,
    }
    credit_mapping["file_sha256"] = sha256(out_dir / "default_of_credit_card_clients.csv")
    json_dump(out_dir / "default_of_credit_card_clients_mappings.json", credit_mapping)
    prepared["default_of_credit_card_clients"] = credit_mapping

    bean_raw = pd.read_csv(raw_dir / "dry_bean_dataset.csv")
    bean = bean_raw.rename(columns={"AspectRation": "AspectRatio", "roundness": "Roundness"})
    duplicate_count = int(bean.duplicated().sum())
    bean = bean.drop_duplicates(keep="first").reset_index(drop=True)
    write_csv(bean, out_dir / "dry_bean_dataset.csv")
    bean_mapping = {
        "schema_version": "1.0",
        "dataset_id": "dry_bean_dataset",
        "display_name": "Dry Bean",
        "dataset_kind": "real",
        "task": "multiclass_classification",
        "target": {
            "column": "class",
            "labels": {"0": "BARBUNYA", "1": "BOMBAY", "2": "CALI", "3": "DERMASON", "4": "HOROZ", "5": "SEKER", "6": "SIRA"},
        },
        "source": source_record(
            name="Dry Bean",
            uci_id=602,
            doi="10.24432/C50S4B",
            citation="Dry Bean [Dataset]. (2020). UCI Machine Learning Repository.",
        ),
        "processing": [
            f"Removed {duplicate_count} exact duplicate rows before any train/test split to prevent cross-partition duplication.",
            "Corrected AspectRation to AspectRatio and roundness to Roundness to match the UCI variable names.",
            "Preserved the original order of the first occurrence of every unique observation.",
        ],
        "feature_types": {"numeric": [column for column in bean.columns if column != "class"]},
        "modeling_requirements": [
            "Create stratified splits only after duplicate removal.",
            "Fit scaling on the training partition only.",
            "Report macro-averaged metrics in addition to accuracy for the seven-class task.",
        ],
        "profile": csv_profile(bean),
        "raw_exact_duplicate_rows": duplicate_count,
    }
    bean_mapping["file_sha256"] = sha256(out_dir / "dry_bean_dataset.csv")
    json_dump(out_dir / "dry_bean_dataset_mappings.json", bean_mapping)
    prepared["dry_bean_dataset"] = bean_mapping

    return prepared


def gaussian_features(rng: np.random.RandomState, n: int = 1000) -> np.ndarray:
    dim = 5
    rho = 0.5
    covariance = np.eye(dim) + (np.ones((dim, dim)) - np.eye(dim)) * rho
    return rng.multivariate_normal(np.zeros(dim), covariance, n)


def linear_target(
    rng: np.random.RandomState, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deterministic = features @ np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    noise = rng.normal(scale=0.01, size=features.shape[0])
    noisy = deterministic + noise
    return (noisy >= 0).astype(int), deterministic, noise


def nonlinear_target(
    rng: np.random.RandomState, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deterministic = (
        -100.0 * np.sin(2.0 * features[:, 0])
        + 2.0 * np.abs(features[:, 1])
        + features[:, 2]
        + np.exp(-features[:, 3])
        - 2.4
    )
    noise = rng.normal(scale=0.01, size=features.shape[0])
    noisy = deterministic + noise
    return (noisy >= 0).astype(int), deterministic, noise


def piecewise_target(
    rng: np.random.RandomState, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, x2 = features[:, 0], features[:, 1], features[:, 2]
    part0 = np.piecewise(x0, [x0 < 0, x0 >= 0], [-1, 1])
    # This exactly reproduces the pinned XAI-Bench implementation. The second
    # predicate is impossible, so the interval [-0.5, 0) receives the default 0.
    part1 = np.piecewise(
        x1,
        [x1 < -0.5, (x1 >= 0.5) & (x1 < 0), (x1 >= 0) & (x1 < 0.5), x1 >= 0.5],
        [-2, -1, 1, 2],
    )
    part2 = (2 * np.cos(x2 * np.pi)).astype(int)
    part2[part2 == 0] = 1
    deterministic = part0 + part1 + part2
    noise = rng.normal(scale=0.01, size=features.shape[0])
    noisy = deterministic + noise
    return (noisy >= 0).astype(int), deterministic, noise


def generated_mapping(
    *,
    dataset_id: str,
    display_name: str,
    position: int,
    directly_relevant: list[str],
    formula: str,
    implementation_notes: list[str],
    frame: pd.DataFrame,
    csv_path: Path,
    oracle_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "display_name": display_name,
        "dataset_kind": "synthetic",
        "task": "binary_classification",
        "target": {"column": "class", "labels": {"0": "negative", "1": "positive"}},
        "source": {
            "name": "XAI-Bench",
            "repository": "https://github.com/abacusai/xai-bench",
            "paper": "Synthetic Benchmarks for Scientific Research in Explainable Machine Learning",
            "paper_url": "https://arxiv.org/abs/2106.12543",
            "paper_doi": "10.48550/arXiv.2106.12543",
            "commit": XAI_BENCH_COMMIT,
            "commit_date": XAI_BENCH_COMMIT_DATE,
            "source_file": "synthetic_datasets/synthetic_gaussian.py",
            "license": "Apache-2.0",
            "license_file": "../LICENSE_XAI_BENCH.txt",
        },
        "generation_parameters": {
            "dataset_seed": DATASET_SEED,
            "shared_rng_stream": True,
            "generation_sequence_position": position,
            "generation_sequence": [
                "gaussian_linear_classification",
                "gaussian_nonlinear_additive_classification",
                "gaussian_piecewise_constant_classification",
            ],
            "rng": "numpy.random.RandomState (MT19937)",
            "num_samples": 1000,
            "dim": 5,
            "mu": [0, 0, 0, 0, 0],
            "rho": 0.5,
            "covariance": "unit diagonal and constant off-diagonal correlation rho=0.5",
            "noise_distribution": "Normal(mean=0, standard_deviation=0.01)",
            "constructor_weight_argument": [4, 3, 2, 1, 0],
            "csv_precision": "IEEE-754 float64 round-trip precision; no three-decimal quantization",
        },
        "feature_columns": [f"feat_{index}" for index in range(5)],
        "directly_relevant_features": directly_relevant,
        "directly_irrelevant_features": [
            feature for feature in [f"feat_{index}" for index in range(5)]
            if feature not in directly_relevant
        ],
        "correlation_caveat": "A feature with no direct term can remain conditionally predictive because all Gaussian features have pairwise correlation rho=0.5. Direct generator relevance is not automatically the attribution ground truth of a fitted model.",
        "latent_function": formula,
        "classification_rule": "class = 1 if latent_function(features) + Normal(0, 0.01) >= 0; otherwise class = 0",
        "implementation_notes": implementation_notes,
        "oracle_companion_file": oracle_path.name,
        "oracle_usage_warning": "The oracle companion file is for evaluation only. Never include row_index, deterministic_score, noise, or noisy_score as model predictors.",
        "modeling_requirements": [
            "Use identical split assignments for every explanation method.",
            "Fit preprocessing on the training partition only.",
            "Use independent model/split seeds; the dataset-generation seed is fixed and must not be presented as a model repeat.",
        ],
        "profile": csv_profile(frame),
        "file_sha256": sha256(csv_path),
        "oracle_file_sha256": sha256(oracle_path),
    }


def prepare_generated_datasets(root: Path) -> dict[str, dict[str, Any]]:
    raw_dir = root / "raw" / "generated"
    out_dir = root / "generated"
    oracle_dir = out_dir / "oracle"
    rng = np.random.RandomState(DATASET_SEED)
    specifications = [
        (
            "gaussian_linear_classification",
            "Gaussian Linear Binary Classification",
            linear_target,
            ["feat_0", "feat_1", "feat_2", "feat_3"],
            "4*x0 + 3*x1 + 2*x2 + x3",
            ["The XAI-Bench weight vector [4, 3, 2, 1, 0] is used directly."],
        ),
        (
            "gaussian_nonlinear_additive_classification",
            "Gaussian Nonlinear Additive Binary Classification",
            nonlinear_target,
            ["feat_0", "feat_1", "feat_2", "feat_3"],
            "-100*sin(2*x0) + 2*abs(x1) + x2 + exp(-x3) - 2.4",
            ["The constructor accepts a weight vector, but the pinned XAI-Bench target implementation does not use it for this dataset."],
        ),
        (
            "gaussian_piecewise_constant_classification",
            "Gaussian Piecewise-Constant Binary Classification",
            piecewise_target,
            ["feat_0", "feat_1", "feat_2"],
            "p0(x0) + p1(x1) + p2(x2), with the exact pinned implementation documented below",
            [
                "p0=-1 for x0<0 and +1 otherwise.",
                "p1=-2 for x1<-0.5; 0 for -0.5<=x1<0; +1 for 0<=x1<0.5; +2 for x1>=0.5.",
                "p2=int(2*cos(pi*x2)); if this integer equals 0, it is replaced by +1.",
                "The zero interval in p1 reproduces the actual pinned XAI-Bench predicate exactly.",
                "The constructor accepts a weight vector, but the pinned XAI-Bench target implementation does not use it for this dataset.",
            ],
        ),
    ]

    prepared: dict[str, dict[str, Any]] = {}
    for position, (dataset_id, display_name, target_fn, relevant, formula, notes) in enumerate(specifications, start=1):
        features = gaussian_features(rng)
        labels, deterministic, noise = target_fn(rng, features)
        frame = pd.DataFrame(features, columns=[f"feat_{index}" for index in range(5)])
        frame["class"] = labels
        csv_path = out_dir / f"{dataset_id}.csv"
        write_csv(frame, csv_path, full_precision=True)

        raw_frame = pd.read_csv(raw_dir / f"{dataset_id}.csv")
        if not np.array_equal(np.round(features, 3), raw_frame.iloc[:, :5].to_numpy()):
            raise AssertionError(f"{dataset_id}: regenerated features do not match the preserved three-decimal source file.")
        if not np.array_equal(labels, raw_frame["class"].to_numpy(dtype=int)):
            raise AssertionError(f"{dataset_id}: regenerated labels do not match the preserved source file.")

        oracle = pd.DataFrame(
            {
                "row_index": np.arange(len(frame), dtype=int),
                "deterministic_score": deterministic,
                "noise": noise,
                "noisy_score": deterministic + noise,
                "class": labels,
            }
        )
        oracle_path = oracle_dir / f"{dataset_id}_oracle.csv"
        write_csv(oracle, oracle_path, full_precision=True)
        mapping = generated_mapping(
            dataset_id=dataset_id,
            display_name=display_name,
            position=position,
            directly_relevant=relevant,
            formula=formula,
            implementation_notes=notes,
            frame=frame,
            csv_path=csv_path,
            oracle_path=oracle_path,
        )
        json_dump(out_dir / f"{dataset_id}_mapping.json", mapping)
        prepared[dataset_id] = mapping

    return prepared


def write_splits(root: Path, datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for dataset_id, frame in datasets.items():
        assignments = pd.DataFrame({"row_index": np.arange(len(frame), dtype=int)})
        per_seed: dict[str, Any] = {}
        for seed in SPLIT_SEEDS:
            split_rng = np.random.RandomState(seed)
            class_groups = {
                value: np.flatnonzero(frame["class"].to_numpy() == value)
                for value in sorted(frame["class"].unique())
            }
            desired_total = int(round(len(frame) * TEST_SIZE))
            exact_counts = {
                value: len(indices) * TEST_SIZE for value, indices in class_groups.items()
            }
            test_counts = {value: int(np.floor(count)) for value, count in exact_counts.items()}
            remainder = desired_total - sum(test_counts.values())
            remainder_order = sorted(
                class_groups,
                key=lambda value: (exact_counts[value] - test_counts[value], -float(value)),
                reverse=True,
            )
            for value in remainder_order[:remainder]:
                test_counts[value] += 1
            test_indices = np.concatenate(
                [split_rng.permutation(indices)[: test_counts[value]] for value, indices in class_groups.items()]
            )
            test_indices = np.sort(test_indices)
            train_mask = np.ones(len(frame), dtype=bool)
            train_mask[test_indices] = False
            train_indices = np.flatnonzero(train_mask)
            values = np.full(len(frame), "train", dtype=object)
            values[test_indices] = "test"
            assignments[f"seed_{seed}"] = values
            per_seed[str(seed)] = {
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "test_class_counts": {
                    str(int(key)): int(value)
                    for key, value in frame.iloc[test_indices]["class"].value_counts().sort_index().items()
                },
            }
        path = split_dir / f"{dataset_id}_splits.csv"
        write_csv(assignments, path)
        summary[dataset_id] = {
            "file": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "seeds": per_seed,
        }
    json_dump(
        split_dir / "split_protocol.json",
        {
            "strategy": "Deterministic class-stratified holdout",
            "implementation": "Within each class, shuffle zero-based row indices with numpy.random.RandomState(seed); allocate round(N*test_size) test rows by largest-remainder class allocation.",
            "test_size": TEST_SIZE,
            "seeds": list(SPLIT_SEEDS),
            "row_index_definition": "Zero-based row position in the corresponding processed CSV.",
            "preprocessing_rule": "Fit every encoder, scaler, imputer, feature selector, and model on train rows only, then apply to test rows.",
            "software": {"numpy": np.__version__},
            "datasets": summary,
        },
    )
    return summary


def validate_dataset(dataset_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "target_is_last_column": frame.columns[-1] == "class",
        "no_missing_values": not bool(frame.isna().any().any()),
        "no_exact_duplicate_rows": not bool(frame.duplicated().any()),
        "target_has_at_least_two_classes": frame["class"].nunique() >= 2,
        "all_columns_have_more_than_one_unique_value": bool((frame.nunique(dropna=False) > 1).all()),
    }
    if dataset_id == "breast_cancer_wisconsin_diagnostic":
        checks["id_excluded"] = "ID" not in frame.columns
        checks["diagnosis_column_replaced_by_class"] = "Diagnosis" not in frame.columns
        checks["expected_feature_count"] = frame.shape[1] - 1 == 30
        checks["binary_target_is_zero_one"] = set(frame["class"].unique()) == {0, 1}
        checks["expected_processed_row_count"] = len(frame) == 569
    if dataset_id == "default_of_credit_card_clients":
        checks["id_excluded"] = "ID" not in frame.columns
        checks["education_codes_normalized"] = set(frame["EDUCATION"].unique()) <= {1, 2, 3, 4}
        checks["marriage_codes_normalized"] = set(frame["MARRIAGE"].unique()) <= {1, 2, 3}
    if dataset_id == "dry_bean_dataset":
        checks["column_names_corrected"] = "AspectRatio" in frame.columns and "Roundness" in frame.columns
    return {
        "dataset_id": dataset_id,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "profile": csv_profile(frame),
    }


def write_documentation(root: Path, manifest: dict[str, Any]) -> None:
    readme = """# Local fiXAIt publication datasets

This directory contains the fixed data release for the local fiXAIt article.

## Directory structure

- `raw/`: immutable snapshot of the files supplied before publication cleaning.
- `real/`: analysis-ready UCI datasets with documented leakage and quality corrections.
- `generated/`: full-precision XAI-Bench synthetic datasets.
- `generated/oracle/`: latent scores and noise for evaluation only; never use these columns as predictors.
- `splits/`: five shared stratified train/test assignments used by every model and explanation method.
- `scripts/`: the exact preparation and audit-report builders used for this release.
- `dataset_manifest.json`: file hashes, software versions, sources, and processing summary.
- `validation_report.json`: machine-readable quality checks.
- `publication_dataset_quality_report.xlsx`: human-readable audit summary.

## Primary analysis files

Use the CSV files directly inside `real/` and `generated/`. The target column is always named `class` and is always the last column. Use the companion JSON mapping for feature types, category semantics, source citations, and dataset-specific cautions.

## Important protocol rules

1. Apply the split assignments before fitting any preprocessing component.
2. Fit encoders, scalers, imputers, feature selection, and models on training rows only.
3. Use the same rows, preprocessing policy, model seed, and model instance for every compared explanation method.
4. Do not use the Breast Cancer Wisconsin `ID` field or the credit-default `ID` field as predictors.
5. Treat integer-coded categorical fields as categorical; their stored integers are not metric distances.
6. Never include the synthetic oracle files or split-assignment columns as model predictors.
7. Distinguish the fixed synthetic dataset seed (`7`) from the five model/split repetition seeds (`7, 17, 27, 37, 47`).
8. Treat Breast Cancer Wisconsin as a retrospective ML benchmark, not as a clinically validated diagnostic system.

## Processing summary

- Breast Cancer Wisconsin (Diagnostic): parsed the official UCI `wdbc.data`; removed record `ID`; mapped B/M to benign/malignant class codes; retained all 30 documented numeric measurements.
- Default of Credit Card Clients: removed unique `ID`; normalized undocumented EDUCATION and MARRIAGE codes into their source-defined `other` levels; removed 35 exact processed duplicates.
- Dry Bean: removed 68 exact duplicate rows before splitting; corrected two column names.
- Synthetic data: regenerated from the pinned XAI-Bench implementation at commit `f0431a7c1628a8b66fd322ce86ddcba0d81201d5`; verified exact agreement with the supplied three-decimal files; exported features at float64 round-trip precision and added separate oracle records.

## Citations and licenses

The three real datasets are from the UCI Machine Learning Repository and are licensed under CC BY 4.0. The synthetic generator is derived from XAI-Bench and its pinned Apache-2.0 source. Use the citations in `CITATION.bib` and retain `LICENSE_XAI_BENCH.txt` when distributing the adapted generator or generated release.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    citation = """@inproceedings{xai-bench-2021,
  title={Synthetic Benchmarks for Scientific Research in Explainable Machine Learning},
  author={Liu, Yang and Khandagale, Sujay and White, Colin and Neiswanger, Willie},
  booktitle={Advances in Neural Information Processing Systems Datasets Track},
  year={2021}
}

@dataset{breast_cancer_wisconsin_diagnostic_uci,
  author={Wolberg, W. and Mangasarian, O. and Street, N. and Street, W.},
  title={Breast Cancer Wisconsin (Diagnostic)},
  year={1993},
  publisher={UCI Machine Learning Repository},
  doi={10.24432/C5DW2B}
}

@dataset{credit_default_uci,
  author={Yeh, I-Cheng},
  title={Default of Credit Card Clients},
  year={2009},
  publisher={UCI Machine Learning Repository},
  doi={10.24432/C55S3H}
}

@dataset{dry_bean_uci,
  title={Dry Bean},
  year={2020},
  publisher={UCI Machine Learning Repository},
  doi={10.24432/C50S4B}
}
"""
    (root / "CITATION.bib").write_text(citation, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the local fiXAIt publication data release.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--xai-bench-source", type=Path, default=None)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not (root / "real").is_dir() or not (root / "generated").is_dir():
        raise FileNotFoundError("dataset_root must contain real/ and generated/ directories.")

    original_hashes = preserve_raw_inputs(root)
    real_metadata = prepare_real_datasets(root)
    generated_metadata = prepare_generated_datasets(root)

    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in real_metadata:
        frames[dataset_id] = pd.read_csv(root / "real" / f"{dataset_id}.csv")
    for dataset_id in generated_metadata:
        frames[dataset_id] = pd.read_csv(root / "generated" / f"{dataset_id}.csv")
    split_summary = write_splits(root, frames)

    validations = [validate_dataset(dataset_id, frame) for dataset_id, frame in frames.items()]
    validation_payload = {
        "overall_status": "pass" if all(item["status"] == "pass" for item in validations) else "fail",
        "datasets": validations,
    }
    json_dump(root / "validation_report.json", validation_payload)
    if validation_payload["overall_status"] != "pass":
        raise AssertionError("At least one publication data validation failed.")

    if args.xai_bench_source is not None:
        license_source = args.xai_bench_source.resolve() / "LICENSE"
        if not license_source.is_file():
            raise FileNotFoundError(f"XAI-Bench license not found: {license_source}")
        shutil.copy2(license_source, root / "LICENSE_XAI_BENCH.txt")

    manifest = {
        "release": "local-fixait-publication-datasets-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed six-dataset release for the local fiXAIt article.",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "raw_snapshot_sha256": original_hashes,
        "real_datasets": real_metadata,
        "synthetic_datasets": generated_metadata,
        "split_assignments": split_summary,
        "validation_report": {
            "file": "validation_report.json",
            "sha256": sha256(root / "validation_report.json"),
            "overall_status": validation_payload["overall_status"],
        },
        "publication_scope_note": "The three XAI-Bench families are additive in their direct data-generating terms. Claims about higher-order interaction recovery require a separate interaction benchmark and are outside this fixed release.",
    }
    write_documentation(root, manifest)
    release_script_dir = root / "scripts"
    release_script_dir.mkdir(parents=True, exist_ok=True)
    release_script = release_script_dir / "prepare_publication_datasets.py"
    shutil.copy2(Path(__file__).resolve(), release_script)
    manifest["documentation"] = {
        name: sha256(root / name)
        for name in ("README.md", "CITATION.bib")
    }
    manifest["documentation"]["scripts/prepare_publication_datasets.py"] = sha256(release_script)
    for relative_name in ("scripts/build_publication_dataset_quality_report.mjs",):
        optional_path = root / relative_name
        if optional_path.exists():
            manifest["documentation"][relative_name] = sha256(optional_path)
    if (root / "LICENSE_XAI_BENCH.txt").exists():
        manifest["documentation"]["LICENSE_XAI_BENCH.txt"] = sha256(root / "LICENSE_XAI_BENCH.txt")
    json_dump(root / "dataset_manifest.json", manifest)

    print(json.dumps({
        "status": "pass",
        "dataset_root": str(root),
        "datasets": {dataset_id: csv_profile(frame) for dataset_id, frame in frames.items()},
        "manifest": str(root / "dataset_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
