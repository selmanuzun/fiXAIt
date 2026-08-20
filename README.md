# fiXAIt

This repository is being reorganized into one package that exposes the original
global fiXAIt method and its instance-level local extension through a shared
CBSFSA/ECFC core.

## Repository layout

- `fixait/`: the only installable runtime package; unified global/local API
- `tests/`: synthetic, regression, compatibility, and real-data smoke tests
- `benchmarks/`: development-only benchmark datasets
- `scripts/`: validation and model-matrix utilities

Only `fixait/` is included in the installable wheel.

## Installation

From the repository root:

```bash
python3 -m pip install .
```

The notebook rank-gradient optimizer uses the optional PyTorch dependency:

```bash
python3 -m pip install ".[optimizer]"
```

## Quick start

An executable VS Code/WSL benchmark notebook is available at
`examples/fixait_benchmark_demo_en.ipynb`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig

data = load_breast_cancer(as_frame=True).frame.rename(
    columns={"target": "diagnosis"}
)
X = data.drop(columns="diagnosis")
y = data["diagnosis"]

explainer = FiXAIt(
    RandomForestClassifier(random_state=42),
    config=FiXAItConfig(
        group_size=7,
        optimize_faithfulness=False,
        optimize_local_faithfulness=False,
    ),
).fit(X, y)

global_result = explainer.explain_global()
local_result = explainer.explain_local(X.iloc[0])

print(global_result.global_fei)
print(global_result.global_fvi)
print(global_result.global_sc.overall)
print(global_result.faithfulness)
print(global_result.fidelity)
print(global_result.optimization_applied)

print(local_result.local_fei)
print(local_result.local_fvi)
print(local_result.local_sc.overall)
print(local_result.fei_fvi_agreement_spearman)
print(local_result.local_faithfulness_spearman)
print(local_result.optimization_applied)
print(local_result.selected_features)
print(local_result.dropped_features)
```

`global_fei` and `global_fvi` are the single final output selected by the
configuration. Following the reference notebook, global FVI is returned without
L1 normalization, FEI thresholding controls the retained features, and the same
feature removals are applied to FVI. `local_fvi` is a signed, normalized
target-score change relative to the chosen baseline. The former local Ridge
formulation remains available as
`legacy_local_fvi` and through `local_fvi_method="legacy_ridge"`. Local FEI uses
the same configured percentage threshold as global FEI, but calculates the
share from absolute local FEI magnitudes. Negative local contributions remain
eligible because their direction is meaningful. The same retained keys are
applied to every local FVI view, and final `local_fvi` is re-normalized.

The default evaluation protocol derives CBSFSA statistics from training rows,
uses a separate validation split for feature/subset selection, and leaves the
test split untouched.

## Target column names

`class` is only an internal compatibility name. A complete DataFrame can use
any target column name:

```python
explainer = FiXAIt(model).fit(data, target_column="risk_label")
```

CSV files are supported directly:

```python
explainer = FiXAIt(model, config=config).fit_csv(
    "credit.csv",
    target_column="creditability",
    usecols=["age", "duration", "credit_amount"],
    read_csv_kwargs={"sep": ","},
)
```

The original form remains valid:

```python
explainer.fit(X, y)
explainer.fit(data, target_column="class")
```

## Mixed data and missing values

The unified API can learn its preprocessing directly from a mixed-type
DataFrame. Object, string, pandas category, and boolean columns are detected as
categorical automatically. Ordinal columns should be declared with their real
order:

```python
explainer = FiXAIt(model, config=config).fit(
    data,
    target_column="credit_result",
    categorical_features=["job", "housing"],  # optional when types are detectable
    ordinal_features=["risk_band"],
    ordinal_categories={
        "risk_band": ["low", "medium", "high"],
    },
)
```

Preprocessing statistics are learned from the training split only. Numeric
missing values use the training median; categorical and ordinal missing values
use the training mode. A category not observed during training receives the
reserved value `-1`. The same fitted transformation is reused for global and
local explanations.

fiXAIt deliberately keeps one output column per input feature so that FEI/FVI
keys remain the user's original feature names. Nominal categories therefore use
deterministic integer codes. This is a natural fit for tree-based models; for a
linear or distance-based estimator, users who do not want that numeric geometry
should supply their preferred numeric encoding as input columns.

## Important configurable parameters

`FiXAItConfig` exposes the parameters that materially affect the algorithm:

| Parameter | Purpose | Default |
|---|---|---:|
| `group_size` | Number of features selected by CBSFSA | `7` |
| `step` | CBSFSA sliding-window step | `1` |
| `alphas` | Ridge regularization candidates | six paper-compatible values |
| `test_size` | Final untouched test fraction | `0.20` |
| `validation_size` | Feature/subset selection fraction | `0.20` |
| `stratify` | Preserve class ratios in splits | `True` |
| `random_state` | Reproducibility seed | `42` |
| `top_k_groups` | Candidate CBSFSA windows; `None` evaluates all | `12` |
| `feature_selection_scope` | `train` for strict evaluation, `full` for compatibility | `train` |
| `sc_metric` | `legacy_overlap`, `jaccard`, or `exact` | `legacy_overlap` |
| `optimize_faithfulness` | Apply the notebook rank-gradient optimizer to final global FEI | `False` |
| `fei_threshold_pct` | Remove global positive-share/local absolute-share FEI values at or below this percentage; `None` disables the percentage threshold | `3.0` |
| `drop_non_positive_fei` | Remove zero and negative global FEI before aligning global FVI; not applied locally | `True` |
| `faithfulness_metric` | `accuracy`, `f1_weighted`, or `neg_log_loss` | `accuracy` |
| `faithfulness_drop_mode` | `metric` or `probability` permutation impact | `metric` |
| `faithfulness_runs_per_feature` | Permutation repetitions per evaluated feature | `30` |
| `faithfulness_optimizer_steps` | Rank-gradient Adam steps when optimization is enabled | `500` |
| `faithfulness_reg_lambda` | Penalty for moving away from the original FEI magnitudes | `0.10` |
| `faithfulness_accept_only_if_improved` | Use calibrated FEI only when validation faithfulness improves enough | `True` |
| `faithfulness_min_improvement` | Minimum validation faithfulness gain required for acceptance | `0.01` |
| `faithfulness_max_weight_change_pct` | Maximum change allowed for each original FEI magnitude; `None` disables the percentage limit | `20.0` |
| `fidelity_top_k` | Maximum FEI features used by the decision-tree surrogate | `7` |
| `fidelity_max_depth` | `auto`, `None`, or a fixed positive depth | `auto` |
| `local_baseline` | `median`, `mean`, or `zero` | `median` |
| `local_score_mode` | `proba`, `margin`, or `logit` | `proba` |
| `local_fvi_method` | `finite_difference` or `legacy_ridge` | `finite_difference` |
| `local_combination_strategy` | `auto`, `exhaustive`, or `ecfc`; automatic ECFC starts at 7 features | `auto` |
| `optimize_local_faithfulness` | Run guarded rank-gradient calibration for local FEI | `False` |
| `local_faithfulness_runs_per_feature` | Reference-value perturbations per retained feature for independent local faithfulness | `30` |
| `local_faithfulness_calibration_runs_per_feature` | Separate reference-value perturbations used to optimize local FEI | `30` |
| `local_faithfulness_optimizer_steps` | Local rank-gradient Adam steps | `500` |
| `local_faithfulness_reg_lambda` | Penalty for moving away from the original local FEI magnitudes | `0.10` |
| `local_faithfulness_accept_only_if_improved` | Accept calibrated local FEI only when held-out faithfulness improves enough | `True` |
| `local_faithfulness_min_improvement` | Minimum held-out local faithfulness gain | `0.01` |
| `local_faithfulness_max_weight_change_pct` | Maximum per-feature local FEI magnitude change | `20.0` |
| `n_jobs` | Parallel worker count | `-1` |
| `model_n_jobs` | Per-model worker count | `1` |
| `verbose` | Progress output | `False` |

Instance-specific choices such as `target_class`, `baseline`, `score_mode`,
`fvi_method`, `optimize_faithfulness`,
`local_faithfulness_runs_per_feature`, and
`local_faithfulness_calibration_runs_per_feature` can also be overridden in each
`explain_local(...)` call.

## Local combination strategy

Local fiXAIt selects its combination method automatically from the CBSFSA
candidate-feature count:

```text
n < 7  -> exhaustive non-empty proper subsets
n >= 7 -> cyclic ECFC subsets
```

The exhaustive branch evaluates `2**n - 2` non-empty proper subsets and removes
the cyclic ordering dependency for small local problems. Its Ridge FEI/FVI
design additionally includes the empty baseline coalition and the full instance,
so it contains exactly `2**n` surrogate rows. The empty coalition is not added
to local SC. The ECFC branch evaluates `n * (n - 1)` subsets and prevents
exponential growth for larger candidate sets; its original surrogate endpoint
behavior remains unchanged. Masked rows and single-feature baseline replacements
are scored in batches rather than through one model call per combination.

The resolved method is recorded in `local_result.metadata`:

```python
print(local_result.metadata["combination_strategy"])
print(local_result.metadata["n_candidate_features"])
print(local_result.metadata["n_combinations"])
print(local_result.metadata["n_surrogate_rows"])
print(local_result.metadata["empty_coalition_included"])
print(local_result.metadata["combination_space_complete"])
```

Before RidgeCV, coalition members and rows are canonicalized by subset size and
selected-feature index. Local FEI and legacy FVI then share the same shuffled,
seeded K-fold definition:

```python
print(local_result.metadata["ridge_cv_strategy"])
print(local_result.metadata["ridge_cv_splits"])
print(local_result.metadata["ridge_cv_shuffle"])
print(local_result.metadata["ridge_cv_random_state"])
```

This prevents contiguous coalition structures from occupying separate folds and
makes alpha/coefficients invariant to the input order of the same coalition set.

## Local FEI thresholding

Global and local explanations share `fei_threshold_pct`, while respecting the
different meaning of their scores. Global selection uses each positive FEI's
share of the positive total. Local selection uses
`abs(local_fei) / sum(abs(local_fei))`, preserves the original sign, and does
not apply `drop_non_positive_fei`. Set `fei_threshold_pct=None` to retain all
candidate local features.

Local SC remains the original pre-threshold coalition consistency result. Local
FEI--FVI agreement, local faithfulness, and fidelity describe the final
post-threshold feature set. If fewer than two features remain, the corresponding
Spearman rank score is `0.0` and its metadata `*_informative` flag is `False`.

## Local FEI--FVI agreement and faithfulness

Local fiXAIt reports two distinct rank-based diagnostics:

```python
print(local_result.fei_fvi_agreement_spearman)
print(local_result.local_faithfulness_spearman)
```

`fei_fvi_agreement_spearman` compares absolute local FEI with the absolute
single-baseline finite-difference FVI. It measures agreement between the two
local fiXAIt impact views; it is not an independent faithfulness evaluation.

`local_faithfulness_spearman` compares absolute local FEI with independent
per-feature perturbation impacts. For every retained feature, values are drawn
from shared, seeded reference-row indices and substituted into the explained
instance. The impact is the mean absolute change in the fixed target score over
`local_faithfulness_runs_per_feature` draws. The sampled impacts and protocol
are recorded in metadata:

```python
print(local_result.metadata["local_faithfulness_impacts"])
print(local_result.metadata["local_faithfulness_sampling"])
print(local_result.metadata["local_faithfulness_runs_per_feature"])
print(local_result.metadata["local_faithfulness_informative"])
```

For backward compatibility, `local_result.faithfulness_spearman` remains a
deprecated read-only alias for `fei_fvi_agreement_spearman`.

## Optional local faithfulness optimization

Local optimization is disabled by default and can be selected independently of
global optimization:

```python
config = FiXAItConfig(
    optimize_local_faithfulness=True,
    local_faithfulness_calibration_runs_per_feature=30,
    local_faithfulness_runs_per_feature=30,
    local_faithfulness_optimizer_steps=500,
    local_faithfulness_reg_lambda=0.10,
    local_faithfulness_accept_only_if_improved=True,
    local_faithfulness_min_improvement=0.01,
    local_faithfulness_max_weight_change_pct=20.0,
)

local_result = FiXAIt(model, config=config).fit(X, y).explain_local(X.iloc[0])
print(local_result.local_fei)
print(local_result.optimization_applied)
print(local_result.metadata["optimization"])
```

It can also be enabled or disabled for one instance without rebuilding the
explainer:

```python
local_result = explainer.explain_local(
    X.iloc[0],
    optimize_faithfulness=True,
)
```

Calibration impacts and held-out evaluation impacts use separate seeded
reference-row draws whenever the reference set is large enough. Rank-gradient
optimization preserves every coefficient sign and zero value, limits each
magnitude change to the configured percentage, and does not rescale the final
weights. With the default guard, the candidate replaces the Ridge local FEI only
when held-out Spearman faithfulness improves by at least `0.01`; otherwise the
original local FEI continues through thresholding. The main result returns only
that selected FEI. Local SC always remains based on the original pre-threshold
Ridge FEI.

## Optional global faithfulness optimization

The user selects exactly one global-output mode. The main result never returns
raw and optimized FEI dictionaries side by side.

Notebook FEI/FVI without rank-gradient optimization:

```python
config = FiXAItConfig(optimize_faithfulness=False)
```

Notebook rank-gradient optimization on the validation split:

```python
config = FiXAItConfig(
    optimize_faithfulness=True,
    faithfulness_runs_per_feature=30,
    faithfulness_optimizer_steps=500,
    faithfulness_reg_lambda=0.10,
    faithfulness_accept_only_if_improved=True,
    faithfulness_min_improvement=0.01,
    faithfulness_max_weight_change_pct=20.0,
)
```

In both modes, the main fields contain only the selected final result:

```python
result = FiXAIt(model, config=config).fit(X, y).explain_global()

print(result.global_fei)
print(result.global_fvi)
print(result.selected_features)
print(result.dropped_features)
print(result.optimization_applied)

optimization = result.metadata["optimization"]
print(optimization["requested"])
print(optimization["accepted"])
print(optimization["validation_faithfulness_improvement"])
print(optimization["mean_weight_change_pct"])
print(optimization["max_weight_change_pct"])
```

When optimization is requested, permutation impacts are derived from validation
rows. Each non-zero FEI magnitude can move by at most its configured percentage,
its original sign is restored, and a zero FEI remains zero. No final min-max
rescaling is applied. With the default guard, the candidate is used only when
validation faithfulness improves by at least `0.01`; otherwise the original FEI
continues through thresholding and FVI alignment. `optimization_applied` therefore
means that the candidate was accepted, not merely requested.

Final faithfulness and fidelity are evaluated on the untouched test rows. Global
self-consistency remains the original fiXAIt core consistency measure and is not
recomputed from calibrated weights.

## Faithfulness and fidelity evaluation

The final global result automatically contains both evaluation scores:

```python
result = explainer.explain_global()

print(result.global_faithfulness.score)
print(result.global_faithfulness.drop_impacts)
print(result.global_fidelity.score)
print(result.global_fidelity.selected_features)
```

Faithfulness can also be evaluated explicitly with other settings:

```python
faithfulness = explainer.evaluate_global_faithfulness(
    split="test",
    metric="accuracy",
    runs_per_feature=30,
    drop_mode="metric",
    conditional_permutation=False,
)
```

Probability-drop faithfulness is available for classifiers with
`predict_proba`:

```python
faithfulness = explainer.evaluate_global_faithfulness(
    split="test",
    drop_mode="probability",
    target_class=1,
)
```

Local stability keeps the explained class fixed and supports separately declared
categorical and ordinal features:

```python
explainer.fit(
    X,
    y,
    categorical_features=["education", "marital_status"],
    ordinal_features=["education_level"],
)

stability = explainer.evaluate_local_stability(
    X.iloc[0],
    n_perturbations=20,
)
```

The quick five-dataset validation script is available at
`scripts/validate_datasets.py`. A broader regression matrix over all five data
sets and decision-tree, random-forest, gradient-boosting, and logistic-regression
models is available at `scripts/benchmark_matrix.py`:

```bash
python scripts/benchmark_matrix.py --max-rows 800 --output benchmark.json
```

Run the same matrix with notebook rank-gradient optimization enabled:

```bash
python scripts/benchmark_matrix.py \
  --max-rows 800 \
  --optimize-faithfulness \
  --optimizer-steps 500 \
  --output benchmark-optimized.json
```
