# ML Experimentation Template

A small, opinionated starting point for tabular machine learning experiments. It gives you a
config-driven training pipeline, leakage-safe preprocessing, task-aware evaluation, reproducible
artifacts and a single verification command — without pretending to be a production MLOps platform.

There is no model registry, no serving layer, no orchestration and no experiment-tracking service.
Metrics are written as JSON files and models as joblib artifacts. If an experiment graduates to
production, expect to replace those pieces.

## Project structure

```
config/config.yaml               Single source of truth for paths, features, split, model, metrics
data/raw/                        Input datasets (sample.csv ships with the template)
data/processed/                  Cleaned datasets written by training
data/external/                   Third-party data pulled in by your own scripts
notebooks/                       Exploration; import from ml_template/ instead of copying logic
ml_template/config.py            Loads and validates config.yaml into typed dataclasses
ml_template/data_processor.py    Loading, validation and cleaning
ml_template/data_validator.py    Data-quality checks, quality gate and report rendering
ml_template/feature_engineer.py  Feature/target split and the preprocessing ColumnTransformer
ml_template/model.py             Model creation from configuration
ml_template/train.py             Training entry point (orchestration only)
ml_template/evaluate.py          Classification and regression metrics, metric persistence
ml_template/predict.py           Inference with the saved pipeline
tests/                           Unit tests
models/                          Saved pipelines
experiments/results/             Metric JSON files, one per training run
reports/                         Figures and write-ups you produce
reports/data_quality/            Data-quality reports, one per training run
scripts/verify.py                The one verification command (format, lint, types, tests)
scripts/lift_analysis.py         Decile lift on the held-out set, for ranking problems
```

## Installation

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate
```

On macOS/Linux use `source .venv/bin/activate` instead.

```bash
pip install -e ".[dev]"
```

## Configuration

Everything experiment-specific lives in `config/config.yaml`:

| Key | Meaning |
| --- | --- |
| `seed` | Random seed for the split and the model |
| `data.raw_path` | Input dataset |
| `data.processed_path` | Where the cleaned dataset is written |
| `data.target` | Target column |
| `data.separator` | CSV delimiter, defaults to `,` |
| `data.target_mapping` | Optional label → integer map, e.g. `"yes": 1`. Quote the keys — YAML reads bare `yes`/`no` as booleans |
| `features.numerical` / `features.categorical` | Feature columns by type |
| `split.test_size` | Test fraction, between 0 and 1 |
| `split.strategy` | `random` (default, stratified for classification) or `temporal`, which holds out the last rows in file order |
| `model.task` | `classification` or `regression` |
| `model.type` | Model family (`xgboost` ships with the template) |
| `model.output_path` | Where the fitted pipeline is saved |
| `model.params` | Hyperparameters passed straight to the model |
| `evaluation.results_dir` | Where metric JSON files are written |
| `evaluation.metrics` | Metrics to report, validated against the task |
| `validation.min_rows` | Minimum usable rows required to train |
| `validation.max_missing_fraction` | Highest tolerated missing share per feature column |
| `validation.max_duplicate_fraction` | Highest tolerated share of duplicate rows |
| `validation.min_class_fraction` | Smallest tolerated class share (classification only) |
| `validation.value_ranges` | Optional per-column `min`/`max` bounds |
| `validation.report_dir` | Where data-quality reports are written |

Relative paths resolve against **the directory you run from**, so run the commands from the project
root. That keeps a copied project pointing at its own data no matter where the package is installed
— deriving paths from the package location breaks under a non-editable install, where they would
point inside `site-packages`. To run from elsewhere, export `ML_PROJECT_ROOT`.

To run against a different config, pass `--config path/to/config.yaml` or export `ML_CONFIG_PATH`.
The template does not load `.env` files — `.env.example` documents the variables, it is not read at
runtime.

## Adding data

1. Put the CSV in `data/raw/`.
2. Point `data.raw_path` at it and set `data.target`.
3. List the feature columns under `features.numerical` and `features.categorical`.

Columns not listed in the config are dropped. Rows with a missing target are dropped; missing
feature values are imputed inside the pipeline, fitted on training data only.

## Data quality

Every training run validates the dataset twice — once as loaded, once after cleaning — and writes a
Markdown report to `validation.report_dir`. Seven independent checks run at both stages, though
cleaning removes duplicates, missing-target rows and out-of-range values, so those findings should
appear in the raw stage only:

| Check | Fails when |
| --- | --- |
| `schema` | A configured feature or target column is absent |
| `minimum_rows` | Fewer rows than `min_rows` |
| `data_types` | A numerical feature is non-numeric (error), or a categorical feature is numeric (warning) |
| `missingness` | A feature exceeds `max_missing_fraction` |
| `duplicates` | Duplicate rows exceed `max_duplicate_fraction` |
| `invalid_values` | Infinities, or values outside a configured `value_ranges` bound |
| `target_quality` | Target missing, constant, not integer-encoded as `0..n-1` for classification, below `min_class_fraction`, or non-numeric for regression |

The two stages are treated differently on purpose:

- **Raw stage is advisory.** Cleaning is expected to remove duplicates, missing-target rows and
  values outside `value_ranges`, so those findings are reported but do not stop the run. The one
  exception is a `schema` error — without the configured columns there is nothing to clean, so
  training stops immediately.
- **Processed stage is the gate.** Any error after cleaning raises `DataQualityError` and training
  stops before the model is trained or the processed dataset is written.

The report is always written before the gate raises, so a failed run still leaves you the evidence.
Warnings never block. To tune strictness, edit the `validation` block in `config.yaml`; to add a new
rule, write another `check_*` function in `ml_template/data_validator.py` and add it to
`validate_dataset`.

## Training

```bash
python -m ml_template.train
```

Installing the project also exposes `ml-train` and `ml-predict` as console scripts, equivalent to the
`python -m` invocations used throughout this README.

The run loads the data, validates it, cleans it, validates it again, splits it, fits preprocessing
plus model on the training split, evaluates on the test split, writes metrics to
`experiments/results/` and saves the fitted pipeline to `model.output_path`.

## Evaluation

Metrics are computed by `ml_template/evaluate.py` and printed at the end of training, and every run
appends a timestamped JSON file to `experiments/results/` containing the metrics, model type,
parameters and row counts.

- Classification: `accuracy`, `precision`, `recall`, `f1`, and for binary targets `roc_auc` and
  `average_precision` (PR-AUC — prefer it on imbalanced data, where ROC-AUC's 0.5 baseline flatters)
- Regression: `mae`, `rmse`, `r2`

Asking for a metric that does not exist for the configured task fails with a clear error rather than
silently reporting the wrong thing.

## Prediction

```bash
python -m ml_template.predict --input data/raw/sample.csv --output reports/predictions.csv
```

The saved pipeline carries its own preprocessing, so inference applies exactly the transformations
learned during training. The input CSV needs the configured feature columns; the target column is
not required.

Prediction does not apply the quality gate: every input row gets a prediction, including rows that
training would have dropped for falling outside `validation.value_ranges`. Such rows are scored by
extrapolating well outside the training distribution, so screen them yourself if that matters.

## Tests

```bash
python -m pytest
```

## Verification

One command runs the same gates as CI — format check, lint, type check, tests:

```bash
python scripts/verify.py
```

It reports each gate separately and exits non-zero if any fail. `.github/workflows/ci.yml` runs this
exact script on Python 3.10 and 3.12.

## Adapting to another problem

Most changes are configuration only:

- **Different dataset**: update `data.*` and `features.*`.
- **Regression instead of classification**: set `model.task: regression` and replace
  `evaluation.metrics` with `[mae, rmse, r2]`.
- **Different hyperparameters**: edit `model.params`.
- **Different model family**: add an entry to `MODEL_FACTORIES` in `ml_template/model.py` mapping
  `(type, task)` to a scikit-learn compatible estimator, then set `model.type` accordingly. If it
  accepts raw class labels, leave it out of `LABEL_ENCODED_MODEL_TYPES` in the same file and the
  label-encoding check will not apply to it.
- **Different preprocessing**: change the numerical or categorical pipeline in
  `ml_template/feature_engineer.py`. Keep transformers unfitted there so the split stays
  leakage-free.
- **Multiclass classification**: works as-is. Encode the target as `0..n-1`; precision, recall and
  F1 switch to macro averaging automatically, and `roc_auc` is dropped from the available metrics,
  so remove it from `evaluation.metrics`.

XGBoost rejects raw class labels, so with the shipped `model.type` a classification target must be
integers `0..n-1`, booleans, or a numeric categorical. A `yes`/`no` column fails the quality gate
with an explicit message rather than a booster error; map it in your data before training.
