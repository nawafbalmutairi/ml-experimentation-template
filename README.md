# ML Experimentation Template

A small, opinionated starting point for tabular machine learning experiments. It gives you a
config-driven training pipeline, leakage-safe preprocessing, task-aware evaluation, reproducible
artifacts and a single verification command — without pretending to be a production MLOps platform.

There is no model registry, no serving layer, no orchestration and no experiment-tracking service.
Metrics are written as JSON files and models as joblib artifacts. If an experiment graduates to
production, expect to replace those pieces.

## Project structure

```
config/config.yaml        Single source of truth for paths, features, split, model and metrics
data/raw/                 Input datasets (sample.csv ships with the template)
data/processed/           Cleaned datasets written by training
data/external/            Third-party data pulled in by your own scripts
notebooks/                Exploration; import from src/ instead of copying logic
src/config.py             Loads and validates config.yaml into typed dataclasses
src/data_processor.py     Loading, validation and cleaning
src/feature_engineer.py   Feature/target split and the preprocessing ColumnTransformer
src/model.py              Model creation from configuration
src/train.py              Training entry point (orchestration only)
src/evaluate.py           Classification and regression metrics, metric persistence
src/predict.py            Inference with the saved pipeline
tests/                    Unit tests
models/                   Saved pipelines
experiments/results/      Metric JSON files, one per training run
reports/                  Figures and write-ups you produce
scripts/verify.py         The one verification command (format, lint, types, tests)
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
| `features.numerical` / `features.categorical` | Feature columns by type |
| `split.test_size` | Test fraction, between 0 and 1 |
| `model.task` | `classification` or `regression` |
| `model.type` | Model family (`xgboost` ships with the template) |
| `model.output_path` | Where the fitted pipeline is saved |
| `model.params` | Hyperparameters passed straight to the model |
| `evaluation.results_dir` | Where metric JSON files are written |
| `evaluation.metrics` | Metrics to report, validated against the task |

Relative paths resolve against the project root. To run against a different config, pass
`--config path/to/config.yaml` or set `ML_CONFIG_PATH` (see `.env.example`).

## Adding data

1. Put the CSV in `data/raw/`.
2. Point `data.raw_path` at it and set `data.target`.
3. List the feature columns under `features.numerical` and `features.categorical`.

Columns not listed in the config are dropped. Rows with a missing target are dropped; missing
feature values are imputed inside the pipeline, fitted on training data only.

## Training

```bash
python -m src.train
```

The run loads and cleans the data, splits it, fits preprocessing plus model on the training split,
evaluates on the test split, writes metrics to `experiments/results/` and saves the fitted pipeline
to `model.output_path`.

## Evaluation

Metrics are computed by `src/evaluate.py` and printed at the end of training, and every run appends
a timestamped JSON file to `experiments/results/` containing the metrics, model type, parameters and
row counts.

- Classification: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (binary targets only)
- Regression: `mae`, `rmse`, `r2`

Asking for a metric that does not exist for the configured task fails with a clear error rather than
silently reporting the wrong thing.

## Prediction

```bash
python -m src.predict --input data/raw/sample.csv --output reports/predictions.csv
```

The saved pipeline carries its own preprocessing, so inference applies exactly the transformations
learned during training. The input CSV needs the configured feature columns; the target column is
not required.

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
- **Different model family**: add an entry to `MODEL_FACTORIES` in `src/model.py` mapping
  `(type, task)` to a scikit-learn compatible estimator, then set `model.type` accordingly.
- **Different preprocessing**: change the numerical or categorical pipeline in
  `src/feature_engineer.py`. Keep transformers unfitted there so the split stays leakage-free.

Multiclass classification needs one change: `average="binary"` in `src/evaluate.py` becomes
`"macro"` or `"weighted"`.
