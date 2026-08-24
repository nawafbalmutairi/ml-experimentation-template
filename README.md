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
ml_template/group_encoder.py     Leak-safe target encoding for group columns
ml_template/model.py             Model creation from configuration
ml_template/train.py             Training entry point (orchestration only)
ml_template/evaluate.py          Classification and regression metrics, metric persistence
ml_template/predict.py           Inference with the saved pipeline
tests/                           Unit tests
models/                          Saved pipelines
experiments/results/             Metric JSON files, one per training run
reports/                         Write-ups you produce (tracked); figures and prediction CSVs (ignored)
reports/data_quality/            Data-quality reports, one per training run
scripts/verify.py                The one verification command (format, lint, types, tests)
scripts/lift_analysis.py         Decile lift on the held-out set, for ranking problems
scripts/cross_validate.py        Cross-validation, honouring the configured split strategy
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
| `data.target` | Target column. Listing it under `features` too is rejected as leakage |
| `data.separator` | CSV delimiter, defaults to `,` |
| `data.target_mapping` | Optional label → integer map, e.g. `"yes": 1`. Keys keep their YAML type, so numeric labels stay numeric; quote text keys — YAML reads bare `yes`/`no` as booleans |
| `features.numerical` / `features.categorical` | Feature columns by type |
| `features.group_keys` | Optional columns naming a group — a household, a ticket, an order — encoded by the group's target rate rather than used raw |
| `split.test_size` | Test fraction, between 0 and 1. Both strategies round the held-out count up, and refuse a split that leaves either half empty |
| `split.strategy` | `random` (default, stratified for classification) or `temporal`, which holds out the last rows in file order |
| `model.task` | `classification` or `regression` |
| `model.type` | Model family: `xgboost`, or `logistic_regression` for classification |
| `model.output_path` | Where the fitted pipeline is saved |
| `model.params` | Hyperparameters passed straight to the model |
| `evaluation.results_dir` | Where metric JSON files are written |
| `evaluation.metrics` | Metrics to report, validated against the task |
| `validation.min_rows` | Minimum usable rows required to train |
| `validation.max_missing_fraction` | Highest tolerated missing share per feature column |
| `validation.max_duplicate_fraction` | Highest tolerated share of duplicate rows |
| `validation.min_class_fraction` | Smallest tolerated class share (classification only) |
| `validation.value_ranges` | Optional `min`/`max` bounds, keyed by a column listed in `features.numerical`; any other key is rejected as a typo rather than silently ignored |
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
Markdown report to `validation.report_dir`, including how many rows cleaning removed between the two
stages. Seven independent checks are available; six run at both stages, and `duplicates` runs at the
raw stage only:

| Check | Fails when |
| --- | --- |
| `schema` | A configured feature or target column is absent |
| `minimum_rows` | Fewer rows than `min_rows` |
| `data_types` | A numerical feature is non-numeric (error), or a categorical feature is numeric (warning) |
| `missingness` | A feature exceeds `max_missing_fraction` |
| `duplicates` | Rows identical across **every** column exceed `max_duplicate_fraction` (raw stage only) |
| `invalid_values` | Infinities, or values outside a configured `value_ranges` bound |
| `target_quality` | Target missing, constant, not integer-encoded as `0..n-1` for classification, below `min_class_fraction`, or non-numeric for regression |

The two stages are treated differently on purpose:

- **Raw stage is advisory, with two exceptions.** Cleaning is expected to remove missing-target rows
  and values outside `value_ranges`, so those findings are reported but do not stop the run. A
  `schema` error stops it immediately — without the configured columns there is nothing to clean.
  So does a `duplicates` error: cleaning is about to remove those rows, and once it has, the share
  is 0% and `max_duplicate_fraction` could never fire. The raw stage is also the only place
  duplication is measurable, because cleaning drops the columns — a customer id, a timestamp — that
  tell one record from another. Two rows sharing every configured feature and label afterwards are
  distinct observations, not duplicates, and are kept.
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
silently reporting the wrong thing. So does a test split holding a single class: the averaging mode
and positive class come from the classes the model was fitted on, not from whichever survived the
split, so a model predicting nothing but the majority class cannot report a perfect 1.0.

## Prediction

```bash
python -m ml_template.predict --input data/raw/sample.csv --output reports/predictions.csv
```

The saved pipeline carries its own preprocessing, so inference applies exactly the transformations
learned during training. The input CSV needs the configured feature columns; the target column is
not required.

The output adds a `prediction` column, and for binary classification a `score` column holding the
probability of the positive class. Sort by `score` to rank a list — targeting problems need the
order, not the label, and where to cut the list is a budget decision rather than a modelling one.

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

## Cross-validation, and when not to trust it

```bash
python scripts/cross_validate.py --config config/config.yaml --folds 5 --repeats 1
```

It reports every metric in `evaluation.metrics` and honours `split.strategy`: a `temporal` config
gets a rolling origin, where each fold trains only on rows preceding the ones it scores. Shuffling a
temporal problem into random folds trains on the future to predict the past and reports a better
number for doing it.

**Read the output as fit to the training distribution, not as performance on new data.** Every fold
resamples the same rows, so cross-validation cannot see any difference between the data you have and
the data the model will meet. Where those differ — a small dataset, a shifting population, a
competition holdout — it can be confidently wrong, and repeating it narrows the error bars around
the wrong number rather than correcting it.

That is not hypothetical. On a real project here it ranked two models backwards, preferring the one
that went on to score 1.4 points *worse* on a genuine holdout, and it overestimated both by 5-7
points. Repeating it across ten fold partitions raised confidence in the wrong answer.

Two habits follow. Use it to separate models that differ a lot, not to choose between models that
differ a little — a gap inside the fold-to-fold spread is unresolved, not decided. And prefer a
genuine holdout whenever the data allows one; the single split that `train.py` performs is a weaker
estimate but an honest one.

## Group columns

People who travel, shop or transact together often share an outcome. A column naming that group — a
household, a ticket, an order, a device — is useless raw: it is high-cardinality, and new data
mostly contains groups training never saw. What generalises is the group's observed outcome rate.

```yaml
features:
  numerical: [age, fare]
  categorical: [sex, embarked]
  group_keys: [surname, ticket]
```

Each key becomes two columns: the group's target rate, and a flag for whether the group was seen at
all. The flag matters — a missing rate means "no labelled member of this group", which is a fact
about the row rather than an absence to impute away.

**Computing this is a well-known way to leak, so it is done as a fitted transformer inside the
pipeline.** A row is never encoded with its own label; a fold learns rates only from its own
training rows; a held-out row is scored from training rows alone. Computing the same statistic over
a whole dataset before splitting is the tempting shortcut and it inflates every estimate that
follows — measured on synthetic data where groups share an outcome 85% of the time, precomputing
overstated a true holdout by 7.1 points against 3.8 for the fitted encoder.

Note the second figure is not zero. Fitting per fold removes the leak, not cross-validation's own
optimism.

## Features the template does not create

The pipeline selects and transforms columns that already exist; it does not derive new ones. When a
dataset's signal lives in a feature that has to be built — a title parsed out of a name, a family
size summed from two columns, a flag for whether an optional field was filled in — write a project
script that reads `data/raw/` and writes the derived columns to `data/processed/`, then point
`data.raw_path` at that file.

Keep such a script to **row-wise** transforms, where each output depends only on the row it came
from. Those cannot leak, so running them before the split is safe. Anything that learns a statistic
from other rows — imputing a column from its group's median, encoding a category by its target rate
— must go inside the pipeline instead, or it will see the test split during training.

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
