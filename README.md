# ml_pipe

A scikit-learn package for running many tests in multiprocessing with minimal garbage creation:
every model x feature subset x hyper-parameter combination, scored on the same splits, best ones refitted.

## Install

```
pip install -e .            # the library
pip install -e ".[dev]"     # + test tools
```

```python
import ml_pipe
from ml_pipe import Classification, FilteringCriteriaClassification

with Classification() as run:
    run.set(X, y, [RandomForestClassifier(), SVC()])
    run.features.standard_scaler()
    run.k_folds(5)
    best = run.filter_by(FilteringCriteriaClassification.accuracy)
```

## Command line

```
ml-pipe init experiment.toml          # example config with every option explained
ml-pipe run experiment.toml --check   # validate the config and the data, no training
ml-pipe run experiment.toml           # train every configuration
ml-pipe predict runs/<date_time>/1_<Model>/model.joblib new.csv -o predictions.csv --proba
```

`python -m ml_pipe ...` works too. A config names the data file, target column and task, the models (any
scikit-learn class name, or an import path like `xgboost.XGBClassifier`), a hyper-parameter grid, preprocessing,
optional feature selection and the validation. Each run writes a new `runs/<date_time>/` folder with
`leaderboard.csv` (every configuration, best first; ties go to fewer features), a copy of the config, and the
top `keep` models, each in its own folder with `model.joblib` and a PDF report.

## Search smarter than every combination

The default tries every model x feature subset x hyper-parameter combination (grid). For large grids:

```python
run.search("random", n_iter=200)                  # 200 combinations drawn at random
run.search("halving", factor=3)                   # successive halving (below)
run.search("halving", n_iter=500, min_rows=200)   # halving on a random sample of the grid
run.forward_selection(max_features=10, tol=0.002) # greedy feature subsets instead of brute force
run.backward_elimination(min_features=3)
run.k_folds(5)
run.search_log                                    # what each round trained
```

- **halving**: every combination is trained on a small share of the rows, the best 1/`factor` go on to
  `factor` x more rows, and the last round, on every row, is an ordinary k-fold / train-test run. Only the
  last round's results are returned, so they compare fairly. Not available with leave-one-out.
- **forward / backward**: for each model x hyper-parameter combination, add (remove) the one feature that
  helps most, until a step gains no more than `tol`. About n²/2 subsets instead of 2ⁿ, so it works with many
  columns. Every subset trained is in the results. Uses the default grid search.
- **optuna** (`pip install ml_pipe[optuna]`): Bayesian search (TPE). Each batch has one trial per worker,
  chosen from the scores of all earlier trials. Hyper-parameters can be ranges as well as lists:
  `run.set_hyper_parameters({"C": Range(0.001, 100, log=True), "max_depth": Range(2, 20, int=True)})`
  with `run.search("optuna", n_iter=100)`. Call `search("optuna")` before `brute_force_features()` and every
  column becomes an on/off switch, so the 2ⁿ subsets are never listed and there is no column limit. A
  configuration proposed twice is not retrained. Config: `C = { low = 0.001, high = 100, log = true }`.
- Rounds are compared by `select_by` (default accuracy / r2): `run.search("halving", select_by=...)`.

In a config file: `[search] strategy = "halving"` (with `n_iter`, `factor`, `min_rows`), and
`[feature_selection] method = "forward"` or `"backward"` (with `max_features`, `min_features`, `tol`).
`select_by` from `[run]` is used for both.

## Use a trained model

```python
path = best[0].save_model("models")       # a .joblib file in models/
model = ml_pipe.load(path)                # only load files you trust
model.predict(new_df)                     # picks the training columns by name; returns the original labels
model.predict_proba(new_df)               # DataFrame, one column per class
model.metrics, model.params, model.feature_columns, model.class_names
```

The saved model includes the run's scaling, polynomial features and feature subset, so pass raw rows.
Models saved before 0.2 (a bare estimator) still load.

Renamed in 0.2: `fetchers` -> `features`, `fetcher_selection()` -> `feature_selection()`,
`brut_force_features()` -> `brute_force_features()`. The old names still work but warn. Imports moved from
`external_classes` / `utils` / `results` into the `ml_pipe` package.

## Usage notes

- `train_test_split(ratio)`: `ratio` is the fraction of rows used for **training** (0.8 = 80% train).
- Scalers and polynomial features (`pipeline.features.standard_scaler()` etc.) are refitted on the
  training rows of every split, so test rows never leak into preprocessing.
- Classification uses stratified splits. All models in a run are scored on the same folds.
- Results carry metrics and predictions only. `filter_by()` (or `refit(results)`) fits the chosen
  models; the fitted model is a scikit-learn `Pipeline` that takes raw rows.
- The worker pool is reused across runs on the same object. Use `with Classification() as c:` or
  call `c.close()` to stop the workers.

## Tests

```
pytest                              # unit tests (load tests are skipped)
pytest --run-load -m load -s        # CPU load tests, a few minutes, prints a CPU timeline
python -m benchmarks.bench          # timing / memory benchmark
```

