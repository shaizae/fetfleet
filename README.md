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
  `tol` only decides when to stop: the last step's tries (one feature more, which did not gain more than
  `tol`) are in the results too, and one of them can still top the leaderboard by a tiny, chance margin.
  To get the subset the search settled on, take the best of the step before (`run.search_log` lists the
  steps).
- Rounds are compared by `select_by` (default accuracy / r2): `run.search("halving", select_by=...)`.

In a config file: `[search] strategy = "halving"` (with `n_iter`, `factor`, `min_rows`), and
`[feature_selection] method = "forward"` or `"backward"` (with `max_features`, `min_features`, `tol`).
`select_by` from `[run]` is used for both.

## Experiment report

After a run, summarize every configuration it trained (classification or regression):

```python
run.k_folds(5)
report = run.experiment_report()          # ranked by accuracy / r2; or experiment_report("f1"), ("mae"), ...
print(report)                             # overview, best configuration, per-model summary, top 10
report.leaderboard                        # DataFrame: every configuration, best first
report.summary                            # DataFrame: per model - configurations, best / mean / std / worst, ...
report.best                               # the best configuration's row
report.save_csv("out")                    # out/leaderboard.csv + out/summary.csv
report.save_pdf_report("out")             # out/experiment_report_<date>.pdf
report.roc_curves()                       # classification: {model: (fpr, tpr, auc)}, best config per model
report.plot_roc_curves()
```

For classification the PDF ends with a ROC curve for each model's best configuration (micro-averaged
for more than two classes; models without `predict_proba` are skipped). `ml-pipe run` writes
`summary.csv` and the experiment PDF into the run folder too.

## Feature research

A stand-alone class (no `Classification` / `Regression` run needed) to understand the features before
choosing them. The task is guessed from `y` (or pass `task="classification"` / `"regression"`).

```python
from ml_pipe import FeatureResearch

research = FeatureResearch(X, y, cv=5)
research.missing_report()               # missing values per column
research.constant_features(0.99)        # constant / quasi-constant columns
research.cardinality()                  # distinct values; flags id-like columns
research.distribution_summary()         # skew, kurtosis, % IQR outliers
research.univariate_scores()            # F-test (+ Pearson / Spearman for regression)
research.mutual_information()           # also catches non-linear relations
research.single_feature_score()         # CV score of a model on each feature alone
research.leakage_check(0.98)            # features that alone score suspiciously high
research.correlation_matrix()           # feature x feature (Spearman by default)
research.high_correlation_pairs(0.9)    # `drop` = the less informative of each pair
research.vif()                          # multicollinearity
research.model_importance(model)        # feature_importances_ / |coef_|
research.permutation_importance(model)  # score drop on held-out folds, any model
research.random_probe(model)            # does each feature beat pure-noise columns?
research.report()                       # one row per feature, every score + mean rank
X = X.drop(columns=research.recommend(probe=True).index)   # constant / id / missing / redundant / noise
research.save_pdf_report("reports")     # all of the above in one PDF, with charts
```

`save_pdf_report` writes `feature_research_<date>.pdf`: overview and target balance, columns to drop and
possible leakage, the feature ranking, a permutation-importance chart (with the best noise column), data
quality, correlated pairs, VIF and a correlation heatmap. The charts are also available on their own:
`research.plot_importance()`, `research.plot_correlation()`.

A feature is called `noise` only when it beats the random noise columns on neither permutation importance
nor mutual information: permutation importance alone under-rates correlated features (shuffling one
costs nothing while its twin is still there).

The data is kept like in a run: `research.features` is a `Features` object and `research.targets` a
`Target` (label-encoded for classification, `research.targets.classes` maps back). A step applied there,
e.g. `research.features.standard_scaler()`, is used by every method after it.

Every method returns a DataFrame or Series. The statistical methods use the numeric columns only, with
missing values filled by the median (`research.skipped` lists the others). The default model is a random
forest; pass any scikit-learn estimator instead.

## Sound when training is done

Pass `sound=` to hear when a run has finished. The sound plays once, when the object goes out of
scope (the end of its `with` block, or when it is garbage-collected), and only if at least one
training run finished.

```python
from ml_pipe import Classification, FilteringCriteriaClassification, Sound

with Classification(sound=Sound.success) as run:   # or sound="success"
    run.set(X, y, models)
    run.k_folds(5)
    best = run.filter_by(FilteringCriteriaClassification.accuracy)
# the success sound plays here

Classification.play_sound(Sound.chime)              # play one any time (a static method)
```

| `Sound`   | What it sounds like                                              |
|-----------|------------------------------------------------------------------|
| `none`    | silent (the default)                                             |
| `ping`    | one soft bell note                                               |
| `double`  | two rising notes, a happy "ding-ding"                            |
| `chime`   | three bell notes ringing into a major chord                      |
| `success` | quick rising arpeggio ending on a bright chord: "it worked!"     |
| `sparkle` | a fast twinkling run up high, playful                            |
| `alarm`   | repeated high/low tone, hard to miss                             |
| `bell`    | the terminal bell only                                           |

The sounds are generated in code (soft bell-like tones), so there are no audio files or extra
dependencies. They play with `winsound` on Windows, `afplay` on macOS and `paplay` / `aplay` on Linux;
if none of these is available, the terminal bell rings instead. Playing a sound never raises an error.

In a config file: `[output] sound = "success"`. The older `ping = true` still works (same as
`sound = "ping"`).

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
- k-folds shuffles the rows, so without a seed each run uses different folds and scores can change a
  little from run to run. Set `run.random_state = 0` (or `random_state` in `[run]`) for repeatable results.
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

