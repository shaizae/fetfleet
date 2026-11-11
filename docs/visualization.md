# Visualization and PDF reporting

This document covers the visualization subsystem in `ml_pipe/visualization/`, in this order:

1. **Codebase analysis**: how reporting worked before, and its weak points.
2. **Research**: which plots are worth drawing for this framework, with references.
3. **Architecture**: error isolation, execution flow, and how plots reach the PDF.
4. **Adding a new visualization**.

---

## 1. Codebase analysis (before this change)

| Piece | Where | What it did |
|---|---|---|
| Experiment execution | `base.BaseML.k_folds / train_test_split`, `workers.py` | Process pool (`spawn`), data in shared memory, one task per configuration. Workers return metrics plus **pooled out-of-fold** `y_true`, `y_pred` and `proba`. Fitted models are stripped. |
| Result aggregation | `results/base.py`, `results/classification.py`, `results/regression.py`, `results/experiment.py` | One `*Results` per configuration. `ExperimentReport` ranks them, builds the `leaderboard` and the per-model `summary`, and keeps each model's best result. |
| Visualization | `ClassificationResults.plot_confusion_matrix / plot_roc_curve`, `RegressionResults.plot_prediction / plot_residuals`, `ExperimentReport.plot_roc_curves` | Global `pyplot` state, with no explicit backend. |
| PDF | `save_pdf_report` on each of the three classes above | reportlab. Each one drew 1–2 fixed charts through temp PNG files. |
| CLI pipeline | `config.Experiment._train` | Trained, then wrote the experiment PDF, **then** refitted and saved the best models, then wrote the per-model PDFs. |

Limitations found:

* **A plotting error stopped the experiment.** In `_train`, the experiment PDF was written *before*
  `refit` and `model.joblib` were saved. Any exception (a Tk backend failure, a degenerate class, a model
  without `predict_proba` in an unexpected shape) aborted `ml-pipe run` and lost every trained model.
  `BaseResults.save_results` also wrote the PDF before the model.
* There was no error isolation between plots. The first failing chart cancelled the rest of the PDF.
* The plots used `pyplot` global state. A GUI backend can fail on headless or Windows machines
  (the test suite already forces `Agg` for this reason, `tests/conftest.py`), and leaked figures can pile up.
* The only plots were a confusion matrix, ROC curves and two regression scatters. Nothing compared
  models, folds, hyper-parameters, feature subsets or cost.
* Some data was never recorded. **Fit time**, **per-fold boundaries** and **memory** were not measured.
  **Feature values** are dropped after `evaluate()`, so residual-vs-feature plots are impossible.
* There is no clustering task.

Plotting never runs inside the worker processes, and none of the shared-memory or pool code touches
matplotlib. The interaction risk was in the **parent** process: a plot failing between "training
finished" and "models saved".

---

## 2. Research: which visualizations, and why

Selection rule: a plot is included only if (a) an established reference recommends it for this task,
(b) it can be computed **from the results the pipeline already collects** (or from a cheap,
exact measurement added to the workers), and (c) it answers a question the leaderboard table does not.

### 2.1 Classification

| Visualization | What it communicates / when to use it | Data needed | Insight | Limitations | In ml_pipe? |
|---|---|---|---|---|---|
| **Confusion matrix: counts + row-normalized** [1][2] | Which classes are confused with which. Normalizing each row turns the diagonal into per-class recall. | `y_true`, `y_pred` | Systematic confusions and errors on minority classes, which accuracy hides | Hard to read past ~20 classes. Counts depend on class sizes, which is why the normalized view is shown too. | Yes, best configuration of each model |
| **Per-class precision / recall / F1** [1] | Per-class version of the weighted averages the leaderboard reports | `y_true`, `y_pred` | Shows a model that is good on average but fails one class | Small classes give noisy values (the support is printed) | Yes, best model |
| **Per-class F1 across models (heatmap)** | Which model is best *for each class* | `y_true`, `y_pred` per model | Complementary models, and classes that every model finds hard | Same noise caveat | Yes, when there are ≥ 2 models |
| **Metric comparison across models** [1] | Accuracy, precision, recall and F1 of each model's best configuration, with the majority-class baseline for accuracy | metrics | Whether a model beats the trivial baseline at all | Weighted averages hide per-class behaviour, so the per-class plots complement it | Yes |
| **ROC curve, one curve per model** [2][7] | Threshold-free ranking quality (TPR vs FPR). Multiclass uses the micro-average over one-vs-rest. | `proba` | AUC comparison that does not depend on the decision threshold | Optimistic under heavy class imbalance [6] | Yes (already existed, now isolated) |
| **ROC per class (multiclass)** [2] | One-vs-rest curve per class, plus micro- and macro-average | `proba` of the best model | Which class is hard to separate | One-vs-rest ignores which class it is confused with | Yes, multiclass only |
| **Precision–recall curve** [3][6][8] | Precision vs recall, with average precision (AP). The baseline is the class prevalence. | `proba` | More informative than ROC when positives are rare [6] | No single "chance" diagonal. Multiclass uses the micro-average. | Yes |
| **Calibration (reliability) diagram** [4][9] | Whether predicted probabilities match the observed frequencies. Shown with the Brier score and a histogram of predicted probabilities. | `proba`, binary | Decides whether the probabilities can be used as risks or thresholds | Needs enough rows per bin. Multiclass calibration is not standard. | Yes, binary only |
| **Class distribution** | Rows per class in the evaluated rows | `y_true` | Imbalance, and context for accuracy and for the baseline | none | Yes |
| Feature importance | Model-intrinsic `feature_importances_` or `abs(coef_)` | **fitted** model | Which inputs the model relies on | Impurity importance is biased toward high-cardinality features [10]. `coef_` is only comparable on standardized inputs. Not causal. | Yes, but only for **refitted** models (results from a run hold unfitted models). For permutation importance use `FeatureResearch`. |
| Learning curve [5] | Score vs training-set size | **Re-training** at several sizes | Whether more data would help | Costs extra fits | **No.** It would need new experiments, not collected results. Use `sklearn.model_selection.learning_curve` separately. |
| Validation curve [5] | Score vs one hyper-parameter | The grid results | Under- or over-fitting along one parameter | — | Yes, as *hyper-parameter sensitivity* below (same plot, built from the grid already trained) |

### 2.2 Regression

| Visualization | What it communicates / when | Data | Insight | Limitations | In ml_pipe? |
|---|---|---|---|---|---|
| **Actual vs predicted** (small multiples, equal aspect, identity line) [11][12] | Overall fit. Systematic bias appears as a tilt away from y = x. | `y_true`, `y_pred` | Shrinkage toward the mean, poorly predicted ranges | Overplotting with many rows (alpha and a cap on the point count) | Yes, one panel per model |
| **Residuals vs predicted** [11][13] | The residual structure the identity plot hides | same | Heteroscedasticity, non-linearity, outliers | A binned median line is used instead of LOWESS to avoid a new dependency | Yes, best model |
| **Residual distribution + normal Q–Q** [13][14] | The shape of the errors | residuals | Skew, heavy tails, whether intervals that assume normality are valid | Normality is a diagnostic, not a requirement of the models | Yes, best model (Q–Q needs SciPy, skipped if it is missing) |
| **Absolute error by actual-value range** | Errors in each decile of the target | same | Whether the model fails in the tails | Bins are quantiles of the evaluated rows | Yes, best model |
| **Error distribution across models** (box plot of the absolute error) [15] | The spread of the errors, beyond their mean | per-model residuals | Two models with equal MAE can have very different tails | — | Yes |
| **MAE / RMSE / R² comparison** | Headline metrics side by side. **Separate panels** because the scales differ (no dual axis). | metrics | Ranking under each metric | MSE is RMSE², so it appears only in the tables | Yes |
| Residuals vs features | Residual structure by input | **feature values of the test rows** | Missing interactions | Not available: `evaluate()` drops `features_test`, and the pooled k-fold order does not map back to rows | **No.** Would need a result-structure change. Documented instead. |
| Learning curve / feature importance | as in 2.1 | | | | as in 2.1 |

### 2.3 Model comparison and optimization (every task)

| Visualization | What / when | Data | Insight | Limitations | In ml_pipe? |
|---|---|---|---|---|---|
| **Model ranking** (best score as a dot, range of all configurations as a line) [16] | Which model family wins, and how sensitive it is to its settings | leaderboard | A model that wins only with one lucky setting | The best of many tries is optimistically biased (selection bias) [16] | Yes |
| **Score distribution across configurations** (box plot + every configuration as a point) [15][17] | The spread of each model's configurations | leaderboard | Robustness to hyper-parameters and feature subsets | Box plots hide multimodality, which is why every point is drawn too | Yes, when a model has ≥ 2 configurations |
| **Per-fold scores of each model's best configuration** (box plot + **paired lines**) [16][18] | Variance across folds. Every model in a run is scored on **identical folds**, so fold *i* is connected across models (paired design, as in scikit-learn's grid-search statistics example [18]). | per-fold scores (new: `fold_sizes`) | Whether the gap between two models is larger than the fold-to-fold noise | k scores are few. Folds are not independent, so no p-values are claimed [16][18]. | Yes, k-fold only (one train/test split has no distribution, and leave-one-out folds are single rows) |
| **Hyper-parameter sensitivity** (validation curve from the grid) [5][19] | Best and mean score at each value of one hyper-parameter, per model | leaderboard + grid | Flat = insensitive. A peak at the edge of the grid = extend the grid. | Marginal view that ignores interactions | Yes, one plot per swept parameter |
| **Two-parameter heatmap** [19] | Interaction between the two most-swept parameters (best score in each cell) | leaderboard + grid | Ridges and interactions | Only two dimensions at a time | Yes, per model with ≥ 2 swept parameters |
| **Score vs number of features** | Feature-selection trade-off (brute force, k-best, forward/backward) | leaderboard (`n_features`) | Where adding features stops helping | — | Yes, when more than one subset size was tried |
| **Runtime per configuration** | Training cost per model | new: `fit_time` | Cost of each model family | Wall time on one worker core, all folds, fit + predict. Varies with machine load. | Yes |
| **Score vs computational cost** (with the Pareto front) | Accuracy–cost trade-off | `fit_time` + score | Cheaper configurations that are almost as good | Same caveat on timing | Yes |
| Memory consumption | — | **not measured** | — | Per-task peak memory cannot be measured reliably inside a shared pool worker | **No.** The report records it as *skipped (not measured)*. Nothing is estimated. |

### 2.4 Clustering

Elbow, silhouette [20], cluster sizes and PCA projections are the standard tools. **ml_pipe has no
clustering task**: no unlabeled runs, no clustering metrics, no cluster results. Adding one would need
a new task type, which is a change to the architecture. Per the requirements, no clustering plots are
added. The registry below makes adding them a local change once a clustering task exists.
Any future PCA view must be labeled as a projection, not the original feature space.

### 2.5 Presentation practices applied to every plot

* Show the distribution, not only a bar of means: every point is drawn over box plots [15][17].
* No dual axes. Different scales go into separate panels (small multiples).
* Each model keeps one color in every plot of a report. The colors are a fixed, colorblind-validated categorical order [21].
  Heatmaps use a single-hue sequential scale. Every chart has a title, axis labels with units, and a legend.
* A selected best configuration is optimistically biased, and the report says so [16].
* Chance or trivial baselines are drawn where they exist: the ROC diagonal, the PR prevalence,
  the majority-class accuracy.
* Non-interactive rendering: `matplotlib.figure.Figure` objects drawn by Agg, without `pyplot`. This is
  matplotlib's recommendation for servers and threads [22][23]. No global figure registry, no GUI backend.

### References

1. scikit-learn: Metrics and scoring. https://scikit-learn.org/stable/modules/model_evaluation.html
2. scikit-learn: Multiclass ROC example. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
3. scikit-learn: Precision-Recall example. https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
4. scikit-learn: Probability calibration. https://scikit-learn.org/stable/modules/calibration.html
5. scikit-learn: Validation and learning curves. https://scikit-learn.org/stable/modules/learning_curve.html
6. Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLoS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
7. Fawcett (2006), *An introduction to ROC analysis*, Pattern Recognition Letters. https://doi.org/10.1016/j.patrec.2005.10.010
8. Davis & Goadrich (2006), *The relationship between Precision-Recall and ROC curves*, ICML. https://doi.org/10.1145/1143844.1143874
9. Niculescu-Mizil & Caruana (2005), *Predicting good probabilities with supervised learning*, ICML. https://doi.org/10.1145/1102351.1102430
10. Strobl et al. (2007), *Bias in random forest variable importance measures*, BMC Bioinformatics. https://doi.org/10.1186/1471-2105-8-25 (see also scikit-learn: Permutation importance. https://scikit-learn.org/stable/modules/permutation_importance.html)
11. scikit-learn: Plotting cross-validated predictions. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html
12. scikit-learn: `PredictionErrorDisplay`. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html
13. scikit-learn: Visualizations API. https://scikit-learn.org/stable/visualizations.html
14. SciPy: `scipy.stats.probplot`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
15. Weissgerber et al. (2015), *Beyond Bar and Line Graphs: Time for a New Data Presentation Paradigm*, PLoS Biology. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128
16. Raschka (2018), *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. https://arxiv.org/abs/1811.12808
17. seaborn: Visualizing categorical data. https://seaborn.pydata.org/tutorial/categorical.html
18. scikit-learn: Statistical comparison of models using grid search. https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html ; Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR. https://www.jmlr.org/papers/v7/demsar06a.html
19. Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR. https://www.jmlr.org/papers/v13/bergstra12a.html
20. Rousseeuw (1987), *Silhouettes*, J. Comput. Appl. Math. https://doi.org/10.1016/0377-0427(87)90125-7 ; scikit-learn silhouette example. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
21. Okabe & Ito, *Color Universal Design*. https://jfly.uni-koeln.de/color/
22. matplotlib: Embedding in a web application server (Figure without pyplot). https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html
23. matplotlib: Backends. https://matplotlib.org/stable/users/explain/figure/backends.html

---

## 3. Architecture

```
 ┌──────────────────────┐   results (metrics + pooled y_true/y_pred/proba,
 │ 1. Execution         │   + fit_time, fold_sizes)           never modified below
 │ BaseML + workers     │──────────────────────────────┐
 └──────────────────────┘                              ▼
                                   ┌───────────────────────────────────┐
                                   │ 2. Aggregation                    │
                                   │ ExperimentReport / *Results       │
                                   └───────────────┬───────────────────┘
                                                   │ VisualizationContext (read-only snapshot,
                                                   ▼ references to arrays, no copies)
                                   ┌───────────────────────────────────┐
                                   │ 3. VisualizationManager           │
                                   │  for spec in registry:            │
                                   │   for item in spec.items(ctx):    │  each item isolated:
                                   │     try render → PNG bytes        │  SUCCESS / SKIPPED / FAILED
                                   │     except Skip → SKIPPED         │  + log + traceback
                                   │     except Exception → FAILED     │
                                   │     finally close Figure          │
                                   └───────────────┬───────────────────┘
                                                   │ VisualizationReport (list of VisualizationResult)
                                                   ▼
                                   ┌───────────────────────────────────┐
                                   │ 4. PdfReport                      │
                                   │  sections in fixed order, every   │
                                   │  SUCCESS figure + caption +       │
                                   │  metrics, appendix of the others. │
                                   │  Build fails → text-only retry.   │
                                   │  Atomic write (tmp + replace).    │
                                   └───────────────────────────────────┘
```

### Modules

| Module | Responsibility |
|---|---|
| `visualization/core.py` | `VizStatus` (SUCCESS / SKIPPED / FAILED), `SkipVisualization`, `Rendered`, `VisualizationResult`, `Visualization` (a spec), `VisualizationManager`, `VisualizationReport` (the status tracker), `safe_call` |
| `visualization/style.py` | Theme (`rc_context`), validated categorical palette, sequential colormap, `new_figure()`, model→color mapping |
| `visualization/context.py` | `ResultView` and `VisualizationContext`: read-only snapshots built from an `ExperimentReport` or a single result |
| `visualization/plots/*.py` | The plot specs, grouped by section: `classification`, `regression`, `comparison`, `features` |
| `visualization/pdf.py` | `PdfReport`: sections, figure blocks, tables, appendix, fallback, atomic write |
| `visualization/background.py` | `ReportWorker`: a single daemon thread that renders reports off the main thread |

### Error isolation, layer by layer

0. **Input snapshot.** `visualize()` builds the context through `VisualizationManager.run_for(build)`. If
   the snapshot itself cannot be built, the result is a report with one FAILED record ("Visualization input"),
   not an exception.
1. **Per item.** `VisualizationManager._run_one` wraps the whole lifecycle of one plot: checking the data,
   drawing, encoding to PNG, closing the figure. `SkipVisualization` and a missing optional import
   (`ImportError`) give **SKIPPED** with a reason. Any other `Exception`, `MemoryError` included, gives **FAILED**
   with the type, message and traceback, logged on the `ml_pipe.visualization` logger. `KeyboardInterrupt` and
   `SystemExit` are not swallowed, so the user can still interrupt.
2. **Per spec.** If `items(ctx)` itself raises, that spec gets one FAILED record and the others still run.
3. **Manager level.** `run()` never raises.
4. **PDF.** If building with images fails, the PDF is rebuilt text-only and the failure is listed in the
   appendix. The file is written to a temporary name and then `os.replace`d, so a half-written PDF never
   appears.
5. **Pipeline.** `config.Experiment._train` now refits and **saves the models first**. The experiment PDF
   and the per-model PDFs are then rendered on the `ReportWorker` thread while the main thread keeps working.
   Each job runs through `safe_call`. A failure is logged (`warning: ...`) and the run still returns its
   folder. `BaseResults.save_results` saves the model before the PDF and only logs a PDF failure.
6. **Workers.** Nothing in the visualization package is imported or run in the pool workers. The only
   worker changes record two measurements: `fit_time` from `time.perf_counter`, and `fold_sizes` (an int
   array). Neither can raise.

### Threading

matplotlib is not thread-safe across *shared* figures. Each report job runs on **one** background thread,
jobs run one after another, and each figure is a private `Figure` with no `pyplot`. The context is built on
the main thread *before* the job is submitted, so the report does not depend on timing. The CLI waits for the
worker before returning (bounded by `ReportWorker.wait(timeout)`), so a process exit cannot cut a PDF short.

### Data handling

`ResultView` holds **references** to the result arrays, never copies. Plots only read them, and a test
checks that results are bit-identical after visualization. Figures are encoded as opaque **RGB** PNGs at
150 dpi. RGB rather than RGBA avoids a second soft-mask image per figure in the PDF. The PNGs live in memory
(`bytes`) and are embedded from `BytesIO`, so there are no temp files. Each figure is cleared and dropped
right after encoding. Once the PDF is written, `last_visualizations` keeps the status records **without**
the image bytes. Scatter plots draw at most 5,000 seeded random points, and box plots skip outlier markers
past 20,000 rows. All statistics still use every row.

### PDF structure

Sections appear only if they have content. The order is: Executive Summary · Experiment Configuration ·
Dataset Overview · Overall Results and Model Comparison · Classification Analysis · Regression Analysis ·
Feature Analysis · Cross-Validation Analysis · Hyperparameter Optimization · Runtime and Resource Analysis ·
Visualization Index (appendix: every visualization with its status) · Errors, Warnings and Missing
Visualizations.

Each figure block contains: the title, the model(s) and configuration ranks it shows, the image (aspect
ratio preserved, scaled to fit the frame), a "What it shows" caption, and the metric values the plot was
drawn from. On landscape pages a figure narrower than 2:1 gets the caption and values beside it. Wider
figures (small multiples, metric panels) span the full width.

### Sample reports

`docs/samples/` contains reports built from scikit-learn's bundled datasets. Rebuild them with
`python docs/samples/make_samples.py`.

| File | Run |
|---|---|
| `multiclass_experiment.pdf` | iris, 4 classifiers × grid, 5-fold |
| `binary_experiment.pdf` | breast cancer, 3 classifiers × SelectKBest subsets × grid, 5-fold |
| `binary_best_model.pdf` | the per-model report of the best binary classifier |
| `regression_experiment.pdf` | diabetes, 3 regressors × grid, 5-fold |
| `regression_best_model.pdf` | the per-model report of the best regressor |

### Tests

`tests/test_visualization.py` (39 tests) covers:

* plot generation for both tasks
* a failing plot, and a single failure among successes (also per item and while listing items)
* invalid, degenerate, NaN and mismatched data
* missing SciPy and a missing plotting stack
* PDFs where all, some or no plots succeed, and the text-only fallback
* results left unchanged (no copies, no mutation)
* a real process-pool run with every plot failing, and `ml-pipe run` with every plot **and** every PDF build
  failing: the leaderboard and models are still saved, and shared memory is released
* figure release, with no pyplot figures and no temp files
* exact image counts in the PDF
* the background worker: failures, ordering and timeout

---

## 4. Adding a new visualization

```python
from ml_pipe.visualization import Rendered, SkipVisualization, Visualization, register
from ml_pipe.visualization.style import new_figure

def _items(ctx):                       # optional: one isolated plot per item (e.g. per model)
    return ctx.best_per_model          # default: a single item (None)

def _render(ctx, view):
    if view.y_pred is None:
        raise SkipVisualization("no predictions")      # -> SKIPPED, with this reason
    fig, ax = new_figure(6, 4)
    ax.plot(...)
    ax.set(title=..., xlabel=..., ylabel=...)
    return Rendered(fig, models=(view.label,), metrics={"n": len(view.y_pred)})

register(Visualization(
    key="my_plot",                     # unique; per-item keys get ":<item>" appended
    title="My plot",
    section="Regression Analysis",     # one of visualization.SECTIONS
    description="What it shows and how to read it.",
    tasks=("regression",),             # or ("classification",), or both
    scopes=("experiment", "result"),   # experiment report, per-model report, or both
    items=_items,
    render=_render,
))
```

Rules:

* Draw on the figure returned by `new_figure()`. Never use `pyplot`, never call `show()`.
* Raise `SkipVisualization(reason)` when the data does not support the plot. Do not draw an empty chart.
* Never modify the arrays in a `ResultView`. Copy before any in-place operation.
* Import optional libraries inside `render`. An `ImportError` becomes SKIPPED.
* Put the numbers the plot shows into `Rendered.metrics`, so the PDF prints them under the figure.

The PDF picks the plot up on its own: every SUCCESS result is placed in its section, in registration order.
