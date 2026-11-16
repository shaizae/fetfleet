"""ml_pipe examples - each function is one small, self-contained example.

Run all of them:          python example.py
Run only some of them:    python example.py classification regression

Every example follows the same 4 steps:
    1. load data           (X = features table, y = target column)
    2. set up a run        (data + the models to try)
    3. train               (train_test_split / k_folds / leave_one_out)
    4. keep the best       (filter_by a metric) and save their reports

Output (PDF reports, PNG plots, CSV, logs) goes into the ./reports folder.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_pipe import ExecutionTracker
from ml_pipe.classification import Classification
from ml_pipe.feature_research import FeatureResearch
from ml_pipe.regression import Regression
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, Sound

REPORTS = Path("reports")


# ============================================================================ data
def load_flowers():
    """Classification data: 150 iris flowers, 4 measurements, 3 species."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # targets are 0/1/2 - replace them with the species names so reports are readable
    y = pd.Series(data.target).map(dict(enumerate(data.target_names)))
    return X, y


def load_patients():
    """Regression data: 442 diabetes patients, 10 features, target = disease progression after a year."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


# ============================================================================ 1. the basics
def classification():
    """The simplest run: try 3 models on an 80/20 split and save the best one."""
    # `with` stops the worker processes when the block ends.
    # sound= is optional: it plays when training is done (ping, double, chime, success, sparkle, alarm, bell).
    with Classification(sound=Sound.ping) as run:
        run.set(*load_flowers(), [RandomForestClassifier(), DecisionTreeClassifier(), SVC()])

        run.features.min_max_scaler()  # scale features to 0..1 (fitted on the training rows only)
        run.train_test_split(0.8)      # train on 80% of the rows, test on the other 20%

        best = run.filter_by(FilteringCriteriaClassification.accuracy)  # the models with the top accuracy
        for result in best:
            print(result)
            result.save_results(REPORTS)


def regression():
    """Same idea for a numeric target: 5-fold cross-validation, keep the lowest error."""
    with Regression() as run:
        run.set(*load_patients(), [LinearRegression(), Ridge(), RandomForestRegressor(random_state=0)])

        run.features.standard_scaler()  # mean 0, std 1
        run.k_folds(5)                  # every model is trained/tested 5 times on different splits

        best = run.filter_by(FilteringCriteriaRegression.rmse)  # rmse / mae: lower is better, r2: higher
        for result in best:
            print(result)
            result.save_results(REPORTS)


# ============================================================================ 2. searching for the best setup
def search_features_and_parameters():
    """Let the pipeline try feature subsets and hyper-parameters for you.

    Every combination model x feature subset x hyper-parameter value is trained and compared.
    """
    with Classification() as run:
        run.set(*load_flowers(), [RandomForestClassifier(random_state=0), SVC()])
        run.features.min_max_scaler()  # 0..1 - chi2 below only accepts non-negative values

        # Feature subsets - pick ONE of these:
        run.feature_selection(chi2, [2, 3])  # only the best 2 and the best 3 features (by a chi2 score)
        # run.brute_force_features()         # every possible subset (fine for few features, slow for many)

        # Hyper-parameters: a name is only applied to the models that have it
        # (n_estimators -> RandomForest, C -> SVC).
        run.set_hyper_parameters({"n_estimators": [50, 100], "C": [0.1, 1, 10]})

        run.k_folds(5)
        best = run.filter_by(FilteringCriteriaClassification.f1)
        print(best[0])
        best[0].save_results(REPORTS)


# ============================================================================ 3. know your features first
def feature_research():
    """Score the features before training, drop the useless ones, then train on the rest."""
    X, y = load_patients()
    research = FeatureResearch(X, y, random_state=0)  # the task (regression) is guessed from y

    print(research.univariate_scores())         # how strongly each feature alone relates to y
    print(research.high_correlation_pairs(0.9))  # pairs of features that carry the same information
    print(research.report())                     # one row per feature, every score + a mean rank (1 = best)

    # The features it recommends dropping, with the reason for each one.
    to_drop = research.recommend(correlation_threshold=0.9, probe=True,
                                 model=RandomForestRegressor(random_state=0))
    print(to_drop)

    research.save_pdf_report(REPORTS, model=RandomForestRegressor(random_state=0))  # everything in one PDF

    # Train only on the features worth keeping.
    with Regression() as run:
        run.set(X.drop(columns=to_drop.index), y, [Ridge(), RandomForestRegressor(random_state=0)])
        run.k_folds(5)
        print(run.filter_by(FilteringCriteriaRegression.r2)[0])


# ============================================================================ 4. reports and plots
def experiment_report():
    """One PDF that compares every model of the run, plus a detailed PDF for the winner."""
    with Classification() as run:
        run.random_state = 0  # same results on every run
        # probability=True lets SVC draw ROC / precision-recall curves
        run.set(*load_flowers(), [RandomForestClassifier(random_state=0), SVC(probability=True, random_state=0)])
        run.features.standard_scaler()
        run.k_folds(5)
        best = run.filter_by(FilteringCriteriaClassification.accuracy)

        # The whole run: summary, model comparison, confusion matrices, ROC, per-fold scores, runtime...
        report = run.experiment_report()
        print(report.save_pdf_report(REPORTS))   # prints the PDF's path
        print(report.last_visualizations.summary())  # e.g. "23 generated, 3 skipped, 0 failed"



        # Only the winning model.
        print(best[0].save_pdf_report(REPORTS))


# ============================================================================ 5. execution logs
def execution_log():
    """Every run records what it did (and what failed, and where) - see docs/execution_tracking.md."""
    logs = REPORTS / "logs"
    with Classification(log_dir=logs) as run:
        run.set(*load_flowers(), [RandomForestClassifier(random_state=0), SVC()])
        run.k_folds(5)

    print(run.execution_log)  # the log as text: result, timings, the tree of stages
    print(run.save_pdf_log)   # saves it as a PDF into `logs` and returns the path
    print(run.save_json_log)  # ... or as JSON

def track_your_own_code():
    """Wrap your own pipeline in an ExecutionTracker: each step is timed and logged.

    Here "Validate" fails on purpose, so the log shows it as FAILED and "Publish" as SKIPPED.
    """
    logs = REPORTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with ExecutionTracker("Nightly retrain", output_dir=logs, console=print) as tracker:
            tracker.plan("Download", "Train", "Validate", "Publish")  # the steps we expect to run

            with tracker.track("Download") as step:
                X, y = load_flowers()
                step.annotate(rows=len(X))  # extra details shown in the log

            with tracker.track("Train"):
                with Classification() as run:  # the run's own stages appear nested under "Train"
                    run.set(X, y, [DecisionTreeClassifier(random_state=0)])
                    run.k_folds(5)

            with tracker.track("Validate"):
                raise ValueError("accuracy below the release threshold")

            with tracker.track("Publish"):
                pass  # never reached
    except ValueError:
        pass  # the tracker logs the error and re-raises it; we expected it here


# ============================================================================ run
EXAMPLES = {
    "classification": classification,
    "regression": regression,
    "search": search_features_and_parameters,
    "research": feature_research,
    "report": experiment_report,
    "log": execution_log,
    "tracking": track_your_own_code,
}

if __name__ == "__main__":
    # On Windows the worker processes re-import this file, so everything must stay under this `if`.
    chosen = sys.argv[1:] or list(EXAMPLES)
    for name in chosen:
        print(f"\n===== {name} =====")
        EXAMPLES[name]()
