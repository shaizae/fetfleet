"""Rebuild the sample reports in this folder:  python docs/samples/make_samples.py

multiclass_experiment.pdf   iris: 4 classifiers x a hyper-parameter grid, 5-fold CV
binary_experiment.pdf       breast cancer: 3 classifiers x SelectKBest subsets x grid, 5-fold CV
binary_best_model.pdf       per-model report of the refitted best binary classifier
regression_experiment.pdf   diabetes: 3 regressors x grid, 5-fold CV
regression_best_model.pdf   per-model report of the refitted best regressor
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml_pipe import Classification, Regression

HERE = Path(__file__).parent


def keep(path: Path, name: str) -> None:
    shutil.move(str(path), HERE / name)
    print(HERE / name)


def main() -> None:
    work = Path(tempfile.mkdtemp())

    d = load_iris()
    with Classification() as run:
        run.random_state = 0
        run.set(pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target_names[d.target]),
                [RandomForestClassifier(n_estimators=100, random_state=0), LogisticRegression(max_iter=1000),
                 SVC(probability=True, random_state=0), LinearSVC()])
        run.features.standard_scaler()
        run.set_hyper_parameters({"max_depth": [2, 4, None], "C": [0.01, 0.1, 1, 10], "kernel": ["rbf", "linear"]})
        run.k_folds(5)
        run.filter_by(run.default_criteria)  # refit the best, so the report can show their importances
        keep(run.experiment_report().save_pdf_report(work), "multiclass_experiment.pdf")

    d = load_breast_cancer()
    with Classification() as run:
        run.random_state = 0
        run.set(pd.DataFrame(d.data[:, :10], columns=d.feature_names[:10]), pd.Series(d.target_names[d.target]),
                [RandomForestClassifier(n_estimators=100, random_state=0), LogisticRegression(max_iter=5000),
                 DecisionTreeClassifier(random_state=0)])
        run.features.standard_scaler()
        run.feature_selection(f_classif, [2, 4, 6, 8, 10])
        run.set_hyper_parameters({"max_depth": [2, 4, 8], "C": [0.1, 1, 10]})
        run.k_folds(5)
        best = run.filter_by(run.default_criteria)
        keep(run.experiment_report().save_pdf_report(work), "binary_experiment.pdf")
        keep(best[0].save_pdf_report(str(work)), "binary_best_model.pdf")

    d = load_diabetes()
    with Regression() as run:
        run.random_state = 0
        run.set(pd.DataFrame(d.data, columns=d.feature_names), d.target,
                [Ridge(), Lasso(max_iter=10000), RandomForestRegressor(n_estimators=100, random_state=0)])
        run.set_hyper_parameters({"alpha": [0.01, 0.1, 1, 10], "max_depth": [3, 6, None], "min_samples_leaf": [1, 10]})
        run.k_folds(5)
        best = run.filter_by(run.default_criteria)
        keep(run.experiment_report().save_pdf_report(work), "regression_experiment.pdf")
        keep(best[0].save_pdf_report(str(work)), "regression_best_model.pdf")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
