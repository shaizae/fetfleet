"""Save -> load -> predict with ml_pipe.SavedModel."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression

import ml_pipe
from ml_pipe import Classification, FilteringCriteriaClassification, FilteringCriteriaRegression, Regression


@pytest.fixture(scope="module")
def iris_named():
    data = load_iris(as_frame=True)
    y = data.target.map(dict(enumerate(data.target_names)))  # "setosa", ...
    return data.data, y


@pytest.fixture(scope="module")
def best_classifier(iris_named):
    X, y = iris_named
    with Classification() as run:
        run.random_state = 0
        run.set(X, y, [LogisticRegression(max_iter=500), RandomForestClassifier(n_estimators=20, random_state=0)])
        run.features.standard_scaler()
        run.feature_selection(f_classif, [2, 4])
        run.k_folds(3)
        return run.filter_by(FilteringCriteriaClassification.accuracy)[0]


def test_round_trip_predicts_original_labels(best_classifier, iris_named, tmp_path):
    X, y = iris_named
    path = best_classifier.save_model(str(tmp_path))
    model = ml_pipe.load(path)
    assert model.task == "classification"
    assert model.feature_columns == list(X.columns)
    assert sorted(model.class_names) == sorted(y.unique())
    pred = model.predict(X)
    assert set(pred) <= set(y.unique())
    assert (pred == y.to_numpy()).mean() > 0.9
    assert set(model.metrics) == {"accuracy", "precision", "recall", "f1"}
    assert model.ml_pipe_version == ml_pipe.__version__


def test_predict_picks_columns_by_name(best_classifier, iris_named):
    X, _ = iris_named
    model = best_classifier.to_model()
    shuffled = X[X.columns[::-1]].assign(extra=1)
    np.testing.assert_array_equal(model.predict(shuffled), model.predict(X))


def test_predict_reports_missing_columns(best_classifier, iris_named):
    X, _ = iris_named
    with pytest.raises(KeyError, match="missing"):
        best_classifier.to_model().predict(X.drop(columns=X.columns[0]))


def test_predict_proba_has_a_column_per_class(best_classifier, iris_named):
    X, y = iris_named
    proba = best_classifier.to_model().predict_proba(X.head(5))
    assert sorted(proba.columns) == sorted(y.unique())
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert list(proba.index) == list(X.head(5).index)


def test_save_to_explicit_file(best_classifier, tmp_path):
    path = best_classifier.to_model().save(tmp_path / "sub" / "best.joblib")
    assert path == tmp_path / "sub" / "best.joblib" and path.exists()


def test_regression_round_trip(regression_data, tmp_path):
    X, y = regression_data
    with Regression() as run:
        run.set(X, y, [LinearRegression()])
        run.train_test_split(0.8)
        best = run.filter_by(FilteringCriteriaRegression.r2)[0]
    model = ml_pipe.load(best.save_model(str(tmp_path)))
    assert model.task == "regression" and model.class_names is None
    np.testing.assert_allclose(model.predict(X), best.model.predict(X.to_numpy()))
    assert "r2" in model.metrics


def test_load_accepts_a_bare_estimator(tmp_path):
    import joblib
    est = LinearRegression().fit([[0], [1]], [0, 1])
    joblib.dump(est, tmp_path / "old.joblib")
    model = ml_pipe.load(tmp_path / "old.joblib")
    assert model.task == "regression"
    np.testing.assert_allclose(model.predict(np.array([[2.0]])), [2.0])
