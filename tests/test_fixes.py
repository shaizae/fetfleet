"""Regression tests for the correctness fixes. Each one fails on the old code."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.workers import select_k_best_many
from ml_pipe.utils import SharedMemory, FilteringCriteriaClassification, FilteringCriteriaRegression, ValidationType


@pytest.fixture(autouse=True)
def no_shared_memory_left():
    yield
    leaked = list(SharedMemory._shms)
    SharedMemory.cleanup()
    assert not leaked


def iris():
    d = load_iris()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)


def test_k_folds_scores_are_out_of_fold():
    """1-NN memorises its training rows. If any fold is predicted by a model that saw
    it (the old bug), random labels score near (k-1)/k instead of chance."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)))
    y = pd.Series(rng.integers(0, 3, 300))
    with Classification() as cls:
        cls.set(X, y, [KNeighborsClassifier(n_neighbors=1)])
        result, = cls.k_folds(5)
    assert result.accuracy < 0.5  # chance is 1/3; the leaking version scored ~0.8


def test_regression_target_keeps_its_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)))
    y = pd.Series(1000 + 3 * X[0] - 2 * X[1])  # exact linear target, large offset
    with Regression() as reg:
        reg.set(X, y, [LinearRegression()])
        np.testing.assert_allclose(reg.targets.data, y.to_numpy())  # not label-encoded ranks
        result, = reg.k_folds(5)
    assert result.r2 > 0.999
    assert result.mae < 1e-6
    np.testing.assert_allclose(np.sort(result.target_test), np.sort(y.to_numpy()))


def test_ratio_is_the_training_fraction_and_split_is_stratified():
    with Classification() as cls:
        cls.set(*iris(), [LogisticRegression(max_iter=500)])
        result, = cls.train_test_split(0.8)
        assert len(cls._train_index) == 120
        assert np.bincount(result.target_test).tolist() == [10, 10, 10]  # stratified


def test_k_folds_are_stratified_and_shared():
    cls = Classification()
    cls.set(*iris(), [LogisticRegression(max_iter=500)])
    ids = cls._fold_ids(5, True, 0, cls.targets.data)
    for fold in range(5):
        assert np.bincount(cls.targets.data[ids == fold]).tolist() == [10, 10, 10]


def test_scaler_is_fitted_on_training_rows_only():
    with Classification() as cls:
        cls.set(*iris(), [LogisticRegression(max_iter=500)])
        cls.features.standard_scaler()
        cls.train_test_split(0.5)
        best = cls.filter_by(FilteringCriteriaClassification.accuracy)[0]
        scaler = best.model.named_steps["pre0"]
        raw = cls.features.raw_data
        np.testing.assert_allclose(scaler.mean_, raw[cls._train_index].mean(axis=0))
        assert not np.allclose(scaler.mean_, raw.mean(axis=0))


def test_preprocessing_after_set_uses_all_new_columns():
    """polynomial_features() after set(): train on every generated column, not the
    first 4 (the old default feature index)."""
    with Classification() as cls:
        cls.set(*iris(), [LogisticRegression(max_iter=500)])
        cls.features.polynomial_features(degree=2)
        result, = cls.k_folds(3)
    assert len(result.features_index) == cls.features.data.shape[1] == 14


def test_refit_returns_pipeline_that_takes_raw_rows():
    with Classification() as cls:
        cls.set(*iris(), [LogisticRegression(max_iter=500)])
        cls.features.min_max_scaler()
        cls.k_folds(3)
        best = cls.filter_by(FilteringCriteriaClassification.accuracy)[0]
    assert isinstance(best.model, Pipeline)
    assert best.name == "LogisticRegression"
    assert best.model.predict(iris()[0].to_numpy()).shape == (150,)


def test_leave_one_out_probabilities_cover_all_classes():
    with Classification() as cls:
        cls.set(*iris(), [SVC(probability=True, random_state=0)])
        result, = cls.leave_one_out()
    assert result.validation == ValidationType.leave_one_out
    assert result._predict_score.shape == (150, 3)


def test_select_k_best_many_matches_selectkbest_per_k():
    X, y = load_iris(return_X_y=True)
    got = select_k_best_many(f_classif, X, y, [1, 2, 3])
    for k, idx in zip([1, 2, 3], got):
        expected = np.where(SelectKBest(f_classif, k=k).fit(X, y).get_support())[0]
        np.testing.assert_array_equal(idx, expected)


def test_select_k_best_many_drops_duplicate_full_sets():
    X, y = load_iris(return_X_y=True)
    assert len(select_k_best_many(chi2, X, y, [10, 20])) == 1


def test_brute_force_refuses_huge_search():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 20)))
    cls = Classification()
    cls.set(X, pd.Series([0, 1] * 5), [LogisticRegression()])
    with pytest.raises(ValueError, match="1,048,575 subsets"):
        cls.brute_force_features()


def test_pool_is_reused_and_closed():
    cls = Classification()
    cls.set(*iris(), [LogisticRegression(max_iter=500)])
    cls.k_folds(3)
    pool = cls._pool
    cls.train_test_split()
    assert cls._pool is pool
    cls.close()
    assert cls._pool is None
    cls.k_folds(3)  # starts a fresh pool after close()
    cls.close()


def test_runs_use_unique_shared_memory_names():
    cls = Classification()
    cls.set(*iris(), [LogisticRegression(max_iter=500)])
    cls.k_folds(3)
    first = cls._run_id
    cls.k_folds(3)
    assert cls._run_id != first
    cls.close()


def test_regression_filter_by_still_refits():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 2)))
    with Regression() as reg:
        reg.set(X, pd.Series(2 * X[0]), [LinearRegression()])
        reg.train_test_split()
        best, = reg.filter_by(FilteringCriteriaRegression.r2)
    np.testing.assert_allclose(best.model.coef_, [2, 0], atol=1e-9)
