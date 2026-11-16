import os
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from ml_pipe.classification import Classification
from ml_pipe.results.classification import ClassificationResults


def test_process_limit_valid():
    Classification.process_limit(1)

    assert Classification._process_limit == 1


def test_process_limit_cannot_exceed_cpu_count():
    Classification.process_limit(os.cpu_count() + 100)

    assert Classification._process_limit == os.cpu_count()


def test_process_limit_negative():
    with pytest.raises(ValueError):
        Classification.process_limit(-1)


@pytest.mark.parametrize("ratio", [-1, 0, 1, 1.5])
def test_train_test_split_rejects_ratio_outside_0_1(classifier, ratio):
    with pytest.raises(ValueError, match="between 0 and 1"):
        classifier.train_test_split(ratio)


def test_k_folds_invalid_splits_low(classifier):
    with pytest.raises(ValueError):
        classifier.k_folds(1)


def test_k_folds_invalid_splits_too_large(classifier):
    dataset_size = len(classifier.targets.data)

    with pytest.raises(ValueError):
        classifier.k_folds(dataset_size + 1)


def test_feature_selection_updates_train_data(classifier):
    number_of_features = [1, 2]
    original_len = len(classifier._train_data)
    classifier.feature_selection(algorithm=chi2, number_of_features=number_of_features)
    assert len(classifier._train_data) == original_len
    assert len(classifier._features_index) == len(number_of_features)
    seen = []
    for i in classifier._features_index:
        assert len(i) in number_of_features
        t = tuple(i)
        assert t not in seen
        seen.append(t)


def test_set_accepts_numpy_target(iris_data):
    X, y = iris_data

    cls = Classification()
    cls.set(X, y.to_numpy(), [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)], )

    assert cls.features is not None
    assert cls.targets is not None
    assert len(cls._train_data) == 2


def test_brute_force_features(classifier):
    original_models = len(classifier._features_index)
    n_features = classifier.features.data.shape[1]
    # every non-empty subset, including the full feature set
    expected_combinations = [combo for i in range(1, n_features + 1) for combo in combinations(range(n_features), i)]
    classifier.brute_force_features()
    assert len(classifier._features_index) == (original_models * len(expected_combinations))
    produced = [tuple(td) for td in classifier._features_index]
    for _ in range(original_models):
        for combo in expected_combinations:
            assert produced.count(combo) == original_models


def test_k_folds_exception(classifier):
    with patch("ml_pipe.base.Pool", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            classifier.k_folds(3)


def test_feature_selection_does_not_need_the_pool(classifier):
    # feature selection scores once in-process; it must not start worker processes
    with patch("ml_pipe.base.Pool", side_effect=RuntimeError("boom"), ):
        classifier.feature_selection(algorithm=chi2, number_of_features=[1, 2], )
    assert len(classifier._features_index) == 2


def test_k_folds_no_shuffle(classifier):
    # shuffle=False: the folds are the rows in order, so two runs score identically
    first = [(r.accuracy, list(r._pred_test)) for r in classifier.k_folds(3, shuffle=False)]
    second = [(r.accuracy, list(r._pred_test)) for r in classifier.k_folds(3, shuffle=False)]
    assert len(first) == 2
    assert first == second


def test_process_limit_zero():
    with pytest.raises(ValueError):
        Classification.process_limit(0)


def test_pool_is_rebuilt_when_process_limit_changes(classifier):
    Classification.process_limit(1)
    classifier.k_folds(3)
    first = classifier._pool
    assert classifier._pool_size == 1
    classifier.k_folds(3)
    assert classifier._pool is first  # same limit: the pool is reused
    Classification.process_limit(2)
    classifier.k_folds(3)
    assert classifier._pool is not first and classifier._pool_size == 2


def test_feature_selection_holds_out_ten_percent_of_rows(iris_data):
    # the rows used to choose features must never be trained or tested on (data leakage)
    X, y = iris_data
    X = X.assign(row_id=np.arange(len(X), dtype=float))  # non-negative, so chi2 accepts it
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [RandomForestClassifier(n_estimators=10, random_state=0)])
        cls.feature_selection(chi2, [2])
        assert len(cls.features.data) == len(cls.targets.data) == 135
        _, held_out = sk_train_test_split(np.arange(150), test_size=0.1, random_state=0)
        remaining_ids = set(cls.features.data[:, -1].astype(int))
        assert remaining_ids.isdisjoint(held_out)
        results = cls.k_folds(3)
        assert all(len(r._pred_test) == 135 for r in results)


# ---------------------------------------------------------------------------
# error handling and fallbacks
# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from ml_pipe.utils import FilteringCriteriaRegression, SharedMemory


class FailingModel(ClassifierMixin, BaseEstimator):
    """Module-level so pool workers can unpickle it."""

    def fit(self, X, y):
        raise RuntimeError("fit failed in worker")


def test_worker_error_propagates_and_pool_recovers(iris_data):
    cls = Classification()
    cls.set(*iris_data, [FailingModel()])
    with pytest.raises(RuntimeError, match="fit failed in worker"):
        cls.k_folds(3)
    assert cls._pool is None  # closed: a failed run never leaves a half-broken pool behind
    assert SharedMemory._shms == {}  # shared memory released even though the run failed

    cls.set(*iris_data, [LogisticRegression(max_iter=500)])
    assert len(cls.k_folds(3)) == 1  # next run starts a fresh pool
    cls.close()


def test_features_dtype_float32(iris_data):
    cls = Classification()
    cls.features_dtype = np.float32
    cls.set(*iris_data, [LogisticRegression(max_iter=500)])
    assert cls._training_features().dtype == np.float32
    with cls:
        result, = cls.k_folds(3)
    assert result.accuracy > 0.9


def tiny_classes():
    """Class 2 has a single row: too small for stratification."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(21, 2)))
    y = pd.Series([0] * 10 + [1] * 10 + [2])
    return X, y


def test_train_test_split_falls_back_when_a_class_is_too_small():
    with Classification() as cls:
        cls.set(*tiny_classes(), [LogisticRegression()])
        result, = cls.train_test_split(0.8)
    assert len(cls._train_index) == 16 and len(result.target_test) == 5


def test_k_folds_warns_and_uses_kfold_when_a_class_is_too_small():
    with Classification() as cls:
        cls.set(*tiny_classes(), [LogisticRegression()])
        with pytest.warns(UserWarning, match="using KFold"):
            result, = cls.k_folds(3)
    assert len(result.target_test) == 21


def test_filter_by_rejects_criteria_of_the_other_task(classifier):
    classifier._results = [ClassificationResults(None)]
    with pytest.raises(ValueError, match="criteria must be one of"):
        classifier.filter_by(FilteringCriteriaRegression.r2)
