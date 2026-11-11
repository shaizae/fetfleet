import os
from itertools import combinations
from unittest.mock import patch

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from ml_pipe.base import _add_hyper_parameter
from ml_pipe.classification import Classification
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.utils import create_shared_memory, cleanup_shared_memory
from ml_pipe.train_data import TrainData, KFoldsTrainData


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


def test_use_more_features_than_exist_in_feature_selection(classifier):
    classifier.feature_selection(algorithm=chi2, number_of_features=[10, 20])
    for i in classifier._features_index:
        assert len(i) == 4


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


@cleanup_shared_memory
def test_train_test_split_generator(classifier):
    x = create_shared_memory(classifier.features.data, "x")
    y = create_shared_memory(classifier.targets.data, "y")

    data = list(classifier._train_test_split_generator(0.8, x, y))

    assert len(data) == len(classifier._train_data)

    for td in data:
        assert td.train_index is not None
        assert td.test_index is not None


@cleanup_shared_memory
def test_k_fold_generator(classifier, iris_data):
    x, y = iris_data
    x = create_shared_memory(x.values, "x")
    y = create_shared_memory(y, "y")

    folds = list(classifier._k_fold_generator(kf=3, features=x, target=y, shuffle=False))

    assert len(folds) == len(classifier._train_data)

    for fold in folds:
        assert isinstance(fold, KFoldsTrainData)
        assert fold.number_of_folds == 3
        assert fold.shuffle is False

        assert fold.features.shape == x.shape
        assert fold.target.shape == y.shape


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
    results = classifier.k_folds(3, shuffle=False)
    assert len(results) == 2


def test_add_hyper_parameter_create_combinations(classifier):
    train_data = TrainData(model=RandomForestClassifier(),results_type= ClassificationResults)
    classifier._hyper_parameter = {"n_estimators": [10, 20], "max_depth": [2, 4]}
    result = _add_hyper_parameter(classifier._hyper_parameter, train_data)
    assert len(result) == 4
    params = [(td.model.get_params()["n_estimators"], td.model.get_params()["max_depth"]) for td in result]
    assert set(params) == {(10, 2), (10, 4), (20, 2), (20, 4)}
    assert len({id(td.model) for td in result}) == 4  # independent model objects


def test_add_hyper_parameter_ignore_invalid_parameters(classifier):
    train_data = TrainData(model=SVC(),results_type= ClassificationResults)
    classifier._hyper_parameter = {"C": [1, 10], "not_a_parameter": [100, 200]}
    result = _add_hyper_parameter(classifier._hyper_parameter, train_data)
    assert len(result) == 2
    for td in result:
        assert td.model.get_params()["C"] in [1, 10]
        assert "not_a_parameter" not in td.model.get_params()


def test_add_hyper_parameter_no_valid_parameters(classifier):
    train_data = TrainData(model=SVC(),results_type= ClassificationResults)
    classifier._hyper_parameter = {"fake_parameter": [1, 2, 3]}
    result = _add_hyper_parameter(classifier._hyper_parameter, train_data)
    assert len(result) == 1
    returned = result[0]
    assert returned is not train_data
    assert isinstance(returned.model, SVC)
    assert returned.model.get_params() == train_data.model.get_params()


def test_add_hyper_parameter_does_not_change_original_model(classifier):
    train_data = TrainData(model=RandomForestClassifier(n_estimators=50),results_type= ClassificationResults)
    classifier._hyper_parameter = {"n_estimators": [100]}
    result = _add_hyper_parameter(classifier._hyper_parameter, train_data)
    assert len(result) == 1
    assert result[0].model.get_params()["n_estimators"] == 100
    assert train_data.model.get_params()["n_estimators"] == 50


# ---------------------------------------------------------------------------
# error handling and fallbacks
# ---------------------------------------------------------------------------
import numpy as np
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
