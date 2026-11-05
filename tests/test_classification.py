import os
from itertools import combinations
from unittest.mock import patch

import pytest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from external_classes.BaseML import _add_hyper_parameter
from external_classes.Classificetion import Classification
from results.ClassificationResults import ClassificationResults
from utils.utils import create_shared_numpy, cleanup_shared_memory
from utils.train_data_classes import TrainData, KFoldsTrainData


def test_set_initializes_data(classifier):
    assert classifier.fetchers is not None
    assert classifier.targets is not None
    assert len(classifier._train_data) == 2


def test_process_limit_valid():
    Classification.process_limit(1)

    assert Classification._process_limit == 1


def test_process_limit_cannot_exceed_cpu_count():
    Classification.process_limit(os.cpu_count() + 100)

    assert Classification._process_limit == os.cpu_count()


def test_process_limit_negative():
    with pytest.raises(ValueError):
        Classification.process_limit(-1)


@pytest.mark.parametrize("ratio", [-0.1, -1])
def test_train_test_split_negative_ratio(classifier, ratio):
    with pytest.raises(ValueError):
        classifier.train_test_split(ratio)


@pytest.mark.parametrize("ratio", [1.1, 2])
def test_train_test_split_ratio_too_large(classifier, ratio):
    with pytest.raises(ValueError):
        classifier.train_test_split(ratio)


def test_train_test_split_runs(classifier):
    results = classifier.train_test_split(0.8)
    assert len(results) == 2


def test_k_folds_invalid_splits_low(classifier):
    with pytest.raises(ValueError):
        classifier.k_folds(1)


def test_k_folds_invalid_splits_too_large(classifier):
    dataset_size = len(classifier.targets.data)

    with pytest.raises(ValueError):
        classifier.k_folds(dataset_size + 1)


def test_k_folds_runs(classifier):
    results = classifier.k_folds(3)
    assert len(results) == 2


def test_leave_one_out_runs(classifier):
    results = classifier.leave_one_out()

    assert len(results) == 2


def test_fetcher_selection_updates_train_data(classifier):
    number_of_features = [1, 2]
    original_len = len(classifier._train_data)
    classifier.fetcher_selection(algorithm=chi2, number_of_features=number_of_features)
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

    assert cls.fetchers is not None
    assert cls.targets is not None
    assert len(cls._train_data) == 2


def test_use_more_fetchers_that_exist_in_fetcher_selection(classifier):
    classifier.fetcher_selection(algorithm=chi2, number_of_features=[10, 20])
    for i in classifier._features_index:
        assert len(i) == 4


def test_brut_force_features(classifier):
    original_models = len(classifier._features_index)
    n_features = classifier.fetchers.data.shape[1]
    # every non-empty subset, including the full feature set
    expected_combinations = [combo for i in range(1, n_features + 1) for combo in combinations(range(n_features), i)]
    classifier.brut_force_features()
    assert len(classifier._features_index) == (original_models * len(expected_combinations))
    produced = [tuple(td) for td in classifier._features_index]
    for _ in range(original_models):
        for combo in expected_combinations:
            assert produced.count(combo) == original_models


@cleanup_shared_memory
def test_train_test_split_generator(classifier):
    x = create_shared_numpy(classifier.fetchers.data, "x")
    y = create_shared_numpy(classifier.targets.data, "y")

    data = list(classifier._train_test_split_generator(0.8, x, y))

    assert len(data) == len(classifier._train_data)

    for td in data:
        assert td.train_index is not None
        assert td.test_index is not None


@cleanup_shared_memory
def test_k_fold_generator(classifier, iris_data):
    x, y = iris_data
    x = create_shared_numpy(x.values, "x")
    y = create_shared_numpy(y, "y")

    folds = list(classifier._k_fold_generator(kf=3, features=x, target=y, shuffle=False))

    assert len(folds) == len(classifier._train_data)

    for fold in folds:
        assert isinstance(fold, KFoldsTrainData)
        assert fold.number_of_folds == 3
        assert fold.shuffle is False

        assert fold.features.shape == x.shape
        assert fold.target.shape == y.shape


def test_train_test_split_exception(classifier):
    with patch(
            "external_classes.BaseML.Pool",
            side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            classifier.train_test_split()


def test_k_folds_exception(classifier):
    with patch("external_classes.BaseML.Pool", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            classifier.k_folds(3)


def test_fetcher_selection_does_not_need_the_pool(classifier):
    # feature selection scores once in-process; it must not start worker processes
    with patch("external_classes.BaseML.Pool", side_effect=RuntimeError("boom"), ):
        classifier.fetcher_selection(algorithm=chi2, number_of_features=[1, 2], )
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


def test_add_hyper_parameter_multiple_models_shared_parameters(classifier):
    train_data_list = [
        TrainData(model=RandomForestClassifier(),results_type= ClassificationResults),
        TrainData(model=AdaBoostClassifier(),results_type= ClassificationResults),
        TrainData(model=SVC(),results_type= ClassificationResults)
    ]

    classifier._hyper_parameter = {
        "n_estimators": [10, 20],  # RandomForest + AdaBoost
        "C": [1, 10]  # SVC only
    }

    results = []

    for train_data in train_data_list:
        results.extend(
            _add_hyper_parameter(classifier._hyper_parameter, train_data)
        )

    # RF -> 2
    # AdaBoost -> 2
    # SVC -> 2
    assert len(results) == 6

    rf_results = [td for td in results if isinstance(td.model, RandomForestClassifier)]

    ada_results = [td for td in results if isinstance(td.model, AdaBoostClassifier)]

    svc_results = [td for td in results if isinstance(td.model, SVC)]

    assert len(rf_results) == 2
    assert len(ada_results) == 2
    assert len(svc_results) == 2

    # Shared parameter check
    assert {td.model.get_params()["n_estimators"] for td in rf_results} == {10, 20}

    assert {td.model.get_params()["n_estimators"] for td in ada_results} == {10, 20}

    # SVC parameter check
    assert {td.model.get_params()["C"] for td in svc_results} == {1, 10}

    # Make sure models are independent objects
    assert rf_results[0].model is not rf_results[1].model
    assert ada_results[0].model is not ada_results[1].model
    assert svc_results[0].model is not svc_results[1].model
