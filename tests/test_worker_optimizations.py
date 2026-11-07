import copy
import pickle

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_pipe.classification import Classification
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.train_data import TrainData
from ml_pipe.utils import SharedMemory, create_shared_numpy, FilteringCriteriaClassification, is_fitted


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    SharedMemory.cleanup()


def test_shared_memory_pickles_without_data():
    big = np.zeros((10_000, 50))
    shm = create_shared_numpy(big, "pickle_test")
    assert len(pickle.dumps(shm)) < 1_000  # name/shape/dtype only, not 4 MB of data


def test_shared_memory_deepcopy_is_identity():
    shm = create_shared_numpy(np.arange(4), "deepcopy_test")
    assert copy.deepcopy(shm) is shm


def test_train_data_copy_clones_model_and_shares_memory():
    shm = create_shared_numpy(np.arange(4), "copy_test")
    model = RandomForestClassifier(n_estimators=7).fit([[0], [1]], [0, 1])
    td = TrainData(model, ClassificationResults, features=shm)
    copied = td.copy()
    assert copied.features is shm
    assert copied.model is not model
    assert copied.model.get_params() == model.get_params()
    assert not is_fitted(copied.model)


def test_results_come_back_without_fitted_model(classifier):
    results = classifier.k_folds(3)
    assert all(not is_fitted(r.model) for r in results)
    assert all(r.features_index is not None for r in results)


@pytest.mark.parametrize("run", ["k_folds", "train_test_split"])
def test_filter_by_refits_best(classifier, run):
    getattr(classifier, run)()
    best = classifier.filter_by(FilteringCriteriaClassification.accuracy)
    assert best and all(is_fitted(r.model) for r in best)


def test_all_tasks_share_one_fold_seed(classifier):
    x = create_shared_numpy(classifier.features.data, "seed_x")
    y = create_shared_numpy(classifier.targets.data, "seed_y")
    Classification.random_state = None
    tasks = list(classifier._k_fold_generator(kf=3, features=x, target=y, shuffle=True))
    assert len({t.random_state for t in tasks}) == 1
