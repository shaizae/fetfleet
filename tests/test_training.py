import numpy as np
import pytest
from sklearn.feature_selection import f_classif

from tests.conftest import FEATURE_INDEXES, K_FOLDS, LEAVE_ONE_OUT_FOLDS, RANDOM_STATE, IRIS_SAMPLES, SELECTED_FEATURES, \
    IRIS_FEATURE_COUNT, ALL_FEATURES
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.workers import train_test_split_mc, train_k_folds_mc, features_selections
from ml_pipe.utils import FeaturesSelectionsData, ValidationType, features_unpackage, \
    cleanup_shared_memory
from ml_pipe.train_data import TrainData, KFoldsTrainData


def test_features_unpackage(shared_array):
    shm, original = shared_array

    indexes = [0, 2, 4]
    feature_indexes = [1, 3]

    result = features_unpackage(shm, indexes, feature_indexes)

    expected = original[indexes][..., feature_indexes]

    np.testing.assert_array_equal(result, expected)

@cleanup_shared_memory
def test_train_test_split_mc( logistic_model, train_indexes, test_indexes, iris_data_shared_memory):
    features, target = iris_data_shared_memory

    data = TrainData(model=logistic_model, features=features, target=target, train_index=train_indexes,
                     test_index=test_indexes, features_index=FEATURE_INDEXES,results_type= ClassificationResults)

    result = train_test_split_mc(data)

    assert isinstance(result, ClassificationResults)

    assert result.validation == ValidationType.train_test_split

    assert result._pred_test is not None

@cleanup_shared_memory
@pytest.mark.parametrize("folds,expected_validation",
                         [(K_FOLDS, ValidationType.k_folds), (LEAVE_ONE_OUT_FOLDS, ValidationType.leave_one_out)])
def test_train_k_folds_mc(iris_data_shared_memory, logistic_model, folds, expected_validation):
    features, target = iris_data_shared_memory

    data = KFoldsTrainData(model=logistic_model, features=features, target=target, number_of_folds=folds, shuffle=True,
                           random_state=RANDOM_STATE, features_index=FEATURE_INDEXES,results_type= ClassificationResults)

    result = train_k_folds_mc(data)

    assert isinstance(result, ClassificationResults)

    assert result.validation == expected_validation

    assert result._pred_test is not None

    assert len(result._pred_test) == IRIS_SAMPLES

@cleanup_shared_memory
def test_features_selection(iris_data_shared_memory):
    features, target = iris_data_shared_memory

    data = FeaturesSelectionsData(features=features, target=target, algorithm=f_classif,
                                  number_of_features=SELECTED_FEATURES)

    result = features_selections(data)

    assert isinstance(result, np.ndarray)

    assert len(result) == SELECTED_FEATURES

    assert all(0 <= x < IRIS_FEATURE_COUNT for x in result)

@cleanup_shared_memory
def test_features_selection_when_request_all(iris_data_shared_memory):
    features, target = iris_data_shared_memory

    data = FeaturesSelectionsData(features=features, target=target, algorithm=f_classif,
                                  number_of_features=ALL_FEATURES)

    result = features_selections(data)

    assert result == list(range(IRIS_FEATURE_COUNT))


# ---------------------------------------------------------------------------
# Worker code paths used by the pipeline. They normally run inside pool workers,
# where coverage can't see them, so they are exercised in-process here.
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipe.workers import _fit_predict, _fold_splits, _single_threaded
from ml_pipe.train_data import SelectColumns, build_estimator
from ml_pipe.utils import create_shared_numpy, is_fitted


@pytest.fixture
def iris_shared():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    xs, ys = create_shared_numpy(X, "worker_x"), create_shared_numpy(y, "worker_y")
    yield xs, ys, X, y
    from ml_pipe.utils import SharedMemory
    SharedMemory.cleanup()


def stratified_fold_ids(y, k):
    from sklearn.model_selection import StratifiedKFold
    ids = np.empty(len(y), dtype=np.int64)
    for fold, (_, test) in enumerate(StratifiedKFold(k, shuffle=True, random_state=0).split(y, y)):
        ids[test] = fold
    return ids


def test_k_folds_mc_with_precomputed_folds_and_preprocessing(iris_shared):
    xs, ys, X, y = iris_shared
    folds = create_shared_numpy(stratified_fold_ids(y, 5), "worker_folds")
    td = KFoldsTrainData(model=LogisticRegression(max_iter=500), results_type=ClassificationResults,
                         features=xs, target=ys, features_index=[0, 2], preprocessing=[StandardScaler()],
                         number_of_folds=5, folds=folds, validation=ValidationType.k_folds)
    result = train_k_folds_mc(td)
    assert result.validation == ValidationType.k_folds
    assert len(result._pred_test) == 150 and result._predict_score.shape == (150, 3)
    assert result.features_index == [0, 2]
    assert result.accuracy > 0.85
    assert not is_fitted(result.model)  # metrics only; the parent refits on demand


def test_train_test_split_mc_with_preprocessing(iris_shared):
    xs, ys, X, y = iris_shared
    td = TrainData(model=LogisticRegression(max_iter=500), results_type=ClassificationResults, features=xs,
                   target=ys, train_index=np.arange(0, 150, 2), test_index=np.arange(1, 150, 2),
                   features_index=[0, 1, 2, 3], preprocessing=[StandardScaler()])
    result = train_test_split_mc(td)
    assert len(result._pred_test) == 75
    assert result.accuracy > 0.85


def test_fold_splits_partition_every_row_once(iris_shared):
    xs, ys, X, y = iris_shared
    folds = create_shared_numpy(stratified_fold_ids(y, 5), "worker_folds")
    td = KFoldsTrainData(model=None, results_type=ClassificationResults, target=ys, number_of_folds=5, folds=folds)
    tests = []
    for train, test in _fold_splits(td):
        assert not set(train) & set(test)
        assert len(train) + len(test) == 150
        tests.extend(test)
    assert sorted(tests) == list(range(150))


def test_fit_predict_aligns_probabilities_when_a_class_is_missing(iris_shared):
    xs, ys, X, y = iris_shared
    td = TrainData(model=LogisticRegression(max_iter=500), results_type=ClassificationResults, features=xs,
                   target=ys, features_index=[0, 1, 2, 3])
    train = np.arange(100)  # iris is sorted: rows 0..99 hold only classes 0 and 1
    test = np.arange(100, 150)
    pred, proba = _fit_predict(td, train, test, classes=np.array([0, 1, 2]))
    assert proba.shape == (50, 3)
    assert np.all(proba[:, 2] == 0)  # the unseen class gets probability 0
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_fit_predict_regression_has_no_probabilities():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X @ [1.0, 2.0]
    xs, ys = create_shared_numpy(X, "reg_x"), create_shared_numpy(y, "reg_y")
    try:
        td = TrainData(model=LinearRegression(), results_type=None, features=xs, target=ys, features_index=[0, 1])
        pred, proba = _fit_predict(td, np.arange(8), np.arange(8, 10))
        np.testing.assert_allclose(pred, y[8:])
        assert proba is None
    finally:
        xs.unlink()
        ys.unlink()


def test_single_threaded_overrides_n_jobs_only_when_set():
    assert _single_threaded(RandomForestClassifier(n_jobs=-1)).n_jobs == 1
    assert _single_threaded(RandomForestClassifier()).n_jobs is None
    assert _single_threaded(LinearRegression(n_jobs=4)).n_jobs == 1
    plain = object()
    assert _single_threaded(plain) is plain  # not an estimator: left alone


# ---------------------------------------------------------------------------
# train_data_classes helpers
# ---------------------------------------------------------------------------
def test_select_columns():
    X = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(SelectColumns([3, 1]).fit(X).transform(X), X[:, [3, 1]])
    np.testing.assert_array_equal(SelectColumns(None).fit_transform(X), X)


def test_build_estimator_without_steps_is_the_model():
    model = LinearRegression()
    assert build_estimator(model, [], None) is model


def test_build_estimator_order_is_preprocess_select_model():
    est = build_estimator(LinearRegression(), [StandardScaler()], [0, 2])
    assert isinstance(est, Pipeline)
    assert [name for name, _ in est.steps] == ["pre0", "select", "model"]


def test_build_estimator_clones_preprocessing():
    scaler = StandardScaler()
    est = build_estimator(LinearRegression(), [scaler], None)
    est.fit(np.ones((3, 2)), [1, 2, 3])
    assert not hasattr(scaler, "mean_")  # the caller's step stays unfitted


def test_task_cost_orders_heavier_tasks_first():
    small = TrainData(RandomForestClassifier(n_estimators=10), ClassificationResults, features_index=[0])
    big = TrainData(RandomForestClassifier(n_estimators=100), ClassificationResults, features_index=[0, 1, 2])
    no_trees = TrainData(LogisticRegression(), ClassificationResults, features_index=[0, 1])
    assert big.cost() > small.cost() > 0
    assert no_trees.cost() == 2
    kfold = KFoldsTrainData(RandomForestClassifier(n_estimators=10), ClassificationResults, features_index=[0],
                            number_of_folds=5)
    assert kfold.cost() == 5 * small.cost()
