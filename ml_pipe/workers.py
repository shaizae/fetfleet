import warnings

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from ml_pipe.utils import force_gc, ValidationType, features_unpackage
from ml_pipe.train_data import TrainData, KFoldsTrainData, build_estimator

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)


def init_worker():
    """Pool initializer: the pool already runs one process per core, so stop
    BLAS/OpenMP inside each worker from starting its own threads on top."""
    threadpool_limits(limits=1)


def _single_threaded(model):
    """Force n_jobs=1 on the worker's copy (joblib threads ignore threadpoolctl)."""
    if hasattr(model, "get_params") and model.get_params(deep=False).get("n_jobs") not in (None, 1):
        model.set_params(n_jobs=1)
    return model


def _strip_fitted_model(result, original_model, features_index, candidate_id=None):
    """Ship back metrics and predictions only; the fitted model is the bulk of the
    pickle. BaseML.refit() rebuilds it in the parent for the results you keep."""
    result.model = original_model
    result.features_index = features_index
    result.candidate_id = candidate_id
    return result


def _fit_predict(train_data: TrainData, train_index, test_index, classes=None):
    """Fit a fresh copy of the model on the training rows, predict the test rows.

    Without preprocessing the columns are cut straight out of shared memory (one copy).
    With preprocessing the raw rows go through a Pipeline that fits the scalers on the
    training rows only, then selects columns of the transformed matrix.
    """
    model = _single_threaded(clone(train_data.model, safe=False))
    if train_data.preprocessing:
        estimator = build_estimator(model, train_data.preprocessing, train_data.features_index)
        x_train = features_unpackage(train_data.features, train_index)
        x_test = features_unpackage(train_data.features, test_index)
    else:
        estimator = model
        x_train = features_unpackage(train_data.features, train_index, train_data.features_index)
        x_test = features_unpackage(train_data.features, test_index, train_data.features_index)

    y = train_data.target.array
    estimator.fit(x_train, y[train_index])
    pred = estimator.predict(x_test)

    proba = None
    if classes is not None and hasattr(estimator, "predict_proba"):
        fold_proba = estimator.predict_proba(x_test)
        # A fold's training rows may miss a class: align columns to all classes.
        proba = np.zeros((len(test_index), len(classes)))
        proba[:, np.searchsorted(classes, estimator.classes_)] = fold_proba
    return pred, proba


def _is_classification(train_data: TrainData) -> bool:
    return getattr(train_data.results_type, "is_classification", False)


@force_gc
def train_test_split_mc(train_data: TrainData):
    classes = np.unique(train_data.target.array) if _is_classification(train_data) else None
    pred, proba = _fit_predict(train_data, train_data.train_index, train_data.test_index, classes)

    result = train_data.results_type(clone(train_data.model, safe=False))
    result.evaluate(train_data.target.array[train_data.test_index], pred, proba)
    result.validation = ValidationType.train_test_split
    return _strip_fitted_model(result, result.model, train_data.features_index, train_data.candidate_id)


def _fold_splits(train_data: KFoldsTrainData):
    """(train, test) row indexes per fold."""
    if train_data.folds is None:  # direct call without precomputed folds
        kf = KFold(n_splits=train_data.number_of_folds, shuffle=train_data.shuffle,
                   random_state=train_data.random_state if train_data.shuffle else None)
        yield from kf.split(np.empty(train_data.target.shape[0]))
        return
    folds = train_data.folds.array
    order = np.argsort(folds, kind="stable")
    # rows with fold id -1 (left out of a halving round) sort first and are skipped
    bounds = np.searchsorted(folds[order], np.arange(train_data.number_of_folds + 1))
    for k in range(train_data.number_of_folds):
        test = order[bounds[k]:bounds[k + 1]]
        train = np.concatenate([order[bounds[0]:bounds[k]], order[bounds[k + 1]:]])
        yield train, test


@force_gc
def train_k_folds_mc(train_data: KFoldsTrainData):
    """Each fold's own model predicts that fold's test rows; the out-of-fold
    predictions are concatenated once and scored together."""
    classes = np.unique(train_data.target.array) if _is_classification(train_data) else None
    y = train_data.target.array
    preds, truths, probas = [], [], []
    for train_index, test_index in _fold_splits(train_data):
        pred, proba = _fit_predict(train_data, train_index, test_index, classes)
        preds.append(pred)
        truths.append(y[test_index])
        if proba is not None:
            probas.append(proba)

    result = train_data.results_type(clone(train_data.model, safe=False))
    result.evaluate(np.concatenate(truths), np.concatenate(preds),
                    np.concatenate(probas) if probas else None)

    if train_data.validation is not None:
        result.validation = train_data.validation
    elif train_data.number_of_folds == train_data.target.shape[0]:
        result.validation = ValidationType.leave_one_out
    else:
        result.validation = ValidationType.k_folds
    return _strip_fitted_model(result, result.model, train_data.features_index, train_data.candidate_id)


def select_k_best_many(algorithm, features: np.ndarray, target: np.ndarray, ks) -> list:
    """Top-k feature indexes for every k in `ks` from a single scoring pass.

    SelectKBest's scores don't depend on k, so score once and re-read the support
    mask per k (same tie-breaking as fitting SelectKBest per k).
    """
    n_features = features.shape[1]
    select = SelectKBest(algorithm, k="all").fit(features, target)
    out, seen = [], set()
    for k in ks:
        if k >= n_features:
            idx = list(range(n_features))
        else:
            select.k = k
            idx = np.where(select.get_support())[0]
        key = tuple(int(i) for i in idx)
        if key not in seen:  # k values above the feature count all give the full set
            seen.add(key)
            out.append(idx)
    return out
