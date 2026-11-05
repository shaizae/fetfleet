import warnings

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from results.ClassificationResults import ClassificationResults
from utils.utils import FeaturesSelectionsData, SharedMemory, force_gc, ValidationType
from utils.train_data_classes import TrainData, KFoldsTrainData

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


def _strip_fitted_model(result, original_model, features_index):
    """Ship back metrics and predictions only; the fitted model is the bulk of the
    pickle. BaseML.refit() rebuilds it in the parent for the results you keep."""
    result.model = original_model
    result.features_index = features_index
    return result


def features_unpackage(shared_memory: SharedMemory, index: list[int], features_index: list[int]):
    return shared_memory.array[index][..., features_index]


@force_gc
def train_test_split_mc(train_data: TrainData):
    original_model = clone(train_data.model, safe=False)
    _single_threaded(train_data.model)
    train_data.fit(features_unpackage(train_data.features, train_data.train_index, train_data.featuresIndex),
                   train_data.target.array[train_data.train_index])

    result = train_data.results_type(train_data.model)
    result.set_test(features_unpackage(train_data.features, train_data.test_index, train_data.featuresIndex),
                    train_data.target.array[train_data.test_index])
    result.predict()
    result.validation = ValidationType.train_test_split
    return _strip_fitted_model(result, original_model, train_data.featuresIndex)


@force_gc
def train_k_folds_mc(train_data: KFoldsTrainData):
    original_model = clone(train_data.model, safe=False)
    _single_threaded(train_data.model)
    kf = KFold(n_splits=train_data.number_of_folds, shuffle=train_data.shuffle, random_state=train_data.randon_state)
    final_result: ClassificationResults = None
    is_loo = False
    if train_data.number_of_folds == train_data.target.shape[0]:
        is_loo = True
    for train_index, test_index in kf.split(train_data.features.array):
        fold_data = train_data.copy()
        fold_data.set_indexes(train_index, test_index)

        result = train_data.results_type(
            fold_data.fit(features_unpackage(fold_data.features, fold_data.train_index, fold_data.featuresIndex),
                          fold_data.target.array[fold_data.train_index]))
        result.set_test(features_unpackage(fold_data.features, fold_data.test_index, fold_data.featuresIndex),
                        fold_data.target.array[fold_data.test_index])

        if final_result is None:
            final_result = result
        else:
            final_result.append_results(result)
    final_result.predict()

    if is_loo:
        final_result.validation = ValidationType.leave_one_out
    else:
        final_result.validation = ValidationType.k_folds
    return _strip_fitted_model(final_result, original_model, train_data.featuresIndex)


@force_gc
def features_selections(features_selections_data: FeaturesSelectionsData):
    features = features_selections_data.features.array
    target = features_selections_data.target.array
    if features_selections_data.number_of_features >= features.shape[1]:
        return list(range(features.shape[1]))
    algorithm = features_selections_data.algorithm
    select = SelectKBest(algorithm, k=features_selections_data.number_of_features)
    select.fit(features, target)
    return np.where(select.get_support())[0]
