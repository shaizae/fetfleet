import os
from multiprocessing.pool import Pool
from typing import Any, Generator, Iterable

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from external_classes.BaseML import BaseML, _add_hyper_parameter
from results.RegressionResults import RegressionResults
from utils.Fetchers import Fetchers
from utils.Target import Target
from utils.multiprocess_functions import init_worker, train_k_folds_mc, train_test_split_mc
from utils.utils import SharedMemory, FilteringCriteriaClassification, \
    cleanup_shared_memory, \
    create_shared_numpy, FilteringCriteriaRegression
from utils.train_data_classes import TrainData, KFoldsTrainData


class Regression(BaseML):
    # _process_limit and random_state come from BaseML; redefining _process_limit here
    # shadowed it, so BaseML.process_limit() silently had no effect on Regression.

    def __init__(self):
        super().__init__()
        self._results: list[RegressionResults] = None


    def set(self, fetchers: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator], ):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)

        self.targets = Target()
        self.targets.load_new_data(target)

        self._features_index = [list(range(self.fetchers.data.shape[1]))]
        self._train_data = [TrainData(model,RegressionResults) for model in models]

    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

        X = create_shared_numpy(self.fetchers.data, "features")
        y = create_shared_numpy(self.targets.data, "target")

        generator = self._train_test_split_generator(ratio, X, y)

        with Pool(processes=self._process_limit, initializer=init_worker, maxtasksperchild=20, ) as pool:
            self._results = list(
                tqdm(pool.imap_unordered(train_test_split_mc, generator, ), desc="training", ))
        return self._results

    def _train_test_split_generator(self, ratio: float, features: SharedMemory, target: SharedMemory) -> Generator[
        TrainData]:

        train_idx, test_idx = train_test_split(np.arange(len(self.fetchers.data)), test_size=ratio,
                                               random_state=Regression.random_state, )
        self._train_index = train_idx
        for td in self._build_training_list(features, target):
            td.set_indexes(train_idx, test_idx)
            yield td

    @cleanup_shared_memory
    def k_folds(            self,            n_splits: int = 5,            shuffle: bool = True,    ):
        X = create_shared_numpy(self.fetchers.data, "features")
        y = create_shared_numpy(self.targets.data, "target")

        generator = self._k_fold_generator(n_splits, X, y, shuffle, )

        with Pool(processes=self._process_limit, initializer=init_worker, maxtasksperchild=20, ) as pool:
            self._results = list(tqdm(pool.imap_unordered(train_k_folds_mc, generator, ), desc="training", ))

        return self._results

    def leave_one_out(self):
        return self.k_folds(len(self.targets.data))

    def _k_fold_generator(self, k: int, X: SharedMemory, y: SharedMemory, shuffle: bool, ):
        seed = self._shared_seed(shuffle, Regression.random_state)
        for td in self._build_training_list(X, y):
            yield KFoldsTrainData.set_from_train_data(td, number_of_folds=k, shuffle=shuffle,
                                                      randon_state=seed, )

    def _build_training_list(self, X: SharedMemory, y: SharedMemory, ) -> Iterable[TrainData]:

        for base in self._train_data:
            for features in self._features_index:
                td = base.copy()
                td.set_features_and_targets(X, y)
                td.set_features_selection(features)

                for result in _add_hyper_parameter(self._hyper_parameter, td):
                    yield result

    def filter_by(self, criteria: FilteringCriteriaRegression):
        if criteria not in FilteringCriteriaRegression:
            raise ValueError("criteria must be one of {}".format(list(FilteringCriteriaRegression)))
        return super().filter_by(criteria)
