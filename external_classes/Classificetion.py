import os
from multiprocessing.pool import Pool
from typing import Generator, Iterable, Any

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from external_classes.BaseML import BaseML,_brut_force, _add_hyper_parameter
from results.ClassificationResults import ClassificationResults
from utils.Fetchers import Fetchers
from utils.Target import Target
from utils.multiprocess_functions import init_worker, train_test_split_mc, features_selections, train_k_folds_mc
from utils.utils import create_shared_numpy, FeaturesSelectionsData, SharedMemory, \
    FilteringCriteriaClassification, \
    cleanup_shared_memory
from utils.train_data_classes import TrainData, KFoldsTrainData


class Classification(BaseML):


    def __init__(self):
        super().__init__()
        self._results: list[ClassificationResults] = None



    def set(self, fetchers: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator]):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)
        self._features_index = []
        self._features_index.append(list(range(self.fetchers.data.shape[1])))

        self.targets = Target()
        self.targets.load_new_data(target)

        self._train_data = [TrainData(model, ClassificationResults) for model in models]

    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        if ratio < 0:
            raise ValueError('ratio must be greater than 0')
        if ratio > 1:
            raise ValueError('ratio must be less than 1')
        target = create_shared_numpy(self.targets.data, "target")
        features = create_shared_numpy(self.fetchers.data, "features")
        train_data = self._train_test_split_generator(ratio, features, target)

        try:
            with Pool(processes=self._process_limit, initializer=init_worker, maxtasksperchild=20) as pool:
                self._results = list(
                    tqdm(pool.imap_unordered(train_test_split_mc, train_data), desc="training"))

        except Exception as e:
            print(f"training fail error={e}")
            raise e

        return self._results

    def _train_test_split_generator(self, ratio: float, features: SharedMemory, target: SharedMemory) -> Generator[
        TrainData]:
        train_index, test_index = train_test_split(
            np.arange(len(self.fetchers.data)), test_size=ratio, random_state=Classification.random_state)
        self._train_index = train_index

        for train_data in self._build_training_list(features, target):
            train_data.set_indexes(train_index, test_index)
            yield train_data

    @cleanup_shared_memory
    def k_folds(self, n_splits: int = 5, shuffle: bool = True):
        if n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")

        if n_splits > len(self.targets.data):
            raise ValueError("n_splits must be less than dataset size")

        X_train = create_shared_numpy(self.fetchers.data, f"features")
        y_train = create_shared_numpy(self.targets.data, f"target")
        train_inputs = self._k_fold_generator(n_splits, X_train, y_train, shuffle)
        try:
            with Pool(processes=self._process_limit, initializer=init_worker, maxtasksperchild=20) as pool:
                self._results = list(
                    tqdm(pool.imap_unordered(train_k_folds_mc, train_inputs), desc="training"))
        except Exception as e:
            print(f"training fail error={e}")
            raise e
        return self._results

    def _k_fold_generator(self, kf: int, features: SharedMemory, target: SharedMemory, shuffle: bool):
        seed = self._shared_seed(shuffle, Classification.random_state)
        for train_data in self._build_training_list(features, target):
            yield KFoldsTrainData.set_from_train_data(train_data, number_of_folds=kf, shuffle=shuffle,
                                                      randon_state=seed)

    def _build_training_list(self, X_train: SharedMemory, y_train: SharedMemory) -> Iterable[TrainData]:
        for train_data_class in self._train_data:
            for indexes in self._features_index:
                train_data = train_data_class.copy()
                train_data.set_features_and_targets(X_train, y_train)
                train_data.set_features_selection(indexes)
                with_params = _add_hyper_parameter(self._hyper_parameter, train_data)
                for params in with_params:
                    yield params

    def leave_one_out(self):
        return self.k_folds(n_splits=len(self.targets.data))

    @cleanup_shared_memory
    def fetcher_selection(self, algorithm: BaseEstimator, number_of_features: list[int]):
        _, train_index = train_test_split(np.arange(len(self.fetchers.data)), test_size=0.1)

        fetchers = create_shared_numpy(self.fetchers.pop_index(train_index), "fetchers")
        target = create_shared_numpy(self.targets.pop_index(train_index), "target")

        features_selections_data = Classification.fetcher_selection_generator(algorithm, number_of_features, fetchers,
                                                                              target)
        try:
            with Pool(processes=self._process_limit, initializer=init_worker, maxtasksperchild=20) as pool:
                results = list(tqdm(pool.imap_unordered(features_selections, features_selections_data),
                                    total=len(number_of_features), desc="fetcher selection"))
        except Exception as e:
            print(f"fetcher selection fail error={e}")
            raise e
        self._features_index = results

    @staticmethod
    def fetcher_selection_generator(algorithm: BaseEstimator, number_of_features: list[int],
                                    features: SharedMemory, target: SharedMemory):
        for number in number_of_features:
            yield FeaturesSelectionsData(algorithm=algorithm, features=features, target=target,
                                         number_of_features=number)

    def brut_force_features(self):
        number_of_features = range(self.fetchers.data.shape[1])
        vec = np.array(number_of_features)
        self._features_index = list(_brut_force(vec))

    @property
    def results(self):
        return self._results

    def filter_by(self, criteria: FilteringCriteriaClassification):
        if criteria not in FilteringCriteriaClassification:
            raise ValueError("criteria must be one of {}".format(list(FilteringCriteriaClassification)))
        return super().filter_by(criteria)


