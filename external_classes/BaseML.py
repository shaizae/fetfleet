import os
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from itertools import combinations, product
from multiprocessing.pool import Pool
from typing import Any, Iterable

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm import tqdm

from utils.Fetchers import Fetchers
from utils.Target import Target
from utils.multiprocess_functions import init_worker, train_k_folds_mc, train_test_split_mc, select_k_best_many
from utils.train_data_classes import TrainData, KFoldsTrainData, build_estimator
from utils.utils import ValidationType, is_fitted, create_shared_numpy, cleanup_shared_memory, run_block_name, \
    SharedMemory


class BaseML(ABC):
    """Runs every model x feature subset x hyper-parameter combination in a process pool.

    The pool is created on first use and reused by later runs on the same object
    (starting worker processes costs seconds on Windows). Call close(), or use the
    object as a context manager, to stop the workers early; otherwise they stop when
    the object is garbage-collected or the interpreter exits.
    """
    _process_limit: int = max(1, os.cpu_count() // 2)
    random_state = None
    # Restart each worker after this many tasks (None = never). Restarting re-imports
    # sklearn (~1 s per worker on Windows); only set it if workers leak memory.
    max_tasks_per_child: int | None = None
    # brut_force_features() refuses more subsets than this (2^n - 1 grows fast).
    max_feature_combinations: int = 50_000
    # e.g. np.float32: halves shared memory and copy time; may change results slightly.
    features_dtype = None

    # set by subclasses
    results_type = None
    criteria_type = None
    _encode_target = True

    def __init__(self):
        self._features_index: list[list[int]] = None
        self._all_features = True  # use every column of the (transformed) data
        self._hyper_parameter: dict[str, list[Any]] = {}
        self.fetchers = None
        self.targets = None
        self._train_data = None
        self._train_index = None  # rows used for training by the last train_test_split
        self._results = None
        self._run_id = None
        self._pool = None
        self._pool_size = None
        self._pool_finalizer = None

    # ------------------------------------------------------------------ setup
    def set(self, fetchers: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator]):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)

        self.targets = Target(encode=self._encode_target)
        self.targets.load_new_data(target)

        self._features_index = [list(range(self.fetchers.data.shape[1]))]
        self._all_features = True
        self._train_data = [TrainData(model, self.results_type) for model in models]

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 1:
            raise ValueError("limit must be greater than 0")
        BaseML._process_limit = min(new_limit, os.cpu_count())

    def set_hyper_parameters(self, hyper_parameter: dict[str, list[Any]]):
        self._hyper_parameter = hyper_parameter

    @property
    def results(self):
        return self._results

    # ------------------------------------------------------------------ pool
    def _get_pool(self) -> Pool:
        if self._pool is not None and self._pool_size == self._process_limit:
            return self._pool
        self.close()  # process limit changed: start a pool of the new size
        pool = Pool(processes=self._process_limit, initializer=init_worker,
                    maxtasksperchild=self.max_tasks_per_child)
        self._pool, self._pool_size = pool, self._process_limit
        self._pool_finalizer = weakref.finalize(self, pool.terminate)
        return pool

    def close(self):
        """Stop the worker processes."""
        if self._pool is None:
            return
        self._pool_finalizer.detach()
        self._pool.terminate()
        self._pool.join()
        self._pool = self._pool_size = self._pool_finalizer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self, func, tasks: list, desc: str) -> list:
        # Longest tasks first, so one slow model doesn't start last and leave the
        # other workers idle at the end.
        tasks = sorted(tasks, key=lambda task: task.cost(), reverse=True)
        pool = self._get_pool()
        # Batch small tasks to cut inter-process round trips; keep ~4 batches per worker
        # so the load still balances.
        chunksize = max(1, len(tasks) // (self._pool_size * 4))
        try:
            return list(tqdm(pool.imap_unordered(func, tasks, chunksize=chunksize),
                             total=len(tasks), desc=desc))
        except BaseException:
            self.close()  # a worker may be stuck mid-task: start clean next run
            raise

    # ------------------------------------------------------------------ data
    def _start_run(self) -> str:
        self._run_id = uuid.uuid4().hex[:12]
        return self._run_id

    def _share(self, array: np.ndarray, kind: str) -> SharedMemory:
        return create_shared_numpy(array, run_block_name(self._run_id or self._start_run(), kind))

    def _training_features(self) -> np.ndarray:
        """Raw features: recorded preprocessing is refitted inside every split."""
        features = self.fetchers.raw_data
        if self.features_dtype is not None:
            features = features.astype(self.features_dtype, copy=False)
        return np.ascontiguousarray(features)

    def _feature_sets(self) -> list:
        if self._all_features:
            return [list(range(self.fetchers.data.shape[1]))]
        return self._features_index

    def _build_training_list(self, X: SharedMemory, y: SharedMemory) -> Iterable[TrainData]:
        preprocessing = self.fetchers.preprocessing
        for base in self._train_data:
            for indexes in self._feature_sets():
                td = base.copy()
                td.set_features_and_targets(X, y)
                td.set_features_selection(indexes)
                td.preprocessing = preprocessing
                yield from _add_hyper_parameter(self._hyper_parameter, td)

    # ------------------------------------------------------------------ splitting
    def _stratify_labels(self, y: np.ndarray):
        """Labels to stratify on, or None (regression)."""
        return None

    @abstractmethod
    def _make_splitter(self, n_splits: int, shuffle: bool, seed):
        ...

    @staticmethod
    def _shared_seed(shuffle: bool, random_state):
        """One seed for all tasks, so every model is scored on the same folds."""
        if not shuffle:
            return None
        return random_state if random_state is not None else int(np.random.randint(2 ** 31 - 1))

    def _split_rows(self, ratio: float, y: np.ndarray):
        rows = np.arange(len(y))
        stratify = self._stratify_labels(y)
        try:
            return sk_train_test_split(rows, train_size=ratio, random_state=self.random_state, stratify=stratify)
        except ValueError:
            if stratify is None:
                raise
            # a class too small to appear on both sides: fall back to a plain split
            return sk_train_test_split(rows, train_size=ratio, random_state=self.random_state)

    def _fold_ids(self, n_splits: int, shuffle: bool, seed, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if n_splits == n:  # leave-one-out: every row is its own fold
            return np.arange(n)
        ids = np.empty(n, dtype=np.int64)
        splitter = self._make_splitter(n_splits, shuffle, seed)
        for fold, (_, test) in enumerate(splitter.split(np.zeros(n), y)):
            ids[test] = fold
        return ids

    # ------------------------------------------------------------------ runs
    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        """ratio = fraction of rows used for training (0.8 -> 80% train, 20% test)."""
        if not 0 < ratio < 1:
            raise ValueError("ratio (fraction of rows used for training) must be between 0 and 1")
        self._start_run()
        X = self._share(self._training_features(), "features")
        y = self._share(self.targets.data, "target")
        tasks = list(self._train_test_split_generator(ratio, X, y))
        self._results = self._run(train_test_split_mc, tasks, "training")
        return self._results

    def _train_test_split_generator(self, ratio: float, features: SharedMemory, target: SharedMemory):
        train_index, test_index = self._split_rows(ratio, target.array)
        self._train_index = train_index
        for td in self._build_training_list(features, target):
            td.set_indexes(train_index, test_index)
            yield td

    @cleanup_shared_memory
    def k_folds(self, n_splits: int = 5, shuffle: bool = True):
        if n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")
        if n_splits > len(self.targets.data):
            raise ValueError("n_splits must be less than dataset size")
        self._start_run()
        X = self._share(self._training_features(), "features")
        y = self._share(self.targets.data, "target")
        tasks = list(self._k_fold_generator(n_splits, X, y, shuffle))
        self._results = self._run(train_k_folds_mc, tasks, "training")
        return self._results

    def leave_one_out(self):
        return self.k_folds(n_splits=len(self.targets.data))

    def _k_fold_generator(self, kf: int, features: SharedMemory, target: SharedMemory, shuffle: bool):
        y = target.array
        seed = self._shared_seed(shuffle, self.random_state)
        # Folds are computed once here, so every model is scored on identical folds.
        folds = self._share(self._fold_ids(kf, shuffle, seed, y), "folds")
        validation = ValidationType.leave_one_out if kf == len(y) else ValidationType.k_folds
        for td in self._build_training_list(features, target):
            yield KFoldsTrainData.set_from_train_data(td, number_of_folds=kf, shuffle=shuffle, randon_state=seed,
                                                      folds=folds, validation=validation)

    # ------------------------------------------------------------------ features
    def fetcher_selection(self, algorithm, number_of_features: list[int]):
        """Choose feature subsets with a SelectKBest score function (chi2, f_classif,
        f_regression, ...). 10% of the rows are used for the selection and then removed,
        so they never take part in training or testing."""
        rows = np.arange(len(self.targets.data))
        _, held_out = sk_train_test_split(rows, test_size=0.1, random_state=self.random_state)
        features = self.fetchers.pop_index(held_out)
        target = self.targets.pop_index(held_out)
        # One scoring pass serves every k; no process pool needed.
        self._features_index = select_k_best_many(algorithm, features, target, number_of_features)
        self._all_features = False

    def brut_force_features(self):
        """Every non-empty subset of the columns, including all of them."""
        n = self.fetchers.data.shape[1]
        total = 2 ** n - 1
        if total > self.max_feature_combinations:
            raise ValueError(f"{n} features give {total:,} subsets (limit {self.max_feature_combinations:,}). "
                             f"Use fetcher_selection(), or raise {type(self).__name__}.max_feature_combinations.")
        self._features_index = [[int(i) for i in combo] for combo in _brut_force(np.arange(n))]
        self._all_features = False

    # ------------------------------------------------------------------ results
    def filter_by(self, criteria):
        if self.criteria_type is not None and criteria not in self.criteria_type:
            raise ValueError("criteria must be one of {}".format(list(self.criteria_type)))
        values = [getattr(result, criteria.value) for result in self._results]

        best_value = max(values) if criteria.maximize else min(values)

        best = [result for result in self._results if getattr(result, criteria.value) == best_value]
        return self.refit(best)

    def refit(self, results):
        """Fit the model of each result in this process so it can be saved or used.

        Workers return only metrics and predictions. train_test_split results are
        refitted on the same training rows; k-fold / leave-one-out results on all rows.
        With preprocessing or a feature subset the fitted model is a Pipeline that takes
        raw rows with all columns, so it can be used on new data as is.
        """
        X, y = self._training_features(), self.targets.data
        preprocessing = self.fetchers.preprocessing
        n_columns = self.fetchers.data.shape[1]
        for result in results:
            if is_fitted(result.model):
                continue
            rows = self._train_index if result.validation == ValidationType.train_test_split else slice(None)
            columns = result.features_index
            if columns is not None and list(columns) == list(range(n_columns)):
                columns = None  # all columns: no selection step needed
            estimator = build_estimator(result.model, preprocessing, columns)
            estimator.fit(X[rows], y[rows])
            result.model = estimator
        return results


def _brut_force(vec: np.ndarray) -> Iterable[tuple]:
    for i in range(1, len(vec) + 1):
        for combo in combinations(vec, i):
            yield combo


def _add_hyper_parameter(_hyper_parameter, data: TrainData) -> list[TrainData]:
    train_data = data.copy()
    model_params = train_data.get_model_params()
    valid_params = {k: v for k, v in _hyper_parameter.items() if k in model_params}
    if not valid_params:
        return [train_data]
    keys = list(valid_params.keys())
    new_train_data = []
    for values in product(*(valid_params[k] for k in keys)):
        td = train_data.copy()
        params = dict(zip(keys, values))
        td.model.set_params(**params)
        new_train_data.append(td)
    return new_train_data
