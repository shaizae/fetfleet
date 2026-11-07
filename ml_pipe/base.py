import multiprocessing
import os
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from itertools import combinations, count, product
from multiprocessing.pool import Pool
from typing import Any, Iterable

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm import tqdm

from ml_pipe.features import Features
from ml_pipe.search import (Candidate, CandidateSpace, FeatureSearch, SearchSettings, greedy_features,
                            has_ranges, import_optuna, optuna_search, successive_halving)
from ml_pipe.target import Target
from ml_pipe.workers import init_worker, train_k_folds_mc, train_test_split_mc, select_k_best_many
from ml_pipe.train_data import TrainData, KFoldsTrainData, build_estimator
from ml_pipe.utils import ValidationType, is_fitted, create_shared_numpy, cleanup_shared_memory, run_block_name, \
    SharedMemory


# Start workers as fresh interpreters on every OS. Linux defaults to fork(), which can deadlock
# when the pool is created from a threaded program (e.g. a threaded caller).
_MP_CONTEXT = multiprocessing.get_context("spawn")


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
    # brute_force_features() refuses more subsets than this (2^n - 1 grows fast).
    max_feature_combinations: int = 50_000
    # e.g. np.float32: halves shared memory and copy time; may change results slightly.
    features_dtype = None

    # set by subclasses
    results_type = None
    criteria_type = None
    default_criteria = None  # metric search() compares candidates with unless told otherwise
    _encode_target = True

    def __init__(self):
        self._features_index: list[list[int]] = None
        self._all_features = True  # use every column of the (transformed) data
        self._brute_force = False  # every subset; with optuna search they are never listed
        self._hyper_parameter: dict[str, list[Any]] = {}
        self.features = None
        self.targets = None
        self._train_data = None
        self._train_index = None  # rows used for training by the last train_test_split
        self._results = None
        self._run_id = None
        self._pool = None
        self._pool_size = None
        self._pool_finalizer = None
        self._search = SearchSettings()
        self._feature_search: FeatureSearch | None = None
        self.search_log: list[dict] = []  # one entry per round of the last run

    # ------------------------------------------------------------------ setup
    def set(self, features: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator]):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.features = Features()
        self.features.load_new_data(features)

        self.targets = Target(encode=self._encode_target)
        self.targets.load_new_data(target)

        self._features_index = [list(range(self.features.data.shape[1]))]
        self._all_features = True
        self._brute_force = False
        self._feature_search = None
        self._train_data = [TrainData(model, self.results_type) for model in models]

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 1:
            raise ValueError("limit must be greater than 0")
        BaseML._process_limit = min(new_limit, os.cpu_count())

    def set_hyper_parameters(self, hyper_parameter: dict[str, list[Any]]):
        self._hyper_parameter = hyper_parameter

    def search(self, strategy: str = "grid", *, n_iter: int | None = None, factor: float = 3,
               min_rows: int | None = None, select_by=None):
        """How the next k_folds / train_test_split chooses what to train.

        grid    every model x feature subset x hyper-parameter combination (default).
        random  n_iter combinations drawn at random from that grid.
        halving successive halving: all combinations (or a random n_iter of them) on a
                small share of the rows, the best 1/factor on factor x more rows, ...,
                the last round on all rows. min_rows = rows in the first round.
        select_by is the metric that decides who goes on (default accuracy / r2).
        """
        criteria = select_by if select_by is not None else self.default_criteria
        if self.criteria_type is not None and criteria not in self.criteria_type:
            raise ValueError(f"select_by must be one of {list(self.criteria_type)}")
        settings = SearchSettings(strategy, n_iter, factor, min_rows, criteria)
        settings.validate()
        if strategy == "optuna":
            import_optuna()  # fail now, not after loading the data
        self._search = settings
        return self

    @property
    def results(self):
        return self._results

    # ------------------------------------------------------------------ pool
    def _get_pool(self) -> Pool:
        if self._pool is not None and self._pool_size == self._process_limit:
            return self._pool
        self.close()  # process limit changed: start a pool of the new size
        pool = Pool(processes=self._process_limit, initializer=init_worker,
                    maxtasksperchild=self.max_tasks_per_child, context=_MP_CONTEXT)
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
        features = self.features.raw_data
        if self.features_dtype is not None:
            features = features.astype(self.features_dtype, copy=False)
        return np.ascontiguousarray(features)

    def _feature_sets(self) -> list:
        if self._all_features:
            return [list(range(self.features.data.shape[1]))]
        if self._brute_force and self._features_index is None:  # set up for optuna, used by another search
            self._features_index = self._all_subsets()
        return self._features_index

    def _all_subsets(self) -> list[list[int]]:
        n = self.features.data.shape[1]
        total = 2 ** n - 1
        if total > self.max_feature_combinations:
            raise ValueError(f"{n} features give {total:,} subsets (limit {self.max_feature_combinations:,}). "
                             f"Use feature_selection(), forward_selection(), search('optuna', ...), or raise "
                             f"{type(self).__name__}.max_feature_combinations.")
        return [[int(i) for i in combo] for combo in _brute_force(np.arange(n))]

    def _build_training_list(self, X: SharedMemory, y: SharedMemory) -> Iterable[TrainData]:
        preprocessing = self.features.preprocessing
        for i, candidate in enumerate(self._space().all()):
            yield self._task(candidate, i, X, y, preprocessing)

    def _space(self) -> CandidateSpace:
        return CandidateSpace([td.model for td in self._train_data], self._feature_sets(), self._hyper_parameter)

    def _task(self, candidate: Candidate, candidate_id: int, X: SharedMemory, y: SharedMemory,
              preprocessing: list) -> TrainData:
        td = self._train_data[candidate.model].copy()
        td.set_features_and_targets(X, y)
        td.set_features_selection(list(candidate.features) if candidate.features is not None else None)
        td.preprocessing = preprocessing
        td.candidate_id = candidate_id
        if candidate.params:
            td.model.set_params(**candidate.params)
        return td

    # ------------------------------------------------------------------ search
    def _criteria(self):
        return self._search.criteria if self._search.criteria is not None else self.default_criteria

    def _default_min_rows(self, n_splits: int) -> int:
        """Smallest first halving round: two rows per fold (per class, for classification)."""
        y = self.targets.data
        n_classes = len(np.unique(y)) if self._stratify_labels(y) is not None else 1
        return max(2 * n_splits * n_classes, 20)

    def _subsample(self, rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """n of `rows`, stratified for classification."""
        if n >= len(rows):
            return rows
        stratify = self._stratify_labels(self.targets.data[rows])
        seed = int(rng.integers(2 ** 31 - 1))
        try:
            picked, _ = sk_train_test_split(rows, train_size=n, random_state=seed, stratify=stratify)
        except ValueError:
            if stratify is None:
                raise
            picked, _ = sk_train_test_split(rows, train_size=n, random_state=seed)
        return np.sort(picked)

    def _execute(self, evaluate, rows: np.ndarray, n_splits: int) -> list:
        """Run the configured search. evaluate(candidates, rows, desc) trains one batch;
        `rows` are the rows a halving round may subsample from."""
        self.search_log = []
        settings, criteria = self._search, self._criteria()
        n_columns = self.features.data.shape[1]
        if has_ranges(self._hyper_parameter) and settings.strategy != "optuna":
            raise ValueError("hyper-parameter ranges (Range) need search('optuna', ...); "
                             f"{settings.strategy} search needs lists of values")
        if settings.strategy == "optuna" and self._feature_search is None:
            feature_sets = None if self._brute_force else self._feature_sets()
            return optuna_search(evaluate, [td.model for td in self._train_data], feature_sets, n_columns,
                                 self._hyper_parameter, settings.n_iter, self._process_limit, criteria,
                                 self.random_state, self.search_log)
        space = self._space()
        if self._feature_search is not None:
            if settings.strategy != "grid":
                raise ValueError(f"{self._feature_search.direction} feature selection can't be combined with "
                                 f"{settings.strategy} search")
            return greedy_features(evaluate, space.tracks(), n_columns, self._feature_search, criteria,
                                   self.search_log)
        rng = np.random.default_rng(self.random_state)
        if settings.n_iter is not None and settings.strategy in ("random", "halving"):
            candidates = space.sample(settings.n_iter, rng)
        else:
            candidates = space.all()
        if settings.strategy != "halving":
            return evaluate(candidates, None, "training")
        min_rows = min(settings.min_rows or self._default_min_rows(n_splits), len(rows))
        return successive_halving(evaluate, candidates, lambda n: self._subsample(rows, n, rng), len(rows),
                                  min_rows, settings.factor, criteria, n_columns, self.search_log)

    def _evaluate_batch(self, func, tasks: list, desc: str) -> list:
        results = self._run(func, tasks, desc)
        results.sort(key=lambda r: r.candidate_id)  # arrival order varies; keep runs reproducible
        return results

    # ------------------------------------------------------------------ splitting
    def _stratify_labels(self, y: np.ndarray):
        """Labels to stratify on, or None (regression)."""
        return None

    @abstractmethod
    def _make_splitter(self, n_splits: int, shuffle: bool, seed, y: np.ndarray = None):
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
        splitter = self._make_splitter(n_splits, shuffle, seed, y)
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
        train_index, test_index = self._split_rows(ratio, y.array)
        self._train_index = train_index
        preprocessing = self.features.preprocessing

        def evaluate(candidates, rows, desc):
            # a halving round trains on part of the training rows; the test rows stay the same
            train = train_index if rows is None else rows
            tasks = []
            for i, candidate in enumerate(candidates):
                td = self._task(candidate, i, X, y, preprocessing)
                td.set_indexes(train, test_index)
                tasks.append(td)
            return self._evaluate_batch(train_test_split_mc, tasks, desc)

        self._results = self._execute(evaluate, np.sort(train_index), n_splits=5)
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
        leave_one_out = n_splits == len(self.targets.data)
        if leave_one_out and self._search.strategy == "halving" and self._feature_search is None:
            raise ValueError("halving search needs k_folds or train_test_split, not leave-one-out")
        self._start_run()
        X = self._share(self._training_features(), "features")
        y = self._share(self.targets.data, "target")
        seed = self._shared_seed(shuffle, self.random_state)
        # Folds are computed once here, so every model is scored on identical folds.
        folds = self._share(self._fold_ids(n_splits, shuffle, seed, y.array), "folds")
        validation = ValidationType.leave_one_out if leave_one_out else ValidationType.k_folds
        preprocessing = self.features.preprocessing
        round_number = count(1)

        def evaluate(candidates, rows, desc):
            round_folds = folds
            if rows is not None:  # halving round: fold ids for these rows only, -1 = row not used
                ids = np.full(len(y.array), -1, dtype=np.int64)
                ids[rows] = self._fold_ids(n_splits, shuffle, seed, y.array[rows])
                round_folds = self._share(ids, f"folds{next(round_number)}")
            tasks = [KFoldsTrainData.set_from_train_data(self._task(c, i, X, y, preprocessing),
                                                         number_of_folds=n_splits, shuffle=shuffle,
                                                         random_state=seed, folds=round_folds,
                                                         validation=validation)
                     for i, c in enumerate(candidates)]
            return self._evaluate_batch(train_k_folds_mc, tasks, desc)

        self._results = self._execute(evaluate, np.arange(len(y.array)), n_splits)
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
            yield KFoldsTrainData.set_from_train_data(td, number_of_folds=kf, shuffle=shuffle, random_state=seed,
                                                      folds=folds, validation=validation)

    # ------------------------------------------------------------------ features
    def feature_selection(self, algorithm, number_of_features: list[int]):
        """Choose feature subsets with a SelectKBest score function (chi2, f_classif,
        f_regression, ...). 10% of the rows are used for the selection and then removed,
        so they never take part in training or testing."""
        rows = np.arange(len(self.targets.data))
        _, held_out = sk_train_test_split(rows, test_size=0.1, random_state=self.random_state)
        features = self.features.pop_index(held_out)
        target = self.targets.pop_index(held_out)
        # One scoring pass serves every k; no process pool needed.
        self._features_index = select_k_best_many(algorithm, features, target, number_of_features)
        self._all_features = False
        self._brute_force = False
        self._feature_search = None

    def brute_force_features(self):
        """Every non-empty subset of the columns, including all of them.

        After search("optuna", ...) the subsets are not listed: optuna switches each
        column on or off per trial, so there is no limit on the number of columns."""
        self._features_index = None if self._search.strategy == "optuna" else self._all_subsets()
        self._all_features = False
        self._brute_force = True
        self._feature_search = None

    def forward_selection(self, max_features: int | None = None, tol: float = 0.0):
        """Greedy alternative to brute_force_features(): start with no features and, for each
        model x hyper-parameter combination, keep adding the feature that improves the score
        most, while it improves by more than tol (and up to max_features). Every subset tried
        is in the results. Scored with the search() metric (default accuracy / r2)."""
        search = FeatureSearch("forward", max_features=max_features, tol=tol)
        search.validate()
        self._feature_search = search
        self._all_features = True
        self._brute_force = False

    def backward_elimination(self, min_features: int = 1, tol: float = 0.0):
        """Start with every feature and keep removing the one whose removal leaves the best
        score, while the score stays within tol of the best seen (down to min_features)."""
        search = FeatureSearch("backward", min_features=min_features, tol=tol)
        search.validate()
        self._feature_search = search
        self._all_features = True
        self._brute_force = False

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
        preprocessing = self.features.preprocessing
        n_columns = self.features.data.shape[1]
        columns_in = list(self.features.columns) if self.features.columns is not None else None
        for result in results:
            result.feature_columns = columns_in
            result.class_names = self.targets.classes
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

    # ------------------------------------------------------------------ old names (deprecated)
    @property
    def fetchers(self):
        _deprecated("fetchers", "features")
        return self.features

    def fetcher_selection(self, algorithm, number_of_features: list[int]):
        _deprecated("fetcher_selection()", "feature_selection()")
        return self.feature_selection(algorithm, number_of_features)

    def brut_force_features(self):
        _deprecated("brut_force_features()", "brute_force_features()")
        return self.brute_force_features()


def _deprecated(old: str, new: str):
    warnings.warn(f"{old} is deprecated, use {new}", DeprecationWarning, stacklevel=3)


def _brute_force(vec: np.ndarray) -> Iterable[tuple]:
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
