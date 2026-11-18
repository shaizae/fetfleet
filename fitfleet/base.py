import multiprocessing
import os
import sys
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from functools import partial, wraps
from itertools import combinations, count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm import tqdm

from fitfleet.encoding import ENCODING_PARAM, ENCODINGS, MAX_ONEHOT
from fitfleet.features import Features
from fitfleet.schema import ColumnSchema, id_like_columns, to_pandas
from fitfleet.search import (Candidate, CandidateSpace, FeatureSearch, SearchSettings, greedy_features,
                            successive_halving)
from fitfleet.target import Target
from fitfleet.tracking import TrackedTask, current_stage, execution, stage, tracked
from fitfleet.workers import init_worker, train_k_folds_mc, train_test_split_mc, select_k_best_many
from fitfleet.train_data import TrainData, KFoldsTrainData, build_estimator
from fitfleet.utils import Sound, ValidationType, in_enum, is_fitted, create_shared_memory, cleanup_shared_memory, run_block_name, \
    SharedMemory


# Start workers as fresh interpreters on every OS. Linux defaults to fork(), which can deadlock
# when the pool is created from a threaded program (e.g. a threaded caller).
_MP_CONTEXT = multiprocessing.get_context("spawn")


def _tracked_run(name: str, operation: str):
    """Track a training run: a stage of the active execution, or an execution of its own (kept in
    `self._execution`, console off) when nothing is tracked. Failures are recorded and re-raised."""
    def decorate(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            arguments = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool, str))}
            arguments.update({f"arg{i}": v for i, v in enumerate(args) if isinstance(v, (int, float, bool, str))})
            with execution(name, module=__name__, operation=operation, console=False,
                           metadata={**self._run_metadata(), **arguments}) as tracker:
                self._execution = tracker
                return method(self, *args, **kwargs)
        return wrapper
    return decorate


class BaseML(ABC):
    """Runs every model x feature subset x hyper-parameter combination in a process pool.

    The pool is created on first use and reused by later runs on the same object
    (starting worker processes costs seconds on Windows). Call close(), or use the
    object as a context manager, to stop the workers early; otherwise they stop when
    the object is garbage-collected or the interpreter exits.

    `sound` (a Sound) plays once when the object goes out of scope - the end of its
    `with` block, or garbage collection - if at least one training run finished.
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
    # How categorical columns are encoded for each model: "auto" (one-hot for linear / distance
    # models, codes for tree models, native categories for HistGradientBoosting), "onehot" or
    # "ordinal" (codes for every model) or "target" (target encoding for every model). See fitfleet.encoding.
    categorical_encoding = "auto"
    # auto encoding: categorical columns with more categories than this are target encoded
    # (instead of one-hot) for models that are not trees.
    max_onehot_categories = MAX_ONEHOT

    # set by subclasses
    results_type = None
    criteria_type = None
    default_criteria = None  # metric search() compares candidates with unless told otherwise
    _encode_target = True

    def __init__(self, sound: Sound | str = Sound.none, log_dir: str | Path = "logs"):
        """log_dir: folder that save_pdf_log / save_json_log write the execution log into."""
        self.sound = Sound(sound)
        self._log_dir = Path(log_dir)
        self._sound_finalizer = None
        self._features_index: list[list[int]] = None
        self._all_features = True  # use every column of the (transformed) data
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
        # ExecutionTracker of the last run (its own one, or the enclosing tracker when the run was
        # started inside a tracked execution). Users see it only through the three log properties.
        self._execution = None

    # ------------------------------------------------------------------ setup
    def set(self, features: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator], *,
            categorical: list[str] | None = None, numeric: list[str] | None = None,
            ordinal: dict[str, list] | None = None, datetime: list[str] | None = None,
            text: list[str] | None = None, drop_ids: bool = True):
        """The data and the models to try.

        `features` may mix numeric, boolean, categorical (text / pandas category), ordinal and
        date and free-text columns and hold missing values; see fitfleet.schema and fitfleet.encoding.
        pandas or polars input. Kinds are detected; declare them to override: `categorical` (e.g.
        codes stored as numbers), `numeric`, `ordinal` ({column: [values, lowest first]}),
        `datetime` (text to parse as dates) and `text` (free text, embedded as numbers).

        drop_ids: leave out text columns where (almost) every row has its own value (ids, keys),
        with a warning - as categories they can't generalize. Declared columns are always kept."""
        features, target = to_pandas(features), to_pandas(target)
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        schema = None
        if isinstance(features, DataFrame):
            declared = {*(categorical or ()), *(numeric or ()), *(ordinal or {}), *(datetime or ()), *(text or ())}
            ids = [c for c in id_like_columns(features) if c not in declared] if drop_ids else []
            if ids and len(ids) < features.shape[1]:
                warnings.warn(f"left out id-like columns {ids}: almost every row has its own value, so they "
                              f"can't generalize (declare them categorical, or set drop_ids=False, to keep them)",
                              stacklevel=2)
                features = features.drop(columns=ids)
            schema = ColumnSchema.infer(features, categorical, numeric, ordinal, datetime, text,
                                        random_state=self.random_state)
            if schema.datetime_index:
                warnings.warn(f"date columns {[schema.columns[i] for i in schema.datetime_index]}: when rows are "
                              f"ordered in time, shuffled folds let models learn from the future and scores come "
                              f"out optimistic - consider sorting the rows by date and using "
                              f"k_folds(shuffle=False), which tests on contiguous blocks of time", stacklevel=2)
        self.features = Features()
        self.features.load_new_data(features, schema)

        self.targets = Target(encode=self._encode_target)
        self.targets.load_new_data(target)

        self._features_index = [list(range(self.features.n_selectable))]
        self._all_features = True
        self._feature_search = None
        self._train_data = [TrainData(model, self.results_type) for model in models]

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 1:
            raise ValueError("limit must be greater than 0")
        BaseML._process_limit = min(new_limit, os.cpu_count())

    def set_hyper_parameters(self, hyper_parameter: dict[str, list[Any]]):
        """Values to try per parameter name; each applies to the models that have that parameter.
        "categorical_encoding" (e.g. ["onehot", "target"]) applies to every model: the encodings of
        a mixed table's categorical columns are then compared on the same folds."""
        bad = [v for v in hyper_parameter.get(ENCODING_PARAM, []) if v not in ENCODINGS]
        if bad:
            raise ValueError(f"{ENCODING_PARAM} values must be in {list(ENCODINGS)}, not {bad}")
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
        if self.criteria_type is not None and not in_enum(self.criteria_type, criteria):
            raise ValueError(f"select_by must be one of {list(self.criteria_type)}")
        settings = SearchSettings(strategy, n_iter, factor, min_rows, criteria)
        settings.validate()
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
        if self._sound_finalizer is not None:
            self._sound_finalizer()  # plays now; a finalizer runs at most once

    # ------------------------------------------------------------------ sound
    def _training_finished(self):
        """Arm the end sound: it plays when this object goes out of scope."""
        if self._sound_finalizer is not None:
            self._sound_finalizer.detach()
        self._sound_finalizer = (weakref.finalize(self, _play_end_sound, self.sound)
                                 if self.sound is not Sound.none else None)

    @staticmethod
    def play_sound(sound: Sound | str):
        """Play `sound`; never raises (no sound device -> silently skipped).

        The sound is rendered as a WAV and played with winsound (Windows), afplay (macOS) or
        paplay / aplay (Linux); if none of those work, the terminal bell rings instead.
        """
        try:
            sound = Sound(sound)
            if sound is Sound.none:
                return
            if sound is not Sound.bell and _play_wav(sound.wav_bytes()):
                return
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass

    def _run(self, func, tasks: list, desc: str) -> list:
        # Longest tasks first, so one slow model doesn't start last and leave the
        # other workers idle at the end.
        tasks = sorted(tasks, key=lambda task: task.cost(), reverse=True)
        pool = self._get_pool()
        # Batch small tasks to cut inter-process round trips; keep ~4 batches per worker
        # so the load still balances.
        chunksize = max(1, len(tasks) // (self._pool_size * 4))
        tracker, stage_id = current_stage()
        if tracker is not None:  # tasks report their pid / timing, and failures say which configuration
            func = TrackedTask(func, tracker.execution_id, stage_id)
        try:
            outcomes = list(tqdm(pool.imap_unordered(func, tasks, chunksize=chunksize),
                                 total=len(tasks), desc=desc))
        except BaseException as e:
            self.close()  # a worker may be stuck mid-task: start clean next run
            if tracker is not None:
                tracker.task_failed(e, parent=stage_id)
            raise
        return outcomes if tracker is None else tracker.tasks_done(outcomes, parent=stage_id)

    # ------------------------------------------------------------------ data
    def _start_run(self) -> str:
        self._run_id = uuid.uuid4().hex[:12]
        return self._run_id

    def _share_data(self) -> tuple[SharedMemory, SharedMemory]:
        """Features and target in shared memory for the workers."""
        with stage("Share data with workers", operation="copy features and target into shared memory") as s:
            X = self._share(self._training_features(), "features")
            y = self._share(self.targets.data, "target")
            s.annotate(rows=int(X.shape[0]), columns=int(X.shape[1]) if len(X.shape) > 1 else 1,
                       megabytes=round((X.array.nbytes + y.array.nbytes) / 2 ** 20, 3))
        return X, y

    def _run_metadata(self) -> dict:
        """What a run trains, for its execution record (shapes and names only, never data)."""
        try:
            return {"task": type(self).__name__, "rows": int(self.features.data.shape[0]),
                    "columns": int(self.features.data.shape[1]),
                    "models": [type(td.model).__name__ for td in self._train_data],
                    "hyper_parameters": sorted(self._hyper_parameter), "search": self._search.strategy,
                    "feature_search": None if self._feature_search is None else self._feature_search.direction,
                    "processes": self._process_limit}
        except Exception:  # noqa: BLE001 - not set up yet; the run itself will say what is missing
            return {"task": type(self).__name__}

    def _share(self, array: np.ndarray, kind: str) -> SharedMemory:
        return create_shared_memory(array, run_block_name(self._run_id or self._start_run(), kind))

    def _training_features(self) -> np.ndarray:
        """Raw features: recorded preprocessing is refitted inside every split."""
        features = self.features.raw_data
        if self.features_dtype is not None:
            features = features.astype(self.features_dtype, copy=False)
        return np.ascontiguousarray(features)

    def _feature_sets(self) -> list:
        if self._all_features:
            return [list(range(self.features.n_selectable))]
        return self._features_index

    def _space(self) -> CandidateSpace:
        if ENCODING_PARAM in self._hyper_parameter and not self.features.mixed:
            raise ValueError(f"{ENCODING_PARAM} is searched only for tables with categorical, date, text "
                             f"columns or missing values; this table is purely numeric")
        return CandidateSpace([td.model for td in self._train_data], self._feature_sets(), self._hyper_parameter)

    def _task(self, candidate: Candidate, candidate_id: int, X: SharedMemory, y: SharedMemory,
              preprocessing: list) -> TrainData:
        td = self._train_data[candidate.model].copy()
        td.set_features_and_targets(X, y)
        td.set_features_selection(list(candidate.features) if candidate.features is not None else None)
        td.preprocessing = preprocessing
        td.candidate_id = candidate_id
        params = dict(candidate.params or {})
        encoding = params.pop(ENCODING_PARAM, None)
        if self.features.mixed:
            td.schema = self.features.schema
            td.encoding = encoding or self.categorical_encoding
            td.encoding_searched = encoding is not None
            td.encoder_options = self._encoder_options()
        if params:
            td.model.set_params(**params)
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
        strategy = (f"{self._feature_search.direction} feature selection" if self._feature_search is not None
                    else f"{self._search.strategy} search")
        with stage(f"Search ({strategy})", operation=f"choose and train configurations: {strategy}") as s:
            results = self._execute_search(evaluate, rows, n_splits)
            s.annotate(configurations_trained=len(results), rounds=len(self.search_log) or 1)
            return results

    def _execute_search(self, evaluate, rows: np.ndarray, n_splits: int) -> list:
        self.search_log = []
        settings, criteria = self._search, self._criteria()
        n_columns = self.features.n_selectable
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
        with stage(f"Train batch: {desc}", operation=f"train {len(tasks)} configuration(s) in the process pool",
                   configurations=len(tasks), processes=self._process_limit):
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
    @_tracked_run("train_test_split", "train and score every configuration on one train / test split")
    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        """ratio = fraction of rows used for training (0.8 -> 80% train, 20% test)."""
        if not 0 < ratio < 1:
            raise ValueError("ratio (fraction of rows used for training) must be between 0 and 1")
        self._start_run()
        X, y = self._share_data()
        with stage("Split rows", operation=f"split rows {ratio:.0%} train / {1 - ratio:.0%} test"):
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
        self._training_finished()
        return self._results

    @_tracked_run("k_folds", "train and score every configuration with k-fold cross-validation")
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
        X, y = self._share_data()
        seed = self._shared_seed(shuffle, self.random_state)
        # Folds are computed once here, so every model is scored on identical folds.
        with stage("Compute folds", operation=f"assign every row to one of {n_splits} folds", n_splits=n_splits):
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
        self._training_finished()
        return self._results

    def leave_one_out(self):
        return self.k_folds(n_splits=len(self.targets.data))

    # ------------------------------------------------------------------ features
    @tracked("Feature selection", operation="choose feature subsets with a SelectKBest score function")
    def feature_selection(self, algorithm, number_of_features: list[int]):
        """Choose feature subsets with a SelectKBest score function (chi2, f_classif,
        f_regression, ...). 10% of the rows are used for the selection and then removed,
        so they never take part in training or testing."""
        rows = np.arange(len(self.targets.data))
        _, held_out = sk_train_test_split(rows, test_size=0.1, random_state=self.random_state)
        mixed = self.features.mixed
        if mixed:  # score the original columns; the codes / NaN need preparing first
            algorithm = self._mixed_score_function(algorithm)
            raw = self.features.raw_data[held_out]
        features = self.features.pop_index(held_out)
        target = self.targets.pop_index(held_out)
        if mixed:
            features = _fill_missing(raw)
        # One scoring pass serves every k; no process pool needed.
        if mixed:
            self._features_index = select_k_best_columns(algorithm, features, target, number_of_features,
                                                         self.features.schema)
        else:
            self._features_index = select_k_best_many(algorithm, features, target, number_of_features)
        self._all_features = False
        self._feature_search = None

    def _encoder_options(self) -> dict:
        return {"max_onehot": self.max_onehot_categories, "regression": not self._encode_target,
                "random_state": self.random_state}

    def _mixed_score_function(self, algorithm):
        """The SelectKBest score function for a table with categorical columns: mutual information
        is told which columns are categories; other score functions would read codes as amounts."""
        name = getattr(algorithm, "__name__", "")
        schema = self.features.schema
        categorical = schema.categorical_index
        if name in ("mutual_info_classif", "mutual_info_regression"):
            mask = np.zeros(schema.n_matrix, dtype=bool)
            mask[schema.matrix_index(categorical)] = True
            return partial(algorithm, discrete_features=mask, random_state=self.random_state)
        if categorical:
            raise ValueError(f"{name or algorithm!r} can't score categorical columns "
                             f"{[self.features.schema.columns[i] for i in categorical]}: use mutual_info_classif / "
                             f"mutual_info_regression, brute_force_features(), forward_selection() or "
                             f"backward_elimination()")
        return algorithm

    @tracked("Brute-force feature subsets", operation="list every non-empty subset of the columns")
    def brute_force_features(self):
        """Every non-empty subset of the columns, including all of them."""
        n = self.features.n_selectable
        total = 2 ** n - 1
        if total > self.max_feature_combinations:
            raise ValueError(f"{n} features give {total:,} subsets (limit {self.max_feature_combinations:,}). "
                             f"Use feature_selection(), or raise {type(self).__name__}.max_feature_combinations.")
        self._features_index = [[int(i) for i in combo] for combo in _brute_force(np.arange(n))]
        self._all_features = False
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

    def backward_elimination(self, min_features: int = 1, tol: float = 0.0):
        """Start with every feature and keep removing the one whose removal leaves the best
        score, while the score stays within tol of the best seen (down to min_features)."""
        search = FeatureSearch("backward", min_features=min_features, tol=tol)
        search.validate()
        self._feature_search = search
        self._all_features = True

    # ------------------------------------------------------------------ results
    def filter_by(self, criteria):
        if self.criteria_type is not None and not in_enum(self.criteria_type, criteria):
            raise ValueError("criteria must be one of {}".format(list(self.criteria_type)))
        values = [getattr(result, criteria.value) for result in self._results]

        best_value = max(values) if criteria.maximize else min(values)

        best = [result for result in self._results if getattr(result, criteria.value) == best_value]
        return self.refit(best)

    def experiment_report(self, criteria=None, *, all_params: bool = False):
        """Summary of every configuration the last run trained (classification or regression).

        criteria: metric to rank by (default accuracy / r2). Hyper-parameters set with
        set_hyper_parameters() are listed per configuration; all_params=True lists every one.
        Returns an ExperimentReport: print() it, or use .leaderboard / .summary / .best,
        .save_csv(folder) and .save_pdf_report(folder).
        """
        from fitfleet.results.experiment import ExperimentReport
        criteria = self.default_criteria if criteria is None else self.criteria_type(criteria)
        if not self._results:
            raise ValueError("no results yet: run k_folds(), train_test_split() or leave_one_out() first")
        task = "classification" if getattr(self.results_type, "is_classification", False) else "regression"
        grid = None if all_params else self._hyper_parameter
        return ExperimentReport(task, self._results, criteria, self.features, grid,
                                hyper_parameters=self._hyper_parameter, class_names=self.targets.classes,
                                search_log=self.search_log, execution=self._execution)

    # ------------------------------------------------------------------ execution log of the last run
    def _tracker(self):
        if self._execution is None:
            raise RuntimeError("no execution log yet: run k_folds(), train_test_split() or leave_one_out() first")
        return self._execution

    @property
    def execution_log(self) -> str:
        """The last run's execution log as text: the summary (result, counts, failed and skipped
        stages with where and why), then the stage tree. print(run.execution_log) shows it."""
        return self._tracker().log()._report()

    @property
    def save_pdf_log(self) -> Path:
        """Write the last run's execution log (summary + stage tree) as a PDF into log_dir; its path."""
        return self._tracker().log()._save_pdf_log(self._log_dir)

    @property
    def save_json_log(self) -> Path:
        """Write the last run's execution log (summary + stage tree + every stage) as JSON into log_dir; its path."""
        return self._tracker().log()._save_json_log(self._log_dir)

    def refit(self, results):
        """Fit the model of each result in this process so it can be saved or used.

        Workers return only metrics and predictions. train_test_split results are
        refitted on the same training rows; k-fold / leave-one-out results on all rows.
        With preprocessing or a feature subset the fitted model is a Pipeline that takes
        raw rows with all columns, so it can be used on new data as is.
        """
        X, y = self._training_features(), self.targets.data
        preprocessing = self.features.preprocessing
        n_columns = self.features.n_selectable
        schema = self.features.schema if self.features.mixed else None
        columns_in = list(self.features.columns) if self.features.columns is not None else None
        for result in results:
            result.feature_columns = columns_in
            result.schema = schema
            result.class_names = self.targets.classes
            result._execution = self._execution  # its save_pdf_report() adds the run's execution report
            if is_fitted(result.model):
                continue
            rows = self._train_index if result.validation == ValidationType.train_test_split else slice(None)
            columns = result.features_index
            if columns is not None and list(columns) == list(range(n_columns)):
                columns = None  # all columns: no selection step needed
            encoding = getattr(result, "categorical_encoding", None) or self.categorical_encoding
            estimator = build_estimator(result.model, preprocessing, columns, schema, encoding,
                                        encoder_options=self._encoder_options())
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


def _play_end_sound(sound: Sound):
    BaseML.play_sound(sound)  # looked up when it runs, so it can be replaced (e.g. in tests)


def _play_wav(data: bytes) -> bool:
    """Play WAV bytes to the end; True if a player was found and worked."""
    try:
        import winsound
        winsound.PlaySound(data, winsound.SND_MEMORY)  # waits until done, so exit doesn't cut it off
        return True
    except ImportError:
        pass
    import shutil
    import subprocess
    import tempfile
    for player in ("afplay", "paplay", "aplay"):
        if shutil.which(player):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
            try:
                return subprocess.run([player, f.name], capture_output=True, timeout=10).returncode == 0
            finally:
                os.unlink(f.name)
    return False


def _deprecated(old: str, new: str):
    warnings.warn(f"{old} is deprecated, use {new}", DeprecationWarning, stacklevel=3)


def select_k_best_columns(algorithm, features: np.ndarray, target: np.ndarray, ks, schema: ColumnSchema) -> list:
    """Top-k *original* columns for every k in `ks`. A column's score is the best score of its matrix
    columns (a text column takes several). Ties keep the column order."""
    scores = algorithm(features, target)
    scores = np.asarray(scores[0] if isinstance(scores, tuple) else scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf)
    per_column = np.array([scores[schema.matrix_index([i])].max() for i in range(len(schema.columns))])
    order = np.argsort(-per_column, kind="stable")
    out, seen = [], set()
    for k in ks:
        key = tuple(sorted(int(i) for i in order[:min(k, len(order))]))
        if key not in seen:
            seen.add(key)
            out.append(np.asarray(key))
    return out


def _fill_missing(X: np.ndarray) -> np.ndarray:
    """NaN replaced by the column median (0 for an all-missing column); for scoring only."""
    if not np.isnan(X).any():
        return X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nan_to_num(np.nanmedian(X, axis=0))
    return np.where(np.isnan(X), medians, X)


def _brute_force(vec: np.ndarray) -> Iterable[tuple]:
    for i in range(1, len(vec) + 1):
        for combo in combinations(vec, i):
            yield combo

