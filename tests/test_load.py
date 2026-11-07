"""CPU load tests for the multiprocessing pipeline.

The regular unit tests use iris (150 rows) with 10-tree forests: each fit takes
milliseconds, so the worker pool is idle almost all the time and you never see
the CPU climb. These tests push real work through the pool and *measure* it.

Run (slow, ~1-2 min, will load your CPU):
    pytest --run-load -m load -s tests/test_load.py

-s prints a per-test utilisation timeline.
"""
import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_pipe.base import BaseML
from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.utils import SharedMemory

pytestmark = pytest.mark.load

N_ROWS = 20_000
N_FEATURES = 40
FOLDS = 3
TREES = 50  # ~2 s per fit on 13k rows -> each task keeps a worker busy for several seconds


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class CpuSampler:
    """Samples the CPU use of this process's children (the pool workers).

    busy_cores = summed cpu% of all workers / 100, i.e. how many cores they use.
    busy_workers = workers above 50% of one core in that sample.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[tuple[float, float, int]] = []  # (t, busy_cores, busy_workers)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _run(self):
        me = psutil.Process()
        known: dict[int, psutil.Process] = {}
        while not self._stop.wait(self.interval):
            busy_cores, busy_workers = 0.0, 0
            for child in me.children(recursive=True):
                proc = known.get(child.pid)
                if proc is None:  # first call only primes cpu_percent
                    known[child.pid] = child
                    try:
                        child.cpu_percent(None)
                    except psutil.Error:
                        pass
                    continue
                try:
                    pct = proc.cpu_percent(None)
                except psutil.Error:
                    continue
                busy_cores += pct / 100
                busy_workers += pct >= 50
            self.samples.append((time.perf_counter() - self._t0, busy_cores, busy_workers))

    # -- summaries ---------------------------------------------------------
    @property
    def peak_busy_workers(self) -> int:
        return max((s[2] for s in self.samples), default=0)

    @property
    def peak_busy_cores(self) -> float:
        return max((s[1] for s in self.samples), default=0.0)

    def mean_busy_cores(self, workers: int) -> float:
        """Mean over the steady phase: samples where at least half the workers are busy
        (excludes process start-up and the tail where the last tasks finish)."""
        steady = [s[1] for s in self.samples if s[2] >= max(1, workers // 2)]
        return float(np.mean(steady)) if steady else 0.0

    def report(self, title: str, workers: int) -> str:
        lines = [f"\n{title}: {workers} workers on {os.cpu_count()} logical CPUs"]
        step = max(1, len(self.samples) // 25)
        for t, cores, busy in self.samples[::step]:
            lines.append(f"  t={t:5.1f}s  cores={cores:5.2f}  busy_workers={busy:2d}  "
                         f"{'#' * int(round(cores * 4))}")
        lines.append(f"  peak busy workers={self.peak_busy_workers}  peak cores={self.peak_busy_cores:.2f}  "
                     f"steady mean cores={self.mean_busy_cores(workers):.2f}")
        return "\n".join(lines)


@pytest.fixture
def workers():
    """Worker count for the test; restores the global limit afterwards."""
    saved = BaseML._process_limit
    n = max(2, min(8, (os.cpu_count() or 2) // 2))
    BaseML.process_limit(n)
    yield n
    BaseML._process_limit = saved


@pytest.fixture(autouse=True)
def no_shared_memory_left():
    yield
    leaked = list(SharedMemory._shms)
    SharedMemory.cleanup()
    assert not leaked, f"shared memory left registered after the run: {leaked}"


def classification_data():
    X, y = make_classification(n_samples=N_ROWS, n_features=N_FEATURES, n_informative=15,
                               n_classes=3, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), pd.Series(y)


def regression_data():
    X, y = make_regression(n_samples=N_ROWS, n_features=N_FEATURES, noise=5, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), pd.Series(y)


def heavy_pipeline(kind: str, n_tasks: int, trees: int = TREES):
    """n_tasks k-fold tasks. Models ask for n_jobs=-1 on purpose: the workers must
    override it, or each worker would start a thread per core (oversubscription)."""
    grid = {"min_samples_split": list(range(2, 2 + n_tasks))}
    if kind == "classification":
        pipe, data = Classification(), classification_data()
        model = RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=0)
    else:
        pipe, data = Regression(), regression_data()
        model = RandomForestRegressor(n_estimators=trees, n_jobs=-1, random_state=0)
    pipe.set(*data, [model])
    pipe.set_hyper_parameters(grid)
    return pipe


def wait_for_workers_to_exit(timeout: float = 10.0) -> list[psutil.Process]:
    me = psutil.Process()
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = [c for c in me.children(recursive=True) if c.is_running()]
        if not alive:
            return []
        time.sleep(0.2)
    return me.children(recursive=True)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_pool_saturates_all_workers(kind, workers):
    """Every worker is busy at the same time, the pool delivers close to `workers`
    cores of compute, and it does not oversubscribe past that."""
    n_tasks = 2 * workers  # two waves, so all workers stay busy for a while
    pipe = heavy_pipeline(kind, n_tasks)

    with pipe, CpuSampler() as cpu:
        results = pipe.k_folds(FOLDS)
    print(cpu.report(f"{kind} k-fold, {n_tasks} tasks", workers))

    assert len(results) == n_tasks
    # all workers ran concurrently (allow one to be between tasks at the sampled moment)
    assert cpu.peak_busy_workers >= workers - 1, cpu.report("not all workers were busy", workers)
    # the pool actually uses the cores it was given
    assert cpu.mean_busy_cores(workers) >= 0.6 * workers, cpu.report("pool under-utilised", workers)
    # n_jobs=-1 / BLAS threads are capped: no worker uses many cores on its own
    assert cpu.peak_busy_cores <= 1.3 * workers + 0.5, cpu.report("oversubscribed", workers)


_RAW_X, _RAW_Y = make_classification(n_samples=N_ROWS, n_features=N_FEATURES, n_informative=15,
                                     n_classes=3, random_state=0)


def _raw_kfold_task(min_samples_split: int) -> int:
    """The same work one pipeline task does (3-fold RF), with no pipeline code around it."""
    from sklearn.model_selection import KFold
    for train, test in KFold(FOLDS, shuffle=True, random_state=0).split(_RAW_X):
        model = RandomForestClassifier(n_estimators=20, min_samples_split=min_samples_split,
                                       n_jobs=1, random_state=0)
        model.fit(_RAW_X[train], _RAW_Y[train]).predict(_RAW_X[test])
    return min_samples_split


def test_parallel_speedup_matches_raw_pool(workers):
    """The pipeline gets the speedup the hardware allows.

    Comparing against the ideal Nx is wrong on most machines: laptops lower their
    clock under all-core load and SMT siblings share a core (measured on an 8-core
    Tiger Lake-H: a bare Pool tops out at ~3.7x with 8 processes). So we time a plain
    multiprocessing.Pool doing the same fits and require the pipeline to be close to it.
    """
    from multiprocessing.pool import Pool

    n_tasks = 2 * workers
    splits = list(range(2, 2 + n_tasks))

    def run_pipeline(n):
        BaseML.process_limit(n)
        with heavy_pipeline("classification", n_tasks, trees=20) as pipe:  # fresh pool: includes start-up
            start = time.perf_counter()
            pipe.k_folds(FOLDS)
            return time.perf_counter() - start

    serial = run_pipeline(1)
    parallel = run_pipeline(workers)

    start = time.perf_counter()  # includes pool start-up, like the pipeline timing
    with Pool(workers) as pool:
        pool.map(_raw_kfold_task, splits, chunksize=1)
    raw = time.perf_counter() - start

    print(f"\n1 worker {serial:.1f}s | {workers} workers: pipeline {parallel:.1f}s, raw Pool {raw:.1f}s")
    print(f"speedup: pipeline {serial / parallel:.2f}x, hardware ceiling (raw Pool) {serial / raw:.2f}x, "
          f"ideal {workers}x")
    assert serial / parallel >= 1.5, "pipeline barely runs in parallel"
    # pipeline overhead (task building, pickling, shared memory, results) stays small
    assert parallel <= 1.25 * raw + 1.0, f"pipeline {parallel:.1f}s vs raw Pool {raw:.1f}s"


def test_repeated_runs_do_not_leak(workers):
    """Back-to-back runs: every run returns all results, shared memory is released,
    worker processes exit, and the parent's memory does not keep growing."""
    me = psutil.Process()
    rss = []
    pipe = heavy_pipeline("classification", workers, trees=10)
    with pipe:  # one pipeline, one reused pool, many runs
        pool_pids = None
        for run in range(4):
            assert len(pipe.k_folds(FOLDS)) == workers
            assert len(pipe.train_test_split()) == workers
            assert not SharedMemory._shms, f"run {run}: shared memory not released"
            pids = sorted(p.pid for p in me.children())
            pool_pids = pool_pids or pids
            assert pids == pool_pids, "pool was not reused between runs"
            rss.append(me.memory_info().rss / 2 ** 20)

    leftover = wait_for_workers_to_exit()
    print(f"\nparent RSS per run (MB): {[round(r) for r in rss]}")
    assert not leftover, f"worker processes still alive: {[p.pid for p in leftover]}"
    # warm-up allocations land in the first run; after that memory should stay flat
    assert rss[-1] - rss[1] < 100, f"parent memory grew {rss[-1] - rss[1]:.0f} MB over repeated runs"


def test_pool_reuse_saves_startup(workers):
    """Second run on the same object skips process start-up."""
    with heavy_pipeline("classification", workers, trees=5) as pipe:
        start = time.perf_counter()
        pipe.k_folds(FOLDS)
        first = time.perf_counter() - start
        start = time.perf_counter()
        pipe.k_folds(FOLDS)
        second = time.perf_counter() - start
    print(f"\nfirst run (starts pool) {first:.2f}s, second run (reuses pool) {second:.2f}s")
    assert second < first
