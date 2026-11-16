import os
import sys

import matplotlib

# Draw plots off-screen. The default TkAgg backend opens a Tk window per figure and
# intermittently fails on Windows ("Can't find a usable init.tcl"), making plot tests flaky.
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid

from fitfleet.features import Features
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from fitfleet.classification import Classification
from fitfleet.utils import create_shared_memory
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from fitfleet.regression import Regression

RANDOM_STATE = 42

SPLIT_TRAIN_SIZE = 100
SPLIT_TEST_SIZE = 50

K_FOLDS = 5
LEAVE_ONE_OUT_FOLDS = 150

FEATURE_INDEXES = [0, 1, 2, 3]

IRIS_SAMPLES = 150


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def iris_data_shared_memory():
    """Iris in shared memory; the blocks are removed afterwards even if the test fails
    (the fixed names would otherwise make the next test fail with "already exists")."""
    from fitfleet.utils import SharedMemory
    data = load_iris()
    X = create_shared_memory(data.data, "features")
    y = create_shared_memory(data.target, "target")
    yield X, y
    for block in (X, y):
        if block.shm.name in SharedMemory._shms:  # the test may have cleaned up already
            block.unlink()


def _track_worker_pools():
    """Remember every object that starts a worker pool so the pool can be stopped after each test.

    A BaseML object keeps its pool (process_limit spawned workers, each importing numpy, pandas,
    sklearn and matplotlib) until the object is garbage-collected. In a test session that happens
    late or never: module fixtures and the tracebacks of failed tests keep the objects alive. The
    pools pile up until Windows runs out of commit memory and the IDE running pytest crashes."""
    import weakref
    from fitfleet.base import BaseML

    owners = weakref.WeakSet()
    get_pool = BaseML._get_pool

    def tracked_get_pool(self):
        owners.add(self)
        return get_pool(self)

    BaseML._get_pool = tracked_get_pool
    return owners


_POOL_OWNERS = _track_worker_pools()

# Tests train on tiny data sets: 2 workers exercise the multiprocessing path at a fraction of the
# memory of the default (half the CPU cores). test_load.py picks its own worker count on purpose.
TEST_PROCESS_LIMIT = 2


def _cap_process_limit():
    from fitfleet.base import BaseML
    BaseML._process_limit = min(BaseML._process_limit, TEST_PROCESS_LIMIT)


_cap_process_limit()


@pytest.fixture(autouse=True)
def close_worker_pools():
    """Stop every worker pool a test started (a later use simply starts a new one)."""
    yield
    for owner in list(_POOL_OWNERS):
        owner.close()
    _POOL_OWNERS.clear()


@pytest.fixture(autouse=True)
def restore_class_settings():
    """process_limit() and random_state are class-wide: undo any test's change."""
    from fitfleet.base import BaseML
    limit, seed = BaseML._process_limit, BaseML.random_state
    own = {cls: cls.__dict__.get("random_state", "unset") for cls in (Classification, Regression)}
    yield
    BaseML._process_limit, BaseML.random_state = limit, seed
    for cls, value in own.items():
        if value == "unset":
            if "random_state" in cls.__dict__:
                delattr(cls, "random_state")
        else:
            cls.random_state = value


@pytest.fixture
def classifier(iris_data):
    X, y = iris_data
    with Classification() as cls:
        cls.set(X, y, [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)])
        yield cls


@pytest.fixture
def shared_array():
    arr = np.arange(20).reshape(5, 4)
    shm = create_shared_memory(arr, f"test_{uuid.uuid4().hex[:12]}")
    yield shm, arr
    shm.unlink()


@pytest.fixture
def features_with_data(iris_data):
    df, _ = iris_data
    f = Features()
    f.load_new_data(df)
    return f


class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


@pytest.fixture
def logistic_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=500,
        random_state=RANDOM_STATE
    )


@pytest.fixture
def train_indexes():
    return list(range(SPLIT_TRAIN_SIZE))


@pytest.fixture
def test_indexes():
    return list(range(SPLIT_TRAIN_SIZE, SPLIT_TRAIN_SIZE + SPLIT_TEST_SIZE))


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42, )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def regression_class(regression_data):
    X, y = regression_data
    with Regression() as reg:
        reg.set(X, y, [LinearRegression(), Ridge(), RandomForestRegressor(n_estimators=10, random_state=42)])
        yield reg


# ---------------------------------------------------------------------------
# Load tests: heavy, multi-core, opt-in.  Run with:  pytest --run-load -m load -s
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption("--run-load", action="store_true", default=False,
                     help="run the CPU load tests (slow, saturate the worker pool)")


def pytest_configure(config):
    config.addinivalue_line("markers", "load: heavy multi-core load test (needs --run-load)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-load"):
        return
    skip = pytest.mark.skip(reason="load test: pass --run-load to run")
    for item in items:
        if item.get_closest_marker("load"):
            item.add_marker(skip)
