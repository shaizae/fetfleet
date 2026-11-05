from __future__ import annotations

import gc
import multiprocessing
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple, Any

import numpy as np
from sklearn.base import BaseEstimator


class SharedMemory:
    _shms: dict[str, SharedMemory] = {}

    def __init__(self, memory: shared_memory.SharedMemory, shape: Tuple[Any, ...], dtype: np.dtype, ):
        self.shm = memory
        self.shape = shape
        self.dtype = dtype

        SharedMemory.add(self)

    def unlink(self):
        self.shm.close()
        self.shm.unlink()

        SharedMemory._shms.pop(self.shm.name, None)

    @staticmethod
    def add(shm: SharedMemory):
        if shm.shm.name in SharedMemory._shms.keys():
            raise ValueError(f"{shm.shm.name} is already exists")
        SharedMemory._shms[shm.shm.name] = shm

    @staticmethod
    def validate(name: str) -> bool:
        return name in list(SharedMemory._shms.keys())

    @staticmethod
    def cleanup():
        for shm in list(SharedMemory._shms.values()):
            shm.unlink()

        SharedMemory._shms.clear()

    @property
    def array(self):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, )

    def __repr__(self):
        return (f"SharedMemory(name={self.shm.name}, shape={self.shape}, dtype={self.dtype})")

    def __deepcopy__(self, memo):
        # The block is shared by design: copying a task must not reopen it.
        return self

    def __reduce__(self):
        # Pickle only (name, shape, dtype). Workers reattach by name, once per process.
        return _attach_shared, (self.shm.name, self.shape, np.dtype(self.dtype).str)


# Blocks this worker process has already attached to, keyed by name.
_ATTACHED: dict[str, SharedMemory] = {}


def _attach_shared(name: str, shape: Tuple[Any, ...], dtype: str) -> SharedMemory:
    # The owning process already holds the block.
    owned = SharedMemory._shms.get(name)
    if owned is not None:
        return owned

    # Cache only inside pool workers. The parent reuses names ("features", "target")
    # across runs, so a cached handle there would point at an unlinked block.
    in_worker = multiprocessing.parent_process() is not None
    cached = _ATTACHED.get(name) if in_worker else None
    if cached is not None and cached.shape == tuple(shape) and cached.dtype == np.dtype(dtype):
        return cached

    try:
        memory = shared_memory.SharedMemory(name=name, track=False)  # Python 3.13+
    except TypeError:
        memory = shared_memory.SharedMemory(name=name)
    attached = SharedMemory.__new__(SharedMemory)  # skip __init__: workers don't own the block
    attached.shm = memory
    attached.shape = tuple(shape)
    attached.dtype = np.dtype(dtype)
    if in_worker:
        _ATTACHED[name] = attached
    return attached


def create_shared_numpy(arr: np.ndarray, name: str) -> SharedMemory:
    if SharedMemory.validate(name):
        raise ValueError(f"{name} is already exists")
    memory = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shared_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=memory.buf)
    shared_array[:] = arr[:]
    shared = SharedMemory(memory=memory, shape=arr.shape, dtype=arr.dtype)
    return shared


def features_unpackage(shared_memory: SharedMemory, index: list[int], features_index: list[int]):
    return shared_memory.array[index][..., features_index]


@dataclass(slots=True)
class FeaturesSelectionsData:
    algorithm: BaseEstimator
    features: SharedMemory
    target: SharedMemory
    number_of_features: int


class ValidationType(StrEnum):
    train_test_split = "train_test_split"
    k_folds = "k_folds"
    leave_one_out = "leave_one_out"


class FilteringCriteriaClassification(StrEnum):
    accuracy = "accuracy"
    f1 = "f1"
    recall = "recall"
    precision = "precision"

    @property
    def maximize(self):
        return self == FilteringCriteriaClassification.accuracy or self == FilteringCriteriaClassification.f1 or self == FilteringCriteriaClassification.recall or self == FilteringCriteriaClassification.precision


class FilteringCriteriaRegression(StrEnum):
    mae = "mae"
    mse = "mse"
    rmse = "rmse"
    r2 = "r2"

    @property
    def maximize(self):
        return self == FilteringCriteriaRegression.r2


def is_fitted(model) -> bool:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except TypeError:  # not an sklearn estimator; assume the caller knows
        return True


def create_path(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)  # creates all missing folders
    return p


def force_gc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            gc.collect(2)

    return wrapper


def cleanup_shared_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            SharedMemory.cleanup()

    return wrapper
