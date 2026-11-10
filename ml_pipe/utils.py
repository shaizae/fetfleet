"""Shared memory helpers, enums and small utilities used across the pipeline.

`_ATTACHED` holds the blocks this worker process has already attached to, keyed by
name. The `_C5` ... `_C7` constants are note frequencies in Hz, and `_PARTIALS` lists
the (frequency ratio, loudness) pairs that give the notes their bell-like timbre.
"""
from __future__ import annotations

import gc
import multiprocessing
import shutil
from enum import StrEnum
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class SharedMemory:
    """A numpy array stored in a named shared memory block, owned by this process.

    Every instance registers itself in a class-level registry so `cleanup()` can
    release all blocks at once. Pickling sends only (name, shape, dtype); workers
    reattach by name, once per process.
    """
    _shms: dict[str, SharedMemory] = {}

    def __init__(self, memory: shared_memory.SharedMemory, shape: Tuple[Any, ...], dtype: np.dtype, ) -> None:
        """Wrap an already created block and register it; raises if the name is taken."""
        self.shm = memory
        self.shape = shape
        self.dtype = dtype

        SharedMemory.add(self)

    def unlink(self) -> None:
        """Close and destroy the block and drop it from the registry."""
        self.shm.close()
        self.shm.unlink()

        SharedMemory._shms.pop(self.shm.name, None)

    @staticmethod
    def add(shm: SharedMemory) -> None:
        """Register a block by name; raises ValueError if the name is already registered."""
        if shm.shm.name in SharedMemory._shms.keys():
            raise ValueError(f"{shm.shm.name} is already exists")
        SharedMemory._shms[shm.shm.name] = shm

    @staticmethod
    def validate(name: str) -> bool:
        """Whether a block with this name is registered in this process."""
        return name in list(SharedMemory._shms.keys())

    @staticmethod
    def cleanup() -> None:
        """Unlink every registered block."""
        for shm in list(SharedMemory._shms.values()):
            shm.unlink()

        SharedMemory._shms.clear()

    @property
    def array(self) -> np.ndarray:
        """A numpy view on the block (no copy)."""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, )

    def __repr__(self) -> str:
        """Name, shape and dtype of the block."""
        return (f"SharedMemory(name={self.shm.name}, shape={self.shape}, dtype={self.dtype})")

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedMemory:
        """Return self: the block is shared by design, so copying a task must not reopen it."""
        return self

    def __reduce__(self) -> tuple[Callable[[str, Tuple[Any, ...], str], SharedMemory], tuple[str, Tuple[Any, ...], str]]:
        """Pickle only (name, shape, dtype); workers reattach by name, once per process."""
        return _attach_shared, (self.shm.name, self.shape, np.dtype(self.dtype).str)


_ATTACHED: dict[str, SharedMemory] = {}


def _attach_shared(name: str, shape: Tuple[Any, ...], dtype: str) -> SharedMemory:
    """Unpickle a SharedMemory: reattach to the named block without taking ownership.

    The owning process gets its own registered instance back. Handles are cached only
    inside pool workers: the parent reuses names ("features", "target") across runs, so
    a cached handle there would point at an unlinked block. `dtype` is a numpy dtype
    string. The attached instance skips __init__ because workers don't own the block,
    and `track=False` (Python 3.13+) keeps the resource tracker from unlinking it.
    """
    owned = SharedMemory._shms.get(name)
    if owned is not None:
        return owned

    in_worker = multiprocessing.parent_process() is not None
    cached = _ATTACHED.get(name) if in_worker else None
    if cached is not None and cached.shape == tuple(shape) and cached.dtype == np.dtype(dtype):
        return cached
    if in_worker:
        _release_other_runs(name)

    try:
        memory = shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        memory = shared_memory.SharedMemory(name=name)
    attached = SharedMemory.__new__(SharedMemory)
    attached.shm = memory
    attached.shape = tuple(shape)
    attached.dtype = np.dtype(dtype)
    if in_worker:
        _ATTACHED[name] = attached
    return attached


def _run_of(name: str) -> str | None:
    """The run id in a name made by run_block_name() ("mlpipe_<run id>_<kind>"), else None."""
    parts = name.split("_")
    return parts[1] if len(parts) >= 3 and parts[0] == "mlpipe" else None


def _release_other_runs(name: str) -> None:
    """A reused worker drops its handles to blocks from earlier runs. On Windows a
    block stays allocated while any process holds a handle, even after unlink.
    A handle that still has a live numpy view can't be closed now; it closes with the view."""
    run = _run_of(name)
    if run is None:
        return
    for old in [n for n in _ATTACHED if _run_of(n) not in (None, run)]:
        block = _ATTACHED.pop(old)
        try:
            block.shm.close()
        except BufferError:
            pass


def run_block_name(run_id: str, kind: str) -> str:
    """Unique name per run, so concurrent pipelines (or two runs on one machine)
    never collide on a shared memory block."""
    return f"mlpipe_{run_id}_{kind}"


SHM_DIR = Path("/dev/shm")


def check_shared_memory_space(nbytes: int) -> None:
    """On Linux, shared memory lives in /dev/shm. Writing past its size doesn't raise: the
    process is killed with SIGBUS. Docker limits /dev/shm to 64 MB unless --shm-size is set.
    Windows and macOS have no fixed-size shared memory filesystem, so nothing is checked there."""
    if not SHM_DIR.is_dir():
        return
    free = shutil.disk_usage(SHM_DIR).free
    if nbytes > free:
        raise MemoryError(
            f"Not enough shared memory: the data needs {nbytes / 2 ** 20:,.0f} MB but {SHM_DIR} has "
            f"{free / 2 ** 20:,.0f} MB free. In Docker, raise the container's shm_size (--shm-size).")


def create_shared_numpy(arr: np.ndarray, name: str) -> SharedMemory:
    """Copy `arr` into a new shared memory block called `name` and return its owner handle.

    Raises ValueError if the name is already in use and MemoryError if /dev/shm is too small.
    """
    if SharedMemory.validate(name):
        raise ValueError(f"{name} is already exists")
    check_shared_memory_space(arr.nbytes)
    memory = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shared_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=memory.buf)
    shared_array[:] = arr[:]
    shared = SharedMemory(memory=memory, shape=arr.shape, dtype=arr.dtype)
    return shared


def features_unpackage(shared_memory: SharedMemory, index: Sequence[int] | np.ndarray,
                       features_index: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
    """Rows `index`, columns `features_index` (None = all) in a single copy."""
    array = shared_memory.array
    if features_index is None:
        return array[index]
    return array[np.ix_(np.asarray(index), np.asarray(features_index))]


class ValidationType(StrEnum):
    """How a model was validated."""
    train_test_split = "train_test_split"
    k_folds = "k_folds"
    leave_one_out = "leave_one_out"


class FilteringCriteriaClassification(StrEnum):
    """Metric used to rank and filter classification results."""
    accuracy = "accuracy"
    f1 = "f1"
    recall = "recall"
    precision = "precision"

    @property
    def maximize(self) -> bool:
        """Whether a higher value is better (true for every classification metric)."""
        return self == FilteringCriteriaClassification.accuracy or self == FilteringCriteriaClassification.f1 or self == FilteringCriteriaClassification.recall or self == FilteringCriteriaClassification.precision


class FilteringCriteriaRegression(StrEnum):
    """Metric used to rank and filter regression results."""
    mae = "mae"
    mse = "mse"
    rmse = "rmse"
    r2 = "r2"

    @property
    def maximize(self) -> bool:
        """Whether a higher value is better (only r2; the error metrics are minimized)."""
        return self == FilteringCriteriaRegression.r2


class Sound(StrEnum):
    """Sound played when a training object goes out of scope after a finished run.

    none is silent; ping is one soft bell note; double is two rising notes, a happy
    "ding-ding"; chime is three bell notes ringing into a major chord; success is a quick
    rising arpeggio ending on a bright chord ("it worked!"); sparkle is a fast, playful
    twinkling run up high; alarm is a repeated high/low tone that is hard to miss; bell
    is the terminal bell only.
    """
    none = "none"
    ping = "ping"
    double = "double"
    chime = "chime"
    success = "success"
    sparkle = "sparkle"
    alarm = "alarm"
    bell = "bell"

    @property
    def notes(self) -> list[tuple[float, float, float]]:
        """(frequency Hz, start s, length s) per note; notes may overlap and ring together."""
        return _SOUND_NOTES.get(self, [])

    def wav_bytes(self, rate: int = 44_100) -> bytes:
        """The sound rendered as an in-memory 16-bit mono WAV file (b"" for none / bell).

        `rate` is the sample rate in Hz. Each note gets a bell / marimba timbre from a few
        partials, the higher ones dying away faster, with a 4 ms fade-in (no click) and a
        20 ms fade-out. The mix is normalized to half of full scale for a comfortable volume.
        """
        import io
        import wave
        notes = self.notes
        if not notes:
            return b""
        total = max(start + length for _, start, length in notes) + 0.05
        mix = np.zeros(int(total * rate))
        for freq, start, length in notes:
            t = np.arange(int(length * rate)) / rate
            tone = np.zeros_like(t)
            for k, (ratio, amp) in enumerate(_PARTIALS, start=1):
                tone += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t * k / (length * 0.35))
            attack = np.minimum(1, t / 0.004)
            release = np.minimum(1, (length - t) / 0.02)
            begin = int(start * rate)
            mix[begin:begin + len(t)] += tone * attack * release
        mix *= 0.5 / np.abs(mix).max()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes((mix * 32767).astype("<i2").tobytes())
        return buffer.getvalue()


_C5, _E5, _G5, _A5 = 523.25, 659.25, 783.99, 880.00
_C6, _D6, _E6, _G6, _A6, _C7 = 1046.50, 1174.66, 1318.51, 1567.98, 1760.00, 2093.00
_PARTIALS = [(1, 1.0), (2, 0.35), (3, 0.12), (4.2, 0.06)]

_SOUND_NOTES: dict[Sound, list[tuple[float, float, float]]] = {
    Sound.ping: [(_A5, 0, 0.8)],
    Sound.double: [(_G5, 0, 0.5), (_C6, 0.14, 0.9)],
    Sound.chime: [(_C6, 0, 1.4), (_E6, 0.15, 1.3), (_G6, 0.30, 1.5)],
    Sound.success: [(_C5, 0, 0.4), (_E5, 0.09, 0.4), (_G5, 0.18, 0.4), (_C6, 0.27, 0.5),
                    (_C5, 0.40, 1.3), (_E5, 0.40, 1.3), (_G5, 0.40, 1.3), (_C6, 0.40, 1.5), (_E6, 0.40, 1.5)],
    Sound.sparkle: [(f, i * 0.06, 0.6) for i, f in enumerate([_C6, _D6, _E6, _G6, _A6, _C7])]
                   + [(_C7, 0.42, 1.0), (_G6, 0.42, 1.0)],
    Sound.alarm: [(f, i * 0.2, 0.18) for i, f in enumerate([1200, 800] * 4)],
}


def is_fitted(model: Any) -> bool:
    """Whether `model` has been fitted, per sklearn's check_is_fitted.

    Objects that aren't sklearn estimators are assumed fitted; the caller knows.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except TypeError:
        return True


def create_path(path: str) -> Path:
    """Create the folder `path`, including any missing parents, and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


GC_EVERY_N_TASKS = 10
_tasks_since_gc = 0


def force_gc[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Full garbage collection every GC_EVERY_N_TASKS calls in this process.
    Collecting after every task can cost more than a small model's fit."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call `func`, then count the call and collect garbage when the count is reached."""
        global _tasks_since_gc
        try:
            return func(*args, **kwargs)
        finally:
            _tasks_since_gc += 1
            if _tasks_since_gc >= GC_EVERY_N_TASKS:
                _tasks_since_gc = 0
                gc.collect(2)

    return wrapper


def cleanup_shared_memory[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Unlink every registered shared memory block once `func` returns or raises."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call `func`, then release all shared memory blocks."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            SharedMemory.cleanup()

    return wrapper
