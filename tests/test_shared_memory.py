import numpy as np
import pytest

# Change this import to your actual file location
from utils.utils import SharedMemory, create_shared_numpy, cleanup_shared_memory


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Prevent shared memory leaks between tests.
    """
    yield
    SharedMemory.cleanup()


def test_shared_memory_roundtrip():
    original = np.array(
        [[1, 2], [3, 4]],
        dtype=np.int32
    )

    shm = create_shared_numpy(
        original,
        "test_roundtrip"
    )

    try:
        loaded = shm.array

        assert np.array_equal(loaded, original)

    finally:
        shm.unlink()


def test_shared_memory_write():
    original = np.zeros(
        (3, 3),
        dtype=np.int32
    )

    shm = create_shared_numpy(
        original,
        "test_write"
    )

    try:
        shm.array[0, 0] = 999

        assert shm.array[0, 0] == 999

    finally:
        shm.unlink()


def test_shared_memory_shape():
    original = np.zeros(
        (5, 7),
        dtype=np.float32
    )

    shm = create_shared_numpy(
        original,
        "test_shape"
    )

    try:
        assert shm.shape == original.shape

    finally:
        shm.unlink()


def test_shared_memory_dtype():
    original = np.zeros(
        (5, 5),
        dtype=np.float64
    )

    shm = create_shared_numpy(
        original,
        "test_dtype"
    )

    try:
        assert shm.dtype == original.dtype

    finally:
        shm.unlink()


def test_shared_memory_repr():
    original = np.zeros((2, 2))

    shm = create_shared_numpy(
        original,
        "test_repr"
    )

    try:
        text = repr(shm)

        assert "SharedMemory" in text
        assert shm.shm.name in text

    finally:
        shm.unlink()


# -------------------------
# SharedMemory lifecycle
# -------------------------

def test_shared_memory_is_registered():
    original = np.ones((2, 2))

    shm = create_shared_numpy(original, "test_registered")

    try:
        assert shm in list(SharedMemory._shms.values())
    finally:
        shm.unlink()


def test_shared_memory_unlink_removes_registry():
    original = np.ones((2, 2))

    shm = create_shared_numpy(original, "test_unlink")

    assert shm in list(SharedMemory._shms.values())

    shm.unlink()

    assert shm not in list(SharedMemory._shms.values())


def test_shared_memory_cleanup():
    create_shared_numpy(np.ones((2, 2)), "test_cleanup_1")

    create_shared_numpy(np.zeros((3, 3)), "test_cleanup_2")

    assert len(SharedMemory._shms) > 0

    SharedMemory.cleanup()

    assert SharedMemory._shms == {}


def test_multiple_shared_memory_objects():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])

    shm1 = create_shared_numpy(arr1, "test_multiple_1")

    shm2 = create_shared_numpy(arr2, "test_multiple_2")

    try:
        assert np.array_equal(shm1.array, arr1)
        assert np.array_equal(shm2.array, arr2)

    finally:
        shm1.unlink()
        shm2.unlink()


# -------------------------
# cleanup_shared_memory decorator tests
# -------------------------

def test_cleanup_decorator_after_success():
    @cleanup_shared_memory
    def create_memory():
        arr = np.ones((10, 10))

        shm = create_shared_numpy(arr, "decorator_success")

        assert len(SharedMemory._shms) > 0

        return shm.array.copy()

    result = create_memory()

    assert np.array_equal(result, np.ones((10, 10)))

    assert SharedMemory._shms == {}


def test_cleanup_decorator_after_exception():
    @cleanup_shared_memory
    def failing_function():
        create_shared_numpy(np.zeros((5, 5)), "decorator_exception")

        assert len(SharedMemory._shms) > 0

        raise RuntimeError("expected error")

    with pytest.raises(            RuntimeError,            match="expected error"    ):
        failing_function()

    assert SharedMemory._shms == {}


def test_cleanup_decorator_keeps_return_value():
    @cleanup_shared_memory
    def return_value():
        create_shared_numpy(np.array([1, 2, 3]), "decorator_return")

        return 42

    result = return_value()

    assert result == 42
    assert SharedMemory._shms == {}


def test_cleanup_decorator_passes_arguments():
    @cleanup_shared_memory
    def create_with_argument(size):
        arr = np.zeros((size, size))

        create_shared_numpy(arr, "decorator_arguments")

        return arr.shape

    result = create_with_argument(7)

    assert result == (7, 7)
    assert SharedMemory._shms == {}


def test_cleanup_decorator_multiple_shared_memory():
    @cleanup_shared_memory
    def create_multiple():
        create_shared_numpy(np.ones((2, 2)), "decorator_multi_1")

        create_shared_numpy(np.zeros((3, 3)), "decorator_multi_2")

        assert len(SharedMemory._shms) > 0

    create_multiple()

    assert SharedMemory._shms == {}

def test_add_two_sherd_memoris_with_the_same_name():
    def create_memory():
        create_shared_numpy(np.ones((2, 2)), "decorator_multi_1")
        create_shared_numpy(np.zeros((3, 3)), "decorator_multi_1")

    with pytest.raises(ValueError, match="decorator_multi_1 is already exists"):
        create_memory()


# -------------------------
# worker side: attach by name, cache per worker, drop older runs
# -------------------------
import pickle

import utils.utils as uu


@pytest.fixture
def as_worker(monkeypatch):
    """Make this process look like a pool worker and give it an empty attach cache."""
    monkeypatch.setattr(uu.multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(uu, "_ATTACHED", {})
    yield uu._ATTACHED
    for block in list(uu._ATTACHED.values()):
        try:
            block.shm.close()
        except BufferError:
            pass


def test_run_block_name_round_trip():
    name = uu.run_block_name("abc123", "features")
    assert name == "mlpipe_abc123_features"
    assert uu._run_of(name) == "abc123"
    assert uu._run_of("features") is None  # names not made by run_block_name


def test_worker_attaches_without_copying_and_caches(as_worker):
    owner = create_shared_numpy(np.arange(6.0).reshape(2, 3), uu.run_block_name("run1", "x"))
    SharedMemory._shms.pop(owner.shm.name)  # the worker doesn't see the parent's registry
    try:
        payload = pickle.dumps(owner)
        first, second = pickle.loads(payload), pickle.loads(payload)
        assert first is second  # attached once per worker
        np.testing.assert_array_equal(first.array, owner.array)
        owner.array[0, 0] = 99  # same memory, not a copy
        assert first.array[0, 0] == 99
        assert owner.shm.name in as_worker
    finally:
        owner.unlink()


def test_worker_drops_blocks_of_older_runs(as_worker):
    old = create_shared_numpy(np.zeros(3), uu.run_block_name("run1", "x"))
    new = create_shared_numpy(np.ones(3), uu.run_block_name("run2", "x"))
    for block in (old, new):
        SharedMemory._shms.pop(block.shm.name)
    try:
        pickle.loads(pickle.dumps(old))
        pickle.loads(pickle.dumps(new))  # a new run arrives: run1's handle is released
        assert list(as_worker) == [new.shm.name]
    finally:
        old.unlink()
        new.unlink()


def test_owner_process_does_not_cache(monkeypatch):
    monkeypatch.setattr(uu, "_ATTACHED", {})
    shm = create_shared_numpy(np.zeros(2), uu.run_block_name("run1", "y"))
    assert pickle.loads(pickle.dumps(shm)) is shm  # owner gets its own object back
    assert uu._ATTACHED == {}


# -------------------------
# force_gc runs a full collection every GC_EVERY_N_TASKS calls
# -------------------------

def test_force_gc_collects_every_n_tasks(monkeypatch):
    collections = []
    monkeypatch.setattr(uu.gc, "collect", lambda generation=2: collections.append(generation))
    monkeypatch.setattr(uu, "_tasks_since_gc", 0)
    monkeypatch.setattr(uu, "GC_EVERY_N_TASKS", 3)

    task = uu.force_gc(lambda x: x * 2)
    assert [task(i) for i in range(7)] == [0, 2, 4, 6, 8, 10, 12]
    assert collections == [2, 2]  # after the 3rd and 6th task


def test_force_gc_counts_failed_tasks(monkeypatch):
    monkeypatch.setattr(uu, "_tasks_since_gc", 0)

    @uu.force_gc
    def boom():
        raise RuntimeError("task failed")

    with pytest.raises(RuntimeError, match="task failed"):
        boom()
    assert uu._tasks_since_gc == 1


# -------------------------
# /dev/shm too small (Docker's 64 MB default): clear error instead of a SIGBUS crash
# -------------------------

def test_too_little_shared_memory_raises_clear_error(monkeypatch, tmp_path):
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(uu, "SHM_DIR", tmp_path)  # pretend this is /dev/shm
    monkeypatch.setattr(uu.shutil, "disk_usage", lambda path: usage(64 * 2 ** 20, 0, 64 * 2 ** 20))
    with pytest.raises(MemoryError, match="needs 76 MB but .* has 64 MB free.*shm_size"):
        create_shared_numpy(np.zeros((10_000, 1_000)), "too_big")
    assert not SharedMemory.validate("too_big")


def test_enough_shared_memory_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(uu, "SHM_DIR", tmp_path)
    shm = create_shared_numpy(np.zeros(10), "fits")
    shm.unlink()


def test_no_shm_dir_skips_check(monkeypatch, tmp_path):
    monkeypatch.setattr(uu, "SHM_DIR", tmp_path / "missing")
    uu.check_shared_memory_space(10 ** 15)  # Windows/macOS: nothing to check
