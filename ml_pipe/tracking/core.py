"""Execution tracking: every stage of a run becomes one structured record.

A record says what ran (stage_name, operation), where (module, location, pid, thread), when
(start_time, end_time, duration), how it ended (status) and, for a failure, why (error_type,
error_message, a plain-language reason and the full traceback).

    with ExecutionTracker("My experiment", output_dir="logs") as tracker:
        with tracker.track("Load data"):
            ...
        with tracker.track("Feature selection"):
            with tracker.track("Correlation"):      # nested: a child of "Feature selection"
                ...

Library code uses the module-level helpers (`stage`, `tracked`, `annotate`). They attach to the
tracker active in the calling context and do nothing - at the cost of one ContextVar lookup -
when no tracker is active, so they are safe in hot paths and in worker processes.

Statuses: PENDING (planned, not started), RUNNING, PASSED, FAILED (raised an exception),
SKIPPED (never ran, with the reason) and CANCELLED (interrupted: Ctrl+C, SystemExit, or still
running when the execution ended).

Exceptions are never swallowed: a failing stage is recorded and the original exception is
re-raised unchanged. Problems in the log outputs themselves (a full disk, a closed console) are
counted in `output_errors` and reported once through the `ml_pipe.execution` logger; they never
replace or hide the application's exception.
"""
from __future__ import annotations

import contextvars
import datetime
import functools
import itertools
import logging
import os
import re
import threading
import time
import traceback as tb_module
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from ml_pipe.tracking.log import ExecutionLog
    from ml_pipe.tracking.outputs import Output

logger = logging.getLogger("ml_pipe.execution")

# Longest error message / metadata value kept in a record (characters).
MAX_MESSAGE = 2000
MAX_VALUE = 300
# Exceptions that mean "stopped from outside", not "the step is broken".
_CANCEL_TYPES = (KeyboardInterrupt, SystemExit, GeneratorExit)


class Status(StrEnum):
    """Lifecycle of one stage."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"

    @property
    def done(self) -> bool:
        """True once the stage can no longer change."""
        return self not in (Status.PENDING, Status.RUNNING)


def _now() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


def _iso(moment: datetime.datetime | None) -> str | None:
    return None if moment is None else moment.isoformat(timespec="milliseconds")


@dataclass
class StageRecord:
    """Everything known about one stage. `to_dict()` is the JSON form written to the logs.

    location  "file.py:123 in function" where the exception was raised (FAILED only)
    reason    plain-language explanation for users: why it failed / was skipped / was cancelled
    origin    for a FAILED stage, the stage_id where the failure started (itself for a root cause;
              a parent re-raising a child's exception points at the child)
    """
    execution_id: str
    stage_id: str
    stage_name: str
    parent_stage_id: str | None
    level: int
    module: str = ""
    operation: str = ""
    status: Status = Status.PENDING
    start_time: str | None = None
    end_time: str | None = None
    duration: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    location: str | None = None
    reason: str | None = None
    origin: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None
    thread: str | None = None
    _clock: float | None = field(default=None, repr=False, compare=False)

    FIELDS = ("execution_id", "stage_id", "stage_name", "parent_stage_id", "level", "module", "operation",
              "status", "start_time", "end_time", "duration", "error_type", "error_message", "traceback",
              "location", "reason", "origin", "metadata", "pid", "thread")

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON-ready data (status as its string value)."""
        out = {name: getattr(self, name) for name in self.FIELDS}
        out["status"] = str(self.status)
        out["metadata"] = dict(self.metadata)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageRecord:
        """Rebuild a record from to_dict() output; unknown keys are ignored."""
        values = {k: data.get(k) for k in cls.FIELDS if k in data}
        values["status"] = Status(values.get("status") or Status.PENDING)
        values["metadata"] = dict(values.get("metadata") or {})
        return cls(**values)

    @property
    def is_root_cause(self) -> bool:
        """A FAILED stage where the failure started (not a parent that only re-raised it)."""
        return self.status is Status.FAILED and (self.origin in (None, self.stage_id))

    @property
    def start(self) -> datetime.datetime | None:
        return None if self.start_time is None else datetime.datetime.fromisoformat(self.start_time)

    @property
    def end(self) -> datetime.datetime | None:
        return None if self.end_time is None else datetime.datetime.fromisoformat(self.end_time)


# ---------------------------------------------------------------- privacy
_SENSITIVE_KEY = re.compile(r"pass(word|wd)?|secret|token|api[_-]?key|credential|auth|private[_-]?key|cookie|session",
                            re.IGNORECASE)
_SENSITIVE_TEXT = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|auth(?:orization)?|credential)s?"
    r"(\s*[=:]\s*|\s+)(['\"]?)[^\s'\",;&]+")
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@")


def redact_text(text: str) -> str:
    """Mask credentials in free text ("password=abc" -> "password=***", "https://u:p@host" -> "https://***@host")."""
    text = _URL_CREDENTIALS.sub(r"\1***@", text)
    return _SENSITIVE_TEXT.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}***", text)


def safe_value(value: Any, key: str = "") -> Any:
    """A JSON-safe, privacy-safe version of a metadata value.

    Keys that look like credentials are masked. Data containers (arrays, DataFrames, Series)
    are reduced to their type and shape: dataset contents never reach the logs. Long values
    are cut to MAX_VALUE characters."""
    if key and _SENSITIVE_KEY.search(key):
        return "***"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = redact_text(value)
        return text if len(text) <= MAX_VALUE else text[:MAX_VALUE] + "..."
    shape = getattr(value, "shape", None)
    if shape is not None and not isinstance(value, type):
        try:
            return f"<{type(value).__name__} shape={tuple(int(s) for s in shape)}>"
        except (TypeError, ValueError):
            return f"<{type(value).__name__}>"
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        try:
            return safe_value(value.item(), key)
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        items = list(value.items())
        out = {str(k): safe_value(v, str(k)) for k, v in items[:50]}
        if len(items) > 50:
            out["..."] = f"{len(items) - 50} more"
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out = [safe_value(v) for v in items[:50]]
        if len(items) > 50:
            out.append(f"... {len(items) - 50} more")
        return out
    if isinstance(value, (datetime.datetime, datetime.date, Path)):
        return str(value)
    return safe_value(repr(value))


# ---------------------------------------------------------------- explaining failures
_FRIENDLY: list[tuple[str, str]] = [  # (exception class name anywhere in the MRO, explanation)
    ("ConfigError", "The experiment configuration or its data is invalid"),
    ("MemoryError", "Not enough memory to finish this step"),
    ("BrokenProcessPool", "A worker process stopped unexpectedly (often out of memory)"),
    ("FileNotFoundError", "A required file or folder was not found"),
    ("PermissionError", "Permission denied while reading or writing a file"),
    ("TimeoutError", "The step took too long and timed out"),
    ("ModuleNotFoundError", "A required package is not installed"),
    ("ImportError", "A required package could not be imported"),
    ("KeyError", "A required column, key or setting was not found"),
    ("IndexError", "An index was out of range"),
    ("ZeroDivisionError", "A calculation divided by zero"),
    ("NotFittedError", "A model was used before it was trained"),
    ("ValueError", "Invalid value or data"),
    ("TypeError", "A value had the wrong type"),
    ("OSError", "An operating-system or file error occurred"),
    ("KeyboardInterrupt", "Cancelled by the user (Ctrl+C)"),
    ("SystemExit", "The program was asked to exit"),
]


def error_message(exc: BaseException) -> str:
    """str(exc) with credentials masked and capped at MAX_MESSAGE characters; the class name if empty."""
    try:
        text = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    except Exception:
        text = ""
    text = redact_text(text.strip()) or type(exc).__name__
    return text if len(text) <= MAX_MESSAGE else text[:MAX_MESSAGE] + "..."


def explain(exc: BaseException) -> str:
    """One plain-language sentence for users: what kind of problem it is, then the message."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    prefix = next((text for name, text in _FRIENDLY if name in names), "An unexpected error occurred")
    message = error_message(exc)
    first_line = message.splitlines()[0] if message else ""
    if first_line in ("", type(exc).__name__):
        return prefix + "."
    if len(first_line) > 300:
        first_line = first_line[:300] + "..."
    return f"{prefix}: {first_line}"


def _location(exc: BaseException) -> str | None:
    """'file.py:line in function' of the frame that raised (the worker's frame for pool tasks)."""
    task = getattr(exc, "ml_pipe_task", None)
    if isinstance(task, dict) and task.get("location"):
        return task["location"]
    frames = tb_module.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    if not frames:
        return None
    frame = frames[-1]
    return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"


def location_of(frames: list[tb_module.FrameSummary]) -> str | None:
    if not frames:
        return None
    frame = frames[-1]
    return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"


def _project_location(exc: BaseException) -> str | None:
    """The innermost frame inside the ml_pipe package (where our code called into a library)."""
    if exc.__traceback__ is None:
        return None
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inside = [f for f in tb_module.extract_tb(exc.__traceback__) if os.path.abspath(f.filename).startswith(package)
              and os.sep + "tracking" + os.sep not in f.filename]
    return location_of(inside)


def format_traceback(exc: BaseException) -> str:
    """The full traceback, including chained causes (for pool tasks: the worker's traceback)."""
    try:
        return "".join(tb_module.format_exception(exc)).rstrip()
    except Exception:  # pragma: no cover - a broken __str__ etc.
        return f"{type(exc).__name__} (traceback unavailable)"


# ---------------------------------------------------------------- context
# (tracker, stage_id) of the stage the current thread / task is inside.
_CURRENT: contextvars.ContextVar[tuple[ExecutionTracker, str] | None] = contextvars.ContextVar(
    "ml_pipe_current_stage", default=None)
_ACTIVE: list[ExecutionTracker] = []
_ACTIVE_LOCK = threading.Lock()


def current_tracker() -> ExecutionTracker | None:
    """The tracker of the calling context, or None (tracking is then off).

    A thread that did not inherit a context (e.g. a background report thread) falls back to the
    one tracker active in this process; with several active trackers it gets None rather than
    risk attaching to another execution's log."""
    current = _CURRENT.get()
    if current is not None and not current[0].finished:
        return current[0]
    with _ACTIVE_LOCK:
        return _ACTIVE[0] if len(_ACTIVE) == 1 else None


def current_stage() -> tuple[ExecutionTracker | None, str | None]:
    """(tracker, stage_id) the caller is inside, or (None, None)."""
    current = _CURRENT.get()
    if current is not None and not current[0].finished:
        return current
    tracker = current_tracker()
    if tracker is None or tracker.root is None:
        return None, None
    return tracker, tracker.root.stage_id


# ---------------------------------------------------------------- stages
class Stage:
    """A tracked block: `with tracker.track(...) as stage:`; usable once.

    stage.annotate(key=value) adds metadata; stage.skip(reason) ends it as SKIPPED instead of PASSED
    (for a step that decides it has nothing to do)."""
    __slots__ = ("tracker", "record", "_token", "_skip_reason", "_args")

    def __init__(self, tracker: ExecutionTracker, args: dict[str, Any]) -> None:
        self.tracker = tracker
        self.record: StageRecord | None = None
        self._token = None
        self._skip_reason: str | None = None
        self._args = args

    @property
    def stage_id(self) -> str | None:
        return None if self.record is None else self.record.stage_id

    def annotate(self, **metadata: Any) -> None:
        if self.record is not None:
            self.tracker._annotate(self.record, metadata)

    def skip(self, reason: str) -> None:
        self._skip_reason = reason

    def __enter__(self) -> Stage:
        if self.record is not None:
            raise RuntimeError("a Stage can be entered only once; call tracker.track() again")
        self.record = self.tracker._begin(**self._args)
        if self.record is not None:
            self._token = _CURRENT.set((self.tracker, self.record.stage_id))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            try:
                _CURRENT.reset(self._token)
            except ValueError:  # exited in another context (e.g. a generator moved threads)
                _CURRENT.set(None)
            self._token = None
        if self.record is not None:
            self.tracker._end(self.record, exc, self._skip_reason)
        return False  # never swallow the exception


class _NullStage:
    """What `stage()` returns when tracking is off: does nothing, costs nothing."""
    __slots__ = ()
    record = None
    stage_id = None
    tracker = None

    def annotate(self, **metadata: Any) -> None:
        pass

    def skip(self, reason: str) -> None:
        pass

    def __enter__(self) -> _NullStage:
        return self

    def __exit__(self, *exc) -> bool:
        return False


NULL_STAGE = _NullStage()


class ExecutionTracker:
    """Records every stage of one execution and sends each change to the outputs.

    name           the root stage (level 0); every other stage descends from it
    output_dir     folder for execution_<id>.log (developer log), .jsonl (one JSON event per line,
                   written as it happens) and, at the end, .json (final records) and _summary.txt.
                   Can also be added later with add_output_dir().
    console        True (stderr), a stream, a callable taking one line (e.g. print or a logger
                   method), or False for no console output
    verbose        console also shows tracebacks and metadata (the developer view)
    console_level  deepest stage level shown live on the console (None = all); the summary always
                   covers every stage
    task_records   one PASSED record per process-pool task (per configuration trained). Off by
                   default: a run can train tens of thousands of configurations; each batch
                   records counts, worker pids and worker time instead. Failed tasks are always
                   recorded.

    Thread-safe. Use as a context manager, or call start() and finish().
    """

    def __init__(self, name: str = "Execution", *, output_dir: str | Path | None = None,
                 console: bool | Any = True, verbose: bool = False, console_level: int | None = None,
                 task_records: bool = False, module: str = "", operation: str = "",
                 metadata: dict[str, Any] | None = None, outputs: list[Output] = ()) -> None:
        self.execution_id = uuid.uuid4().hex[:12]
        self.name = name
        self.task_records = task_records
        self.module = module
        self.operation = operation
        self._root_metadata = metadata or {}
        self._lock = threading.RLock()
        self._records: dict[str, StageRecord] = {}
        self._children: dict[str | None, list[str]] = {}
        self._pending: dict[tuple[str | None, str], list[StageRecord]] = {}  # plan(): (parent, name) -> records
        self._counter = itertools.count(0)
        self._outputs: list[Output] = []
        self.output_errors: list[str] = []
        self._failed_outputs: set[int] = set()
        self.root: StageRecord | None = None
        self.finished = False
        self._root_token = None
        self.output_paths: dict[str, Path] = {}
        if console:
            from ml_pipe.tracking.outputs import ConsoleOutput
            self.add_output(console if isinstance(console, ConsoleOutput) else
                            ConsoleOutput(None if console is True else console, verbose=verbose,
                                          max_level=console_level))
        for output in outputs:
            self.add_output(output)
        if output_dir is not None:
            self.add_output_dir(output_dir)

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> ExecutionTracker:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(exc)
        return False

    def start(self) -> ExecutionTracker:
        """Open the root stage and make this the active tracker of the calling context."""
        if self.root is not None:
            raise RuntimeError("this tracker was already started")
        with _ACTIVE_LOCK:
            _ACTIVE.append(self)
        self.root = self._begin(self.name, module=self.module, operation=self.operation or self.name,
                                metadata=self._root_metadata, parent=None)
        self._root_token = _CURRENT.set((self, self.root.stage_id))
        return self

    def finish(self, exc: BaseException | None = None) -> None:
        """Close the root stage (PASSED, FAILED or CANCELLED), settle every open stage, write the summary.

        Planned stages that never ran become SKIPPED with the reason; stages still RUNNING (e.g.
        a background job that timed out) become CANCELLED. Safe to call twice."""
        if self.finished or self.root is None:
            return
        if self._root_token is not None:
            try:
                _CURRENT.reset(self._root_token)
            except ValueError:
                _CURRENT.set(None)
            self._root_token = None
        try:
            self._settle(exc)
            self._end(self.root, exc, None)
        finally:
            self.finished = True
            with _ACTIVE_LOCK:
                if self in _ACTIVE:
                    _ACTIVE.remove(self)
            log = self.log()
            self._emit("execution_finished", self, log)
            self._emit("close")

    def _settle(self, exc: BaseException | None) -> None:
        with self._lock:
            open_records = [r for r in self._records.values() if not r.status.done and r is not self.root]
        cause = self._first_root_cause()
        for record in open_records:
            if record.status is Status.RUNNING:
                record.reason = ("still running when the execution ended"
                                 + (" (cancelled)" if isinstance(exc, _CANCEL_TYPES) else ""))
                self._close(record, Status.CANCELLED)
            else:
                if record.reason is None:
                    if isinstance(exc, _CANCEL_TYPES):
                        record.reason = "Not run: the execution was cancelled."
                    elif cause is not None:
                        record.reason = f"Not run because '{cause.stage_name}' failed."
                    else:
                        record.reason = "Not run: not needed in this execution."
                self._close(record, Status.SKIPPED, started=False)

    # ------------------------------------------------------------ outputs
    def add_output(self, output: Output) -> Output:
        """Attach an output; it first receives every stage recorded so far (replay)."""
        with self._lock:
            self._outputs.append(output)
            try:
                output.attach(self)
            except Exception as e:  # noqa: BLE001 - an output must never break the run
                self._output_failed(output, e)
        return output

    def add_output_dir(self, folder: str | Path) -> dict[str, Path]:
        """Write the developer log, the JSON event stream and (at the end) the summary into `folder`."""
        from ml_pipe.tracking.outputs import JsonlOutput, LogFileOutput, SummaryFiles
        folder = Path(folder)
        stem = folder / f"execution_{self.execution_id}"
        paths = {"log": stem.with_suffix(".log"), "jsonl": stem.with_suffix(".jsonl"),
                 "json": stem.with_suffix(".json"), "summary": folder / f"execution_{self.execution_id}_summary.txt"}
        self.add_output(LogFileOutput(paths["log"]))
        self.add_output(JsonlOutput(paths["jsonl"]))
        self.add_output(SummaryFiles(paths["json"], paths["summary"]))
        self.output_paths.update(paths)
        return paths

    def _emit(self, event: str, *args: Any) -> None:
        for output in list(self._outputs):
            if id(output) in self._failed_outputs and event != "close":
                continue
            try:
                getattr(output, event)(*args)
            except Exception as e:  # noqa: BLE001 - logging must not hide the application's error
                self._output_failed(output, e)

    def _output_failed(self, output: Output, error: Exception) -> None:
        message = f"{type(output).__name__} stopped: {type(error).__name__}: {error}"
        self.output_errors.append(message)
        if id(output) not in self._failed_outputs:
            self._failed_outputs.add(id(output))
            logger.warning("execution log output failed (the run continues): %s", message)

    # ------------------------------------------------------------ stages
    def track(self, name: str, *, module: str | None = None, operation: str = "", parent: str | None = None,
              **metadata: Any) -> Stage:
        """A context manager (or, applied to a function, a decorator) that records one stage.

        parent: explicit parent stage_id - for code running in another thread, where the
        caller's stage is not visible. Default: the stage the caller is inside."""
        args = {"name": name, "module": module, "operation": operation, "parent": parent, "metadata": metadata}
        return Stage(self, args)

    def stage(self, name: str | None = None, **kwargs: Any) -> Callable:
        """Decorator form: @tracker.stage("Training") records every call as a stage."""
        def decorate(func: Callable) -> Callable:
            label = name or func.__qualname__
            options = dict(kwargs)
            options["module"] = options.get("module") or func.__module__

            @functools.wraps(func)
            def wrapper(*args, **kw):
                with self.track(label, **options):
                    return func(*args, **kw)
            return wrapper
        return decorate

    def plan(self, *names: str, parent: str | None = None, module: str = "") -> list[StageRecord]:
        """Announce stages that will run under `parent` (default: the current stage) as PENDING.

        track() with the same name under the same parent picks the planned record up; any still
        PENDING when the execution ends become SKIPPED, with the failure that prevented them."""
        parent_id = self._parent_id(parent)
        records = [self._new_record(name, parent_id, module=module) for name in names]
        with self._lock:
            for record in records:
                self._pending.setdefault((parent_id, record.stage_name), []).append(record)
        return records

    def skip(self, name: str, reason: str, *, parent: str | None = None, module: str = "") -> StageRecord:
        """Record (or mark a planned) stage as SKIPPED with a reason users can read."""
        parent_id = self._parent_id(parent)
        record = self._planned(name, parent_id) or self._new_record(name, parent_id, module=module)
        record.reason = reason
        self._close(record, Status.SKIPPED, started=False)
        return record

    def record(self, name: str, status: Status | str, *, parent: str | None = None, module: str = "",
               operation: str = "", duration: float | None = None, start: float | None = None,
               error_type: str | None = None, error_message: str | None = None, traceback: str | None = None,
               location: str | None = None, reason: str | None = None, pid: int | None = None,
               **metadata: Any) -> StageRecord:
        """Add a stage that already ended elsewhere (a worker task, a plot). `start` = time.time()."""
        parent_id = self._parent_id(parent)
        record = self._new_record(name, parent_id, module=module, operation=operation, metadata=metadata)
        started = _now() if start is None else datetime.datetime.fromtimestamp(start).astimezone()
        record.start_time = _iso(started)
        record.pid = pid if pid is not None else os.getpid()
        record.thread = threading.current_thread().name
        self._emit("stage_started", record)
        record.error_type, record.error_message, record.traceback = error_type, error_message, traceback
        record.location, record.reason = location, reason
        record.duration = duration
        record.end_time = _iso(started + datetime.timedelta(seconds=duration or 0.0))
        status = Status(status)
        if status is Status.FAILED:
            record.origin = record.stage_id
        self._close(record, status, timed=False)
        return record

    def annotate(self, **metadata: Any) -> None:
        """Add metadata to the stage the caller is inside."""
        tracker, stage_id = current_stage()
        if tracker is self and stage_id in self._records:
            self._annotate(self._records[stage_id], metadata)

    def _annotate(self, record: StageRecord, metadata: dict[str, Any]) -> None:
        with self._lock:
            for key, value in metadata.items():
                record.metadata[str(key)] = safe_value(value, str(key))

    def _parent_id(self, parent: str | None) -> str | None:
        if parent is not None:
            return parent
        current = _CURRENT.get()
        if current is not None and current[0] is self:
            return current[1]
        return None if self.root is None else self.root.stage_id

    def _planned(self, name: str, parent_id: str | None) -> StageRecord | None:
        """The first planned, still PENDING stage `name` under `parent_id` (O(1): plans are indexed)."""
        with self._lock:
            queue = self._pending.get((parent_id, name))
            while queue:
                record = queue.pop(0)
                if record.status is Status.PENDING:
                    return record
        return None

    def _new_record(self, name: str, parent_id: str | None, *, module: str = "", operation: str = "",
                    metadata: dict[str, Any] | None = None) -> StageRecord:
        with self._lock:
            parent = self._records.get(parent_id) if parent_id is not None else None
            stage_id = f"{self.execution_id}.{next(self._counter)}"
            record = StageRecord(self.execution_id, stage_id, str(name), parent_id,
                                 0 if parent is None else parent.level + 1, module=module or "",
                                 operation=operation or "")
            if metadata:
                self._annotate(record, metadata)
            self._records[stage_id] = record
            self._children.setdefault(parent_id, []).append(stage_id)
            return record

    def _begin(self, name: str, module: str | None, operation: str, parent: str | None,
               metadata: dict[str, Any]) -> StageRecord | None:
        if self.finished:
            return None
        parent_id = self._parent_id(parent)
        with self._lock:
            record = self._planned(name, parent_id)
            if record is None:
                record = self._new_record(name, parent_id)
            record.module = module if module is not None else (record.module or _caller_module())
            record.operation = operation or record.operation or name
            if metadata:
                self._annotate(record, metadata)
            record.status = Status.RUNNING
            record.start_time = _iso(_now())
            record._clock = time.perf_counter()
            record.pid = os.getpid()
            record.thread = threading.current_thread().name
            self._emit("stage_started", record)
        return record

    def _end(self, record: StageRecord, exc: BaseException | None, skip_reason: str | None) -> None:
        try:
            if exc is None:
                if skip_reason is not None:
                    record.reason = skip_reason
                    self._close(record, Status.SKIPPED)
                else:
                    self._close(record, Status.PASSED)
                return
            self._describe_failure(record, exc)
            self._close(record, Status.CANCELLED if isinstance(exc, _CANCEL_TYPES) else Status.FAILED)
        except Exception as e:  # noqa: BLE001 - never replace the application's exception
            self._output_failed(_TrackerItself(), e)

    def _describe_failure(self, record: StageRecord, exc: BaseException) -> None:
        origin = getattr(exc, "__ml_pipe_origin__", None)
        cancelled = isinstance(exc, _CANCEL_TYPES)
        record.error_type = type(exc).__name__
        record.error_message = error_message(exc)
        child = self._records.get(origin[1]) if isinstance(origin, tuple) and origin[0] == self.execution_id else None
        if child is not None and child is not record:
            # the exception comes from a child stage: point at it instead of repeating the traceback
            record.origin = child.origin or child.stage_id
            source = self._records.get(record.origin, child)
            record.location = source.location
            record.reason = (f"Cancelled while '{source.stage_name}' was running." if cancelled else
                             f"Stopped because '{source.stage_name}' failed: {source.reason or record.error_message}")
            record.metadata["failed_child"] = source.stage_id
            return
        record.origin = record.stage_id
        record.traceback = format_traceback(exc)
        record.location = _location(exc)
        project = _project_location(exc)
        if project and project != record.location:
            record.metadata["project_location"] = project
        task = getattr(exc, "ml_pipe_task", None)
        if isinstance(task, dict):
            record.metadata["task"] = safe_value(task)
        record.reason = explain(exc)
        try:
            exc.__ml_pipe_origin__ = (self.execution_id, record.stage_id)
        except (AttributeError, TypeError):  # exceptions with __slots__
            pass

    def _close(self, record: StageRecord, status: Status, *, started: bool = True, timed: bool = True) -> None:
        with self._lock:
            if record.status.done:
                return  # already closed (e.g. settled at finish, then a late thread exits)
            record.status = status
            if started and timed:
                end = _now()
                record.end_time = _iso(end)
                if record._clock is not None:
                    record.duration = round(time.perf_counter() - record._clock, 6)
            self._emit("stage_finished", record)

    def _first_root_cause(self) -> StageRecord | None:
        with self._lock:
            return next((r for r in self._records.values() if r.is_root_cause), None)

    # ------------------------------------------------------------ pool tasks
    def task_failed(self, exc: BaseException, parent: str | None = None) -> None:
        """Record the pool task an exception came from (its ml_pipe_task context) as a FAILED child stage.

        The exception is only annotated, never caught: the caller re-raises it."""
        task = getattr(exc, "ml_pipe_task", None)
        if not isinstance(task, dict) or getattr(exc, "__ml_pipe_origin__", None):
            return
        try:
            name = f"Task: {task.get('model', 'task')} #{task.get('candidate_id')}"
            record = self.record(name, Status.FAILED, parent=parent, module=task.get("module", ""),
                                 operation=task.get("operation", "train one configuration"),
                                 duration=task.get("duration"), start=task.get("start"), pid=task.get("pid"),
                                 error_type=type(exc).__name__, error_message=error_message(exc),
                                 traceback=format_traceback(exc), location=task.get("location"),
                                 reason=explain(exc), task=task)
            exc.__ml_pipe_origin__ = (self.execution_id, record.stage_id)
        except Exception as e:  # noqa: BLE001
            self._output_failed(_TrackerItself(), e)

    def tasks_done(self, outcomes: list, parent: str | None = None) -> list:
        """Unwrap TrackedTask outcomes: record batch statistics (and per-task stages if task_records)."""
        results, pids, worker_seconds = [], set(), 0.0
        parent_id = self._parent_id(parent)
        for outcome in outcomes:
            result, info = outcome
            results.append(result)
            pids.add(info[0])
            worker_seconds += info[2]
            if self.task_records:
                self.record(f"Task: {getattr(result, 'name', 'task')} #{getattr(result, 'candidate_id', '?')}",
                            Status.PASSED, parent=parent_id, module="ml_pipe.workers",
                            operation="train one configuration", start=info[1], duration=round(info[2], 6),
                            pid=info[0])
        if parent_id in self._records:
            self._annotate(self._records[parent_id], {"tasks": len(results), "worker_pids": sorted(pids),
                                                      "worker_seconds": round(worker_seconds, 3)})
        return results

    # ------------------------------------------------------------ reading
    def records(self) -> list[StageRecord]:
        """Every record so far, in creation order (live objects; don't modify them)."""
        with self._lock:
            return list(self._records.values())

    def get(self, stage_id: str) -> StageRecord | None:
        return self._records.get(stage_id)

    def log(self) -> ExecutionLog:
        """An ExecutionLog snapshot: filter, tree, summary and details."""
        from ml_pipe.tracking.log import ExecutionLog
        with self._lock:
            records = [StageRecord.from_dict(r.to_dict()) for r in self._records.values()]
        return ExecutionLog(records, finished=self.finished)

    @property
    def status(self) -> Status | None:
        return None if self.root is None else self.root.status

    def __repr__(self) -> str:
        return f"ExecutionTracker({self.name!r}, id={self.execution_id}, stages={len(self._records)})"


class _TrackerItself:
    """Stands in for an output when the tracker's own bookkeeping fails."""


def _caller_module() -> str:
    """Module of the first frame outside this package (for stages opened without module=)."""
    import sys
    frame = sys._getframe(2)
    here = __name__.rsplit(".", 1)[0]
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith(here) and name != "contextlib":
            return name
        frame = frame.f_back
    return ""


# ---------------------------------------------------------------- module-level helpers
def stage(name: str, **kwargs: Any) -> Stage | _NullStage:
    """`with stage("Load data"):` - a stage of the active tracker, or a no-op when tracking is off."""
    tracker = current_tracker()
    if tracker is None:
        return NULL_STAGE
    kwargs.setdefault("module", _caller_module())
    return tracker.track(name, **kwargs)


def tracked(name: str | None = None, *, operation: str = "", **metadata: Any) -> Callable:
    """Decorator: each call is a stage of the active tracker (plain call when tracking is off)."""
    def decorate(func: Callable) -> Callable:
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kw):
            tracker = current_tracker()
            if tracker is None:
                return func(*args, **kw)
            with tracker.track(label, module=func.__module__, operation=operation or label, **metadata):
                return func(*args, **kw)
        return wrapper
    return decorate


def annotate(**metadata: Any) -> None:
    """Add metadata to the caller's current stage (no-op when tracking is off)."""
    tracker, stage_id = current_stage()
    if tracker is not None:
        record = tracker.get(stage_id)
        if record is not None:
            tracker._annotate(record, metadata)


@contextmanager
def execution(name: str, *, module: str = "", operation: str = "", metadata: dict[str, Any] | None = None,
              **tracker_kwargs: Any) -> Iterator[ExecutionTracker]:
    """A stage of the active tracker if there is one, else a new ExecutionTracker whose root is this stage.

    Entry points (Experiment.run, BaseML.k_folds, ...) use it so they are tracked on their own and
    nest when called inside a bigger tracked execution. Yields the tracker."""
    tracker = current_tracker()
    if tracker is not None:
        with tracker.track(name, module=module, operation=operation, **(metadata or {})):
            yield tracker
        return
    with ExecutionTracker(name, module=module, operation=operation, metadata=metadata, **tracker_kwargs) as tracker:
        yield tracker
