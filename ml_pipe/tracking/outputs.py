"""Where execution records go: the console, a developer log file, a JSON event stream, summary files.

Every output receives the same calls from the tracker, always under the tracker's lock (so lines
from concurrent threads never interleave inside one output):

    attach(tracker)                      once, then a replay of the stages recorded so far
    stage_started(record) / stage_finished(record)
    execution_finished(tracker, log)     once, with the final ExecutionLog
    close()

Files are opened in append mode, written as UTF-8 and flushed after every record, so a crash or
Ctrl+C still leaves a readable log up to the last stage. Only the parent process writes.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

from ml_pipe.tracking.core import Status, StageRecord

if TYPE_CHECKING:
    from ml_pipe.tracking.core import ExecutionTracker
    from ml_pipe.tracking.log import ExecutionLog

LABEL = {Status.PENDING: "PLAN", Status.RUNNING: "START", Status.PASSED: "PASS", Status.FAILED: "FAIL",
         Status.SKIPPED: "SKIP", Status.CANCELLED: "CANCEL"}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


class Output:
    """Base class: every hook does nothing."""

    def attach(self, tracker: ExecutionTracker) -> None:
        for record in tracker.records():
            if record.start_time is not None:
                self.stage_started(record)
            if record.status.done:
                self.stage_finished(record)

    def stage_started(self, record: StageRecord) -> None:
        pass

    def stage_finished(self, record: StageRecord) -> None:
        pass

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        pass

    def close(self) -> None:
        pass


class ConsoleOutput(Output):
    """User-facing live view: one line when a stage starts and one when it ends, indented by level.

    target    None (sys.stderr at write time), a text stream, or a callable taking one line
    verbose   developer view: tracebacks and metadata too
    max_level deepest level shown live (None = all); the summary at the end lists everything

    Failures show the plain-language reason and where they happened; raw tracebacks only in
    verbose mode (they are always in the log file)."""

    def __init__(self, target: TextIO | Callable[[str], Any] | None = None, *, verbose: bool = False,
                 max_level: int | None = None, show_start: bool = True) -> None:
        self.target = target
        self.verbose = verbose
        self.max_level = max_level
        self.show_start = show_start
        self._tracker: ExecutionTracker | None = None

    def attach(self, tracker: ExecutionTracker) -> None:
        self._tracker = tracker  # no replay: the console only shows what happens from now on

    def _write(self, text: str) -> None:
        target = self.target if self.target is not None else sys.stderr
        if callable(target) and not hasattr(target, "write"):
            try:
                target(text)
            except UnicodeEncodeError:  # print() to a console that can't show some character
                target(text.encode("ascii", "replace").decode("ascii"))
            return
        try:
            target.write(text + "\n")
        except UnicodeEncodeError:
            encoding = getattr(target, "encoding", None) or "ascii"
            target.write(text.encode(encoding, "replace").decode(encoding) + "\n")
        flush = getattr(target, "flush", None)
        if flush is not None:
            flush()

    def _visible(self, record: StageRecord) -> bool:
        return self.max_level is None or record.level <= self.max_level

    def _prefix(self, record: StageRecord) -> str:
        return f"[{datetime.datetime.now():%H:%M:%S}] {'  ' * record.level}"

    def stage_started(self, record: StageRecord) -> None:
        if self.show_start and self._visible(record):
            self._write(f"{self._prefix(record)}{LABEL[Status.RUNNING]:<6} {record.stage_name}")

    def stage_finished(self, record: StageRecord) -> None:
        if not self._visible(record) and record.status is not Status.FAILED:
            return
        line = f"{self._prefix(record)}{LABEL[record.status]:<6} {record.stage_name}"
        if record.duration is not None:
            line += f" ({_duration(record.duration)})"
        if record.status in (Status.FAILED, Status.SKIPPED, Status.CANCELLED) and record.reason:
            line += f" - {record.reason}"
        if record.status is Status.FAILED and record.is_root_cause and record.location:
            line += f" [at {record.location}]"
        self._write(line)
        if self.verbose:
            if record.metadata:
                self._write(f"{' ' * 11}{'  ' * record.level}metadata: "
                            f"{json.dumps(record.metadata, default=str)}")
            if record.traceback:
                self._write(record.traceback)

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        self._write("")
        self._write(log._summary_text(details=self.verbose, glyphs=glyphs_for(self.target)))


def write_safely(text: str, stream: TextIO | None = None) -> None:
    """print() that replaces characters the stream's encoding can't show instead of raising."""
    stream = stream if stream is not None else sys.stdout
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding) + "\n")


def glyphs_for(target: Any) -> bool:
    """Whether the target can print box-drawing characters (else the tree uses ASCII)."""
    target = target if target is not None else sys.stderr
    if callable(target) and not hasattr(target, "write"):  # print or a log function
        target = sys.stdout
    encoding = getattr(target, "encoding", None)
    if encoding is None:
        return False
    try:
        "├└│─".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


class _FileOutput(Output):
    """Lazily opened, line-buffered UTF-8 file in append mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None
        self._lock = threading.Lock()

    def _line(self, text: str) -> None:
        with self._lock:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a", encoding="utf-8", newline="\n")
            self._file.write(text + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None


class LogFileOutput(_FileOutput):
    """Developer log: every transition with ids, pid, thread, module, timing, error details and the
    full traceback; ends with the summary and the stage tree."""

    def attach(self, tracker: ExecutionTracker) -> None:
        self._line(f"# ml_pipe execution log  execution_id={tracker.execution_id}  name={tracker.name!r}  "
                   f"pid={os.getpid()}")
        super().attach(tracker)

    def _head(self, record: StageRecord, moment: str | None) -> str:
        return (f"{moment or '-'} | {LABEL[record.status]:<6} | exec={record.execution_id} | "
                f"stage={record.stage_id} | parent={record.parent_stage_id or '-'} | level={record.level} | "
                f"pid={record.pid or '-'} | thread={record.thread or '-'} | module={record.module or '-'}")

    def stage_started(self, record: StageRecord) -> None:
        self._line(f"{self._head(record, record.start_time)} | {record.stage_name} | {record.operation}")

    def stage_finished(self, record: StageRecord) -> None:
        parts = [self._head(record, record.end_time or record.start_time), record.stage_name]
        if record.duration is not None:
            parts.append(f"duration={record.duration:.6f}s")
        if record.error_type:
            parts.append(f"error={record.error_type}: {record.error_message}")
        if record.location:
            parts.append(f"location={record.location}")
        if record.reason:
            parts.append(f"reason={record.reason}")
        if record.metadata:
            parts.append("metadata=" + json.dumps(record.metadata, default=str, sort_keys=True))
        self._line(" | ".join(parts))
        if record.traceback:
            self._line("    " + record.traceback.replace("\n", "\n    "))

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        self._line("")
        self._line(log._summary_text(details=True))
        self._line("")
        self._line(log._tree(glyphs=True))
        if tracker.output_errors:
            self._line("output problems: " + "; ".join(tracker.output_errors))


class JsonlOutput(_FileOutput):
    """Machine-readable event stream: one JSON object per line, written as it happens.

    {"event": "stage_started" | "stage_finished", "time": ..., "record": {...}}
    {"event": "execution_finished", "time": ..., "summary": {...}}
    The last event of a stage_id carries its final state; a file without execution_finished is an
    execution that was interrupted (ExecutionLog.load handles both)."""

    def _event(self, event: str, **payload: Any) -> None:
        payload = {"event": event, "time": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                   **payload}
        self._line(json.dumps(payload, default=str, ensure_ascii=False))

    def stage_started(self, record: StageRecord) -> None:
        self._event("stage_started", record=record.to_dict())

    def stage_finished(self, record: StageRecord) -> None:
        self._event("stage_finished", record=record.to_dict())

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        self._event("execution_finished", summary=log.summary_dict())


class SummaryFiles(Output):
    """At the end: the final records as one JSON document and the summary + tree as text."""

    def __init__(self, json_path: str | Path, text_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self.text_path = Path(text_path)

    def attach(self, tracker: ExecutionTracker) -> None:
        pass

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"summary": log.summary_dict(), "stages": [r.to_dict() for r in log.records]}
        _atomic_write(self.json_path, json.dumps(document, indent=2, default=str, ensure_ascii=False))
        _atomic_write(self.text_path, log._summary_text(details=True) + "\n\n" + log._tree(glyphs=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, path)


class MemoryOutput(Output):
    """Keeps every event in a list (tests, notebooks): [(event, stage_id, status), ...]."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, str | None]] = []

    def attach(self, tracker: ExecutionTracker) -> None:
        super().attach(tracker)

    def stage_started(self, record: StageRecord) -> None:
        self.events.append(("stage_started", record.stage_id, str(record.status)))

    def stage_finished(self, record: StageRecord) -> None:
        self.events.append(("stage_finished", record.stage_id, str(record.status)))

    def execution_finished(self, tracker: ExecutionTracker, log: ExecutionLog) -> None:
        self.events.append(("execution_finished", None, str(tracker.status)))
