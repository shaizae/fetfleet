"""Render reports off the main thread.

One daemon thread runs the submitted jobs one after another (matplotlib figures are
then only ever touched by that thread). Every job runs through safe_call(), so a
failing report is logged and recorded, never raised into the submitting thread.
"""
from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

from ml_pipe.visualization.core import safe_call

logger = logging.getLogger("ml_pipe.visualization")

_STOP = object()


class ReportWorker:
    """submit(fn, ..., what=...) -> Future of (value, error); wait() before the program exits."""

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self._log = log
        self._queue: queue.Queue = queue.Queue()
        self._futures: list[Future] = []
        self._thread = threading.Thread(target=self._loop, name="ml_pipe-report", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                return
            future, func, args, kwargs, what = job
            if not future.set_running_or_notify_cancel():
                continue
            future.set_result(safe_call(func, *args, what=what, log=self._log, **kwargs))

    def submit(self, func: Callable[..., Any], *args, what: str = "report", **kwargs) -> Future:
        """Queue func(*args, **kwargs); the Future resolves to (value, error or None)."""
        future: Future = Future()
        self._futures.append(future)
        self._queue.put((future, func, args, kwargs, what))
        return future

    def wait(self, timeout: float | None = None) -> list[tuple[Any, BaseException | None]]:
        """Wait for every job (up to `timeout` seconds in total), then stop the thread.

        Jobs still running at the timeout are reported as errors and left to finish in the
        background (the thread is a daemon, so it never blocks interpreter exit)."""
        self._queue.put(_STOP)
        self._thread.join(timeout)
        out = []
        for future in self._futures:
            if future.done():
                out.append(future.result())
            else:
                error = TimeoutError(f"report still rendering after {timeout} s")
                logger.warning("%s", error)
                if self._log is not None:
                    self._log(f"warning: {error}; the experiment results are unaffected")
                out.append((None, error))
        return out
