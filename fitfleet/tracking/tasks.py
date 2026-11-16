"""Process-pool side of execution tracking.

Worker processes never write log records themselves (several processes appending to one file is
how logs get interleaved or corrupted). Instead every task is wrapped in a TrackedTask, which
carries the execution_id and stage_id of the batch it belongs to and:

- on success returns (result, (pid, start, seconds)); the parent unwraps it with
  ExecutionTracker.tasks_done() and records worker pids and worker time on the batch stage;
- on failure attaches `fitfleet_task` (execution_id, stage_id, pid, the configuration, the raising
  frame) to the exception and re-raises it unchanged. Exception attributes survive pickling, so
  the parent knows exactly which configuration failed, in which process and where.

Only the parent process writes the logs, so records are never mixed between processes.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any, Callable

from fitfleet.tracking.core import location_of, safe_value


class TrackedTask:
    """Picklable wrapper around a pool function (a module-level function, as Pool requires)."""
    __slots__ = ("func", "execution_id", "stage_id")

    def __init__(self, func: Callable[[Any], Any], execution_id: str, stage_id: str | None) -> None:
        self.func = func
        self.execution_id = execution_id
        self.stage_id = stage_id

    def __getstate__(self):
        return self.func, self.execution_id, self.stage_id

    def __setstate__(self, state) -> None:
        self.func, self.execution_id, self.stage_id = state

    def __call__(self, task: Any) -> tuple[Any, tuple[int, float, float]]:
        start, clock = time.time(), time.perf_counter()
        try:
            result = self.func(task)
        except BaseException as exc:
            try:
                exc.fitfleet_task = self._context(task, exc, start, time.perf_counter() - clock)
            except Exception:  # noqa: BLE001 - never replace the task's own exception
                pass
            raise
        return result, (os.getpid(), start, time.perf_counter() - clock)

    def _context(self, task: Any, exc: BaseException, start: float, seconds: float) -> dict[str, Any]:
        model = getattr(task, "model", None)
        try:
            params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
        except Exception:  # noqa: BLE001
            params = {}
        features = getattr(task, "features_index", None)
        return {
            "execution_id": self.execution_id,
            "stage_id": self.stage_id,
            "pid": os.getpid(),
            "module": getattr(self.func, "__module__", ""),
            "operation": getattr(self.func, "__name__", "task"),
            "candidate_id": getattr(task, "candidate_id", None),
            "model": type(model).__name__ if model is not None else type(task).__name__,
            "params": safe_value(params),
            "n_features": None if features is None else len(features),
            "location": location_of(traceback.extract_tb(exc.__traceback__)),
            "start": start,
            "duration": round(seconds, 6),
        }
