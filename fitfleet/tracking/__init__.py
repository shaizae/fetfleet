"""Centralized execution tracking: what ran, where, when, whether it passed, and why it failed.

See docs/execution_tracking.md. The pieces:

    ExecutionTracker   records the stages of one execution and feeds the outputs
    stage / tracked    module-level helpers for library code (no-ops when nothing is tracked)
    ExecutionLog       filter, tree, summary and details - live or loaded from disk
    outputs            console, developer log file, JSON event stream, summary files
"""
from fitfleet.tracking.core import (ExecutionTracker, Stage, StageRecord, Status, annotate, current_stage,
                                   current_tracker, execution, explain, redact_text, stage, tracked)
from fitfleet.tracking.log import ExecutionLog
from fitfleet.tracking.outputs import ConsoleOutput, JsonlOutput, LogFileOutput, MemoryOutput, Output, SummaryFiles
from fitfleet.tracking.tasks import TrackedTask

__all__ = ["ExecutionTracker", "ExecutionLog", "Stage", "StageRecord", "Status", "stage", "tracked", "annotate",
           "execution", "current_tracker", "current_stage", "explain", "redact_text", "Output", "ConsoleOutput",
           "LogFileOutput", "JsonlOutput", "SummaryFiles", "MemoryOutput", "TrackedTask"]
