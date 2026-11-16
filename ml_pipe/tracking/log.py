"""Read execution records - live (tracker.log()) or from disk (ExecutionLog.load()).

Users output the log only through a run's three properties (run.execution_log, run.save_pdf_log,
run.save_json_log - see ml_pipe.base.BaseML) and the `ml-pipe log` command. The text and PDF
builders here (_report, _summary_text, _tree, _details, _add_to_pdf, _save_pdf_log,
_save_json_log) are internal to the package.
"""
from __future__ import annotations

import datetime
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_pipe.tracking.core import Status, StageRecord

if TYPE_CHECKING:
    import pandas as pd

_STATUS_ALIASES = {"PASS": Status.PASSED, "OK": Status.PASSED, "FAIL": Status.FAILED, "ERROR": Status.FAILED,
                   "SKIP": Status.SKIPPED, "CANCEL": Status.CANCELLED, "CANCELED": Status.CANCELLED}
RULE = "=" * 60


def _stdout_glyphs() -> bool:
    """Whether sys.stdout can print box-drawing characters (not on a cp1252 Windows console)."""
    import sys
    from ml_pipe.tracking.outputs import glyphs_for
    return glyphs_for(sys.stdout)


def parse_status(value: str | Status) -> Status:
    """'pass', 'FAILED', 'skip', ... -> Status (raises ValueError for anything else)."""
    if isinstance(value, Status):
        return value
    text = str(value).strip().upper()
    if text in _STATUS_ALIASES:
        return _STATUS_ALIASES[text]
    try:
        return Status(text)
    except ValueError:
        raise ValueError(f"unknown status {value!r}; use one of {[s.value for s in Status]}") from None


def _moment(value: str | datetime.datetime | None) -> datetime.datetime | None:
    if value is None:
        return None
    moment = datetime.datetime.fromisoformat(value) if isinstance(value, str) else value
    return moment if moment.tzinfo is not None else moment.astimezone()


def _seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"


class ExecutionLog:
    """An immutable snapshot of one execution's records (or several, when loaded from a folder of runs)."""

    def __init__(self, records: Iterable[StageRecord], *, finished: bool = True) -> None:
        self.records: list[StageRecord] = list(records)
        self.finished = finished
        self._by_id = {r.stage_id: r for r in self.records}
        self._children: dict[str | None, list[StageRecord]] = {}
        for r in self.records:
            self._children.setdefault(r.parent_stage_id, []).append(r)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> ExecutionLog:
        """Read execution_<id>.json (final), execution_<id>.jsonl (event stream; also readable while
        the run is going or after a crash) or a folder holding them (all executions in it)."""
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("execution_*.json")) or sorted(path.glob("execution_*.jsonl"))
            if not files:
                files = sorted(path.rglob("execution_*.json")) or sorted(path.rglob("execution_*.jsonl"))
            if not files:
                raise FileNotFoundError(f"no execution_*.json / .jsonl log in {path}")
            logs = [cls.load(f) for f in files]
            return cls([r for log in logs for r in log.records], finished=all(log.finished for log in logs))
        if not path.exists():
            raise FileNotFoundError(f"execution log not found: {path}")
        if path.suffix == ".jsonl":
            return cls._from_events(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        return cls([StageRecord.from_dict(r) for r in document.get("stages", [])], finished=True)

    @classmethod
    def _from_events(cls, path: Path) -> ExecutionLog:
        latest: dict[str, StageRecord] = {}
        finished = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash: keep everything before it
                if event.get("event") == "execution_finished":
                    finished = True
                elif "record" in event:
                    record = StageRecord.from_dict(event["record"])
                    latest[record.stage_id] = record  # dict keeps first-seen order; last state wins
        return cls(latest.values(), finished=finished)

    # ------------------------------------------------------------ navigation
    @property
    def roots(self) -> list[StageRecord]:
        return self._children.get(None, [])

    @property
    def root(self) -> StageRecord | None:
        return self.roots[0] if self.roots else None

    @property
    def execution_ids(self) -> list[str]:
        return list(dict.fromkeys(r.execution_id for r in self.records))

    def children(self, stage: StageRecord | str) -> list[StageRecord]:
        stage_id = stage.stage_id if isinstance(stage, StageRecord) else stage
        return list(self._children.get(stage_id, []))

    def parent(self, stage: StageRecord) -> StageRecord | None:
        return self._by_id.get(stage.parent_stage_id) if stage.parent_stage_id else None

    def get(self, stage_id: str) -> StageRecord | None:
        return self._by_id.get(stage_id)

    def find(self, key: str) -> StageRecord | None:
        """A stage by id, else by exact name (case-insensitive), else the first whose name contains `key`."""
        if key in self._by_id:
            return self._by_id[key]
        low = key.lower()
        exact = [r for r in self.records if r.stage_name.lower() == low]
        if exact:
            return exact[0]
        return next((r for r in self.records if low in r.stage_name.lower()), None)

    def path(self, stage: StageRecord, include_root: bool = False) -> str:
        """'Train and validate > k_folds > Search (grid)' - where in the execution the stage sits."""
        names = []
        node: StageRecord | None = stage
        while node is not None:
            names.append(node.stage_name)
            node = self.parent(node)
        if not include_root and len(names) > 1:
            names = names[:-1]
        return " > ".join(reversed(names))

    def subtree(self, stage: StageRecord) -> list[StageRecord]:
        out, todo = [], [stage]
        while todo:
            node = todo.pop()
            out.append(node)
            todo.extend(reversed(self.children(node)))
        return out

    # ------------------------------------------------------------ filtering
    def filter(self, *, status: str | Status | Iterable[str | Status] | None = None, execution_id: str | None = None,
               stage_name: str | None = None, module: str | None = None,
               level: int | tuple[int | None, int | None] | None = None,
               since: str | datetime.datetime | None = None, until: str | datetime.datetime | None = None,
               error_type: str | None = None, root_causes_only: bool = False) -> list[StageRecord]:
        """Records matching every given criterion, in execution order.

        status      one status or several ("PASS", "FAILED", "skip", ...)
        execution_id the id or a prefix of it
        stage_name / module   case-insensitive substring
        level       exact level, or (min, max) with None for an open end
        since / until  datetime or ISO string; the stage must start at/after `since` and end
                    (or, if still running, start) at/before `until`
        error_type  exception class name, case-insensitive
        root_causes_only  failed stages where a failure started (not parents that re-raised it)
        """
        if status is not None:
            statuses = {parse_status(status)} if isinstance(status, (str, Status)) else {parse_status(s) for s in status}
        else:
            statuses = None
        start, end = _moment(since), _moment(until)
        name, mod = (stage_name or "").lower(), (module or "").lower()
        out = []
        for r in self.records:
            if statuses is not None and r.status not in statuses:
                continue
            if execution_id and not r.execution_id.startswith(execution_id):
                continue
            if name and name not in r.stage_name.lower():
                continue
            if mod and mod not in (r.module or "").lower():
                continue
            if level is not None:
                low, high = (level, level) if isinstance(level, int) else level
                if (low is not None and r.level < low) or (high is not None and r.level > high):
                    continue
            if start is not None and (r.start is None or r.start < start):
                continue
            if end is not None:
                last = r.end or r.start
                if last is None or last > end:
                    continue
            if error_type and (r.error_type or "").lower() != error_type.lower():
                continue
            if root_causes_only and not r.is_root_cause:
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------ figures
    def stages(self) -> list[StageRecord]:
        """Every stage except the execution roots."""
        return [r for r in self.records if r.parent_stage_id is not None]

    def counts(self) -> dict[str, int]:
        """{status: number of stages} over every stage below the root(s)."""
        counts = {s.value: 0 for s in Status}
        for r in self.stages():
            counts[r.status.value] += 1
        return counts

    def root_causes(self) -> list[StageRecord]:
        return [r for r in self.records if r.is_root_cause]

    @property
    def outcome(self) -> str:
        """PASSED, FAILED, CANCELLED, PASSED WITH CONTAINED FAILURES or INCOMPLETE (log ends mid-run)."""
        roots = self.roots
        if not roots:
            return "EMPTY"
        statuses = {r.status for r in roots}
        if not self.finished or statuses & {Status.RUNNING, Status.PENDING}:
            return "INCOMPLETE (the log ends while stages were running - interrupted or still in progress)"
        if Status.CANCELLED in statuses:
            return "CANCELLED"
        if Status.FAILED in statuses:
            return "FAILED"
        contained = len(self.root_causes())
        if contained:
            return f"PASSED WITH {contained} CONTAINED FAILURE(S) (the run finished; see FAILED STAGES)"
        return "PASSED"

    @property
    def total_duration(self) -> float | None:
        durations = [r.duration for r in self.roots if r.duration is not None]
        if durations:
            return sum(durations)
        starts = [r.start for r in self.records if r.start]
        ends = [r.end for r in self.records if r.end]
        return (max(ends) - min(starts)).total_seconds() if starts and ends else None

    def summary_dict(self) -> dict[str, Any]:
        root = self.root
        return {
            "execution_id": ", ".join(self.execution_ids),
            "name": None if root is None else root.stage_name,
            "outcome": self.outcome.split(" (")[0],
            "status": None if root is None else root.status.value,
            "start_time": None if root is None else root.start_time,
            "end_time": None if root is None else root.end_time,
            "duration": self.total_duration,
            "total_stages": len(self.stages()),
            "counts": self.counts(),
            "failed": [{"stage_id": r.stage_id, "stage": r.stage_name, "path": self.path(r), "module": r.module,
                        "location": r.location, "error_type": r.error_type, "reason": r.reason}
                       for r in self.root_causes()],
            "skipped": [{"stage_id": r.stage_id, "stage": r.stage_name, "path": self.path(r), "reason": r.reason}
                        for r in self.filter(status=Status.SKIPPED)],
        }

    # ------------------------------------------------------------ text
    def _summary_text(self, details: bool = False, glyphs: bool = True) -> str:
        """The execution summary: totals, then every failure (where and why) and every skipped stage.

        details=True is the developer view: stage ids, modules, pids, metadata and tracebacks."""
        root = self.root
        counts = self.counts()
        lines = ["EXECUTION SUMMARY", RULE]
        if root is not None:
            lines.append(f"Execution    : {root.stage_name}")
        lines.append(f"Execution ID : {', '.join(self.execution_ids) or '-'}")
        lines.append(f"Result       : {self.outcome}")
        if root is not None and root.start_time:
            lines.append(f"Started      : {root.start_time}")
            lines.append(f"Finished     : {root.end_time or '-'}")
        lines.append(f"Total stages : {len(self.stages())}")
        lines.append("")
        for status in (Status.PASSED, Status.FAILED, Status.SKIPPED, Status.CANCELLED, Status.RUNNING,
                       Status.PENDING):
            if counts[status.value] or status in (Status.PASSED, Status.FAILED, Status.SKIPPED):
                lines.append(f"{status.value + ':':<11}{counts[status.value]}")
        lines.append("")
        total = self.total_duration
        lines.append(f"Total execution time: {'-' if total is None else f'{total:.2f} seconds'}")

        causes = self.root_causes()
        if causes:
            lines += ["", "FAILED STAGES:"]
            for i, r in enumerate(causes, start=1):
                lines.append(f"{i}. {r.stage_name}")
                lines.append(f"   Where   : {self.path(r, include_root=True)}")
                if r.location:
                    lines.append(f"   Location: {r.location}" + (f"  (module {r.module})" if r.module else ""))
                lines.append(f"   Error   : {r.error_type}")
                lines.append(f"   Reason  : {r.reason or r.error_message}")
                stopped = [s.stage_name for s in self.records if s.status is Status.FAILED and s is not r
                           and s.origin == r.stage_id]
                if stopped:
                    lines.append(f"   Also stopped: {', '.join(stopped)}")
                if details:
                    lines.append(f"   Stage ID: {r.stage_id}  pid {r.pid}  thread {r.thread}  "
                                 f"duration {_seconds(r.duration)}")
                    if r.metadata:
                        lines.append(f"   Metadata: {json.dumps(r.metadata, default=str)}")
                    if r.traceback:
                        lines.append("   Traceback:")
                        lines += ["     " + t for t in r.traceback.splitlines()]
        cancelled = self.filter(status=Status.CANCELLED)
        if cancelled:
            lines += ["", "CANCELLED STAGES:"]
            for i, r in enumerate(cancelled, start=1):
                lines.append(f"{i}. {r.stage_name}")
                lines.append(f"   Where   : {self.path(r, include_root=True)}")
                lines.append(f"   Reason  : {r.reason or '-'}")
        skipped = self.filter(status=Status.SKIPPED)
        if skipped:
            lines += ["", "SKIPPED STAGES:"]
            for i, r in enumerate(skipped, start=1):
                lines.append(f"{i}. {r.stage_name}")
                lines.append(f"   Reason: {r.reason or '-'}")
        lines.append(RULE)
        return "\n".join(lines)

    def _tree(self, stage: StageRecord | str | None = None, *, max_depth: int | None = None,
             glyphs: bool | None = None, show_ids: bool = False, reasons: bool = True) -> str:
        """The stage hierarchy with status and duration per stage.

        stage     expand only this stage (a record, id or name); default every root
        max_depth levels shown below the starting stage (None = all)
        glyphs    box-drawing branches (True) or ASCII (False); None = whatever sys.stdout can print"""
        if glyphs is None:
            glyphs = _stdout_glyphs()
        if isinstance(stage, str):
            found = self.find(stage)
            if found is None:
                raise KeyError(f"no stage {stage!r}")
            stage = found
        starts = [stage] if stage is not None else self.roots
        tee, last, pipe, blank = ("├── ", "└── ", "│   ", "    ") if glyphs else ("|-- ", "`-- ", "|   ", "    ")
        rows: list[tuple[str, StageRecord]] = []

        def walk(node: StageRecord, prefix: str, connector: str, depth: int) -> None:
            rows.append((prefix + connector + node.stage_name + (f" [{node.stage_id}]" if show_ids else ""), node))
            if max_depth is not None and depth >= max_depth:
                hidden = len(self.subtree(node)) - 1
                if hidden:
                    rows[-1] = (rows[-1][0] + f"  (+{hidden} sub-stages)", node)
                return
            kids = self.children(node)
            child_prefix = prefix + ((blank if connector == last else pipe) if connector else "")
            for i, kid in enumerate(kids):
                walk(kid, child_prefix, last if i == len(kids) - 1 else tee, depth + 1)

        for root in starts:
            walk(root, "", "", 0)
        width = min(max((len(label) for label, _ in rows), default=0), 70)
        out = []
        for label, node in rows:
            line = f"{label:<{width}}  {node.status.value:<9} {_seconds(node.duration):>9}"
            if reasons and node.status is not Status.PASSED and node.status is not Status.RUNNING and node.reason:
                if node.status is not Status.FAILED or node.is_root_cause:
                    line += f"  {node.reason[:120]}"
            out.append(line.rstrip())
        return "\n".join(out)

    def _report(self, details: bool = False, *, glyphs: bool | None = None, max_depth: int | None = None,
               show_ids: bool = False) -> str:
        """The whole execution report as one text: the summary (result, counts, failed / cancelled /
        skipped stages with where and why), then the stage tree. print() it, write it to a file, or
        add it to a PDF with _add_to_pdf().

        details=True adds the developer view (stage ids, pids, metadata, tracebacks of the failures).
        glyphs: see _tree(); None = whatever sys.stdout can print."""
        if glyphs is None:
            glyphs = _stdout_glyphs()
        tree = self._tree(max_depth=max_depth, glyphs=glyphs, show_ids=show_ids) or "(no stages)"
        summary = self._summary_text(details=details, glyphs=glyphs)
        return f"{summary}\n\nSTAGE TREE\n{RULE}\n{tree}\n{RULE}"

    def __str__(self) -> str:
        return self._report()

    @classmethod
    def of(cls, execution: Any) -> ExecutionLog | None:
        """An ExecutionLog from an ExecutionTracker (a snapshot), an ExecutionLog (as is), or None / False (None)."""
        if execution is None or execution is False:
            return None
        if isinstance(execution, ExecutionLog):
            return execution
        if hasattr(execution, "log"):
            return execution.log()
        raise TypeError(f"expected an ExecutionTracker or ExecutionLog, got {type(execution).__name__}")

    def _details(self, stage: StageRecord | str) -> str:
        """Everything recorded about one stage (developer view)."""
        record = self.find(stage) if isinstance(stage, str) else stage
        if record is None:
            raise KeyError(f"no stage {stage!r}")
        lines = [f"{record.stage_name}  [{record.status.value}]",
                 f"  path         : {self.path(record, include_root=True)}"]
        for name in StageRecord.FIELDS:
            if name in ("stage_name", "status", "traceback", "metadata"):
                continue
            value = getattr(record, name)
            if value not in (None, ""):
                lines.append(f"  {name:<13}: {value}")
        if record.metadata:
            lines.append(f"  metadata     : {json.dumps(record.metadata, default=str, indent=2)}")
        kids = self.children(record)
        if kids:
            lines.append(f"  children     : " + ", ".join(f"{k.stage_name} ({k.status.value})" for k in kids))
        if record.traceback:
            lines.append("  traceback:")
            lines += ["    " + t for t in record.traceback.splitlines()]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per stage, every field as a column (plus `path`)."""
        import pandas as pd
        rows = [{**r.to_dict(), "path": self.path(r, include_root=True)} for r in self.records]
        return pd.DataFrame(rows, columns=[*StageRecord.FIELDS, "path"])

    # ------------------------------------------------------------ PDF
    def _add_to_pdf(self, pdf, section: str = "Execution Summary", max_rows: int = 200,
                   details: bool = False) -> None:
        """Add the execution report to a PdfReport section: result and totals, failures with location
        and reason, cancelled and skipped stages, then the stage tree (the same content as _report()).

        details=True also adds every failure's traceback; by default those stay in the log files.
        max_rows: stage-tree lines shown (the rest are counted)."""
        summary = self.summary_dict()
        counts = summary["counts"]
        pdf.paragraph(section, "How the execution went, stage by stage, up to the moment this report was "
                               "generated. Full details (tracebacks, metadata) are in the execution_<id>.log "
                               "and .jsonl files of the run folder.")
        root = self.root
        pdf.key_values(section, [
            ("execution", "-" if root is None else root.stage_name),
            ("execution id", summary["execution_id"] or "-"),
            ("result", self.outcome),
            ("started", "-" if root is None or not root.start_time else root.start_time),
            ("finished", "-" if root is None or not root.end_time else root.end_time),
            ("stages", f"{summary['total_stages']}: {counts['PASSED']} passed, {counts['FAILED']} failed, "
                       f"{counts['SKIPPED']} skipped, {counts['CANCELLED']} cancelled"
                       + (f", {counts['RUNNING']} running" if counts["RUNNING"] else "")),
            ("execution time so far" if not self.finished else "execution time",
             "-" if self.total_duration is None else f"{self.total_duration:.2f} s"),
        ])
        causes = self.root_causes()
        if causes:
            pdf.heading(section, "Failed stages")
            items = []
            for r in causes:
                stopped = [s.stage_name for s in self.records if s.status is Status.FAILED and s is not r
                           and s.origin == r.stage_id]
                items.append(f"{self.path(r, include_root=True)} - {r.error_type} at "
                             f"{r.location or 'unknown location'}: {r.reason or r.error_message}"
                             + (f" (also stopped: {', '.join(stopped)})" if stopped else ""))
            pdf.bullets(section, items)
        cancelled = self.filter(status=Status.CANCELLED)
        if cancelled:
            pdf.heading(section, "Cancelled stages")
            pdf.bullets(section, [f"{self.path(r, include_root=True)}: {r.reason or '-'}" for r in cancelled])
        skipped = self.filter(status=Status.SKIPPED)
        if skipped:
            pdf.heading(section, "Skipped stages")
            pdf.bullets(section, [f"{self.path(r, include_root=True)}: {r.reason or '-'}" for r in skipped])
        lines = self._tree(glyphs=False, reasons=False).splitlines()  # PDF fonts have no box-drawing glyphs
        if lines:
            pdf.heading(section, "Stage tree")
            shown = "\n".join(lines[:max_rows])
            if len(lines) > max_rows:
                shown += f"\n... {len(lines) - max_rows} more stages are in the execution log files."
            pdf.preformatted(section, shown)
        if details:
            for r in causes:
                if r.traceback:
                    pdf.heading(section, f"Traceback: {r.stage_name}")
                    pdf.preformatted(section, r.traceback)

    # ------------------------------------------------------------ export
    def _export_path(self, target: str | Path, suffix: str) -> Path:
        """`target` itself if it names a .pdf / .json file, else execution-log_<id>_<time><suffix> in it.
        (Not execution_*: load() of a run folder must not read this copy as a second execution.)"""
        target = Path(target)
        if target.suffix.lower() == suffix:
            path = target
        else:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ident = "_".join(self.execution_ids) or "empty"
            path = target / f"execution-log_{ident}_{stamp}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_pdf_log(self, folder: str | Path, details: bool = True, max_rows: int = 1000) -> Path:
        """Write the execution log as a PDF (result, counts, failed / cancelled / skipped stages, the
        stage tree and, with details=True, the failures' tracebacks) and return its path.
        `folder`: a folder (file named execution-log_<id>_<time>.pdf) or a .pdf path."""
        from ml_pipe.visualization.pdf import PdfReport
        root = self.root
        pdf = PdfReport("Execution Log", "" if root is None else f"{root.stage_name} - {self.outcome.split(' (')[0]}")
        self._add_to_pdf(pdf, max_rows=max_rows, details=details)
        return pdf.build(self._export_path(folder, ".pdf"))

    def _save_json_log(self, folder: str | Path) -> Path:
        """Write the execution log as JSON and return its path: {"summary": ..., "tree": [the stage tree,
        one line per stage], "stages": [every record]}. ExecutionLog.load() reads it back.
        `folder`: a folder (file named execution-log_<id>_<time>.json) or a .json path."""
        path = self._export_path(folder, ".json")
        document = {"summary": self.summary_dict(), "tree": self._tree(glyphs=False).splitlines(),
                    "stages": [r.to_dict() for r in self.records]}
        path.write_text(json.dumps(document, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return path

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"ExecutionLog({', '.join(self.execution_ids)}: {len(self.records)} records, {self.outcome.split(' (')[0]})"
