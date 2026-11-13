"""Execution tracking: PASS / FAIL / SKIP / CANCEL detection, hierarchy, outputs, filtering, integration."""
import gc
import json
import os
import threading
import time
import weakref

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from ml_pipe.base import BaseML
from ml_pipe.classification import Classification
from ml_pipe.cli import main
from ml_pipe.config import Experiment
from ml_pipe.tracking import (ConsoleOutput, ExecutionLog, ExecutionTracker, MemoryOutput, Output, Status,
                              annotate, current_tracker, explain, redact_text, stage, tracked)
from ml_pipe.tracking import core


class BrokenClassifier(ClassifierMixin, BaseEstimator):
    """Fails in fit, inside a pool worker (importable by spawned workers as test_tracking.BrokenClassifier)."""

    def __init__(self, level=1):
        self.level = level

    def fit(self, X, y):
        raise ValueError("Input contains invalid feature values.")


def quiet(name="Execution", **kwargs):
    return ExecutionTracker(name, console=False, **kwargs)


def by_name(tracker, name):
    return next(r for r in tracker.records() if r.stage_name == name)


@pytest.fixture
def iris():
    data = load_iris(as_frame=True)
    return data.data, data.target


# ---------------------------------------------------------------- PASS / FAIL detection
def test_successful_stage_is_passed_with_timing():
    with quiet() as t:
        with t.track("Load data", rows=10) as s:
            time.sleep(0.01)
            s.annotate(columns=3)
    record = by_name(t, "Load data")
    assert record.status is Status.PASSED
    assert record.duration >= 0.009  # slept 10 ms (sleep can wake a hair early on some clocks)
    assert record.start <= record.end
    assert record.metadata == {"rows": 10, "columns": 3}
    assert record.execution_id == t.execution_id and record.stage_id.startswith(t.execution_id)
    assert record.parent_stage_id == t.root.stage_id and record.level == 1
    assert record.pid == os.getpid() and record.thread == threading.current_thread().name
    assert record.module == __name__
    assert t.root.status is Status.PASSED and t.finished


def test_failed_stage_records_why_and_where_and_reraises_the_same_exception():
    error = ValueError("Input contains invalid feature values.")
    with pytest.raises(ValueError) as caught:
        with quiet() as t:
            with t.track("Feature Selection"):
                raise error
    assert caught.value is error  # original exception, unchanged
    record = by_name(t, "Feature Selection")
    assert record.status is Status.FAILED
    assert record.error_type == "ValueError"
    assert record.error_message == "Input contains invalid feature values."
    assert "Traceback" in record.traceback and "raise error" in record.traceback
    assert record.location.startswith("test_tracking.py:") and "in test_failed_stage" in record.location
    assert record.reason == "Invalid value or data: Input contains invalid feature values."
    assert record.is_root_cause and t.root.status is Status.FAILED


def test_nested_failure_marks_parents_and_points_at_the_cause():
    with pytest.raises(KeyError):
        with quiet() as t:
            with t.track("Feature Selection"):
                with t.track("Correlation"):
                    raise KeyError("price")
    child, parent = by_name(t, "Correlation"), by_name(t, "Feature Selection")
    assert child.parent_stage_id == parent.stage_id and child.level == 2
    assert child.status is parent.status is Status.FAILED
    assert child.is_root_cause and not parent.is_root_cause
    assert parent.origin == child.stage_id and t.root.origin == child.stage_id
    assert parent.traceback is None  # not repeated: the child has it
    assert "Correlation" in parent.reason and parent.metadata["failed_child"] == child.stage_id
    assert child.reason == "A required column, key or setting was not found: price"


def test_example_hierarchy_with_skipped_stage():
    with pytest.raises(ValueError):
        with quiet() as t:
            t.plan("Data Loading", "Data Preprocessing", "Feature Selection", "Model Training")
            with t.track("Data Loading"):
                pass
            with t.track("Data Preprocessing"):
                pass
            with t.track("Feature Selection"):
                with stage("Correlation"):  # module-level helper attaches to the active tracker
                    raise ValueError("bad")
            with t.track("Model Training"):
                pass
    log = t.log()
    statuses = {r.stage_name: r.status for r in log.records}
    assert statuses == {"Execution": Status.FAILED, "Data Loading": Status.PASSED,
                        "Data Preprocessing": Status.PASSED, "Feature Selection": Status.FAILED,
                        "Correlation": Status.FAILED, "Model Training": Status.SKIPPED}
    skipped = by_name(t, "Model Training")
    assert skipped.reason == "Not run because 'Correlation' failed." and skipped.start_time is None
    tree = log.tree(glyphs=False).splitlines()
    assert tree[0].startswith("Execution") and "FAILED" in tree[0]
    assert tree[4].startswith("|   `-- Correlation") and "bad" in tree[4]
    assert tree[5].startswith("`-- Model Training") and "SKIPPED" in tree[5]


def test_multiple_contained_failures_are_all_reported():
    with quiet() as t:
        for name in ("A", "B"):
            try:
                with t.track(name):
                    raise RuntimeError(f"{name} broke")
            except RuntimeError:
                pass
        with t.track("C"):
            pass
    log = t.log()
    assert t.root.status is Status.PASSED
    assert [r.stage_name for r in log.root_causes()] == ["A", "B"]
    assert log.outcome.startswith("PASSED WITH 2 CONTAINED FAILURE(S)")
    text = log.summary_text()
    assert "1. A" in text and "2. B" in text and "A broke" in text and "B broke" in text


def test_explicit_skips():
    with quiet() as t:
        t.plan("Report")
        t.skip("Report", "reports are turned off")
        with t.track("Optional step") as s:
            s.skip("nothing to do")
    assert by_name(t, "Report").status is Status.SKIPPED
    assert by_name(t, "Report").reason == "reports are turned off"
    assert by_name(t, "Optional step").status is Status.SKIPPED
    assert by_name(t, "Optional step").duration is not None


def test_keyboard_interrupt_is_cancelled_not_failed():
    with pytest.raises(KeyboardInterrupt):
        with quiet() as t:
            t.plan("Train", "Save")
            with t.track("Train"):
                raise KeyboardInterrupt
    assert by_name(t, "Train").status is Status.CANCELLED
    assert by_name(t, "Train").reason == "Cancelled by the user (Ctrl+C)."
    assert by_name(t, "Save").status is Status.SKIPPED
    assert "cancelled" in by_name(t, "Save").reason
    assert t.root.status is Status.CANCELLED and t.log().outcome == "CANCELLED"


def test_stage_left_running_in_another_thread_is_cancelled_at_the_end():
    started, release = threading.Event(), threading.Event()
    with quiet() as t:
        parent = t.root.stage_id

        def job():
            with t.track("Background", parent=parent):
                started.set()
                release.wait(5)

        thread = threading.Thread(target=job)
        thread.start()
        started.wait(5)
    release.set()
    thread.join(5)
    record = by_name(t, "Background")
    assert record.status is Status.CANCELLED and "still running" in record.reason
    assert sum(1 for r in t.records() if r.stage_name == "Background") == 1  # late exit didn't duplicate it


def test_decorators():
    @tracked("Compute", operation="add numbers")
    def compute(a, b):
        return a + b

    assert compute(1, 2) == 3  # no tracker: a plain call
    with quiet() as t:
        assert compute(2, 3) == 5

        @t.stage("Twice")
        def twice(x):
            return 2 * x

        assert twice(4) == 8 and twice(5) == 10
    assert by_name(t, "Compute").operation == "add numbers"
    assert [r.stage_name for r in t.records()].count("Twice") == 2


def test_helpers_are_noops_without_a_tracker():
    assert current_tracker() is None
    with stage("anything") as s:
        s.annotate(x=1)
        s.skip("nothing")
    annotate(y=2)
    with pytest.raises(ValueError):  # exceptions still pass through
        with stage("anything"):
            raise ValueError


# ---------------------------------------------------------------- outputs
def test_output_files_are_complete_and_consistent(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with quiet(output_dir=tmp_path) as t:
            t.plan("Later")
            with t.track("Ok"):
                pass
            with t.track("Broken", password="hunter2"):
                1 / 0
    paths = t.output_paths
    assert all(paths[k].exists() for k in ("log", "jsonl", "json", "summary"))

    events = [json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "execution_finished"
    started = [e["record"]["stage_id"] for e in events if e["event"] == "stage_started"]
    finished = [e["record"]["stage_id"] for e in events if e["event"] == "stage_finished"]
    assert len(started) == len(set(started)) and len(finished) == len(set(finished))  # no duplicates
    assert set(finished) == {r.stage_id for r in t.records()}  # every stage closed exactly once
    assert set(started) == {r.stage_id for r in t.records() if r.start_time}  # skipped stages never started

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["summary"]["counts"] == {"PENDING": 0, "RUNNING": 0, "PASSED": 1, "FAILED": 1,
                                             "SKIPPED": 1, "CANCELLED": 0}
    developer_log = paths["log"].read_text(encoding="utf-8")
    assert "ZeroDivisionError" in developer_log and "Traceback" in developer_log
    assert "hunter2" not in developer_log and "hunter2" not in paths["jsonl"].read_text(encoding="utf-8")
    assert "EXECUTION SUMMARY" in paths["summary"].read_text(encoding="utf-8")


def test_output_dir_added_later_replays_earlier_stages(tmp_path):
    with quiet() as t:
        with t.track("Before"):
            pass
        t.add_output_dir(tmp_path)
        with t.track("After"):
            pass
    names = {r.stage_name for r in ExecutionLog.load(t.output_paths["jsonl"]).records}
    assert {"Before", "After", "Execution"} <= names


def test_interrupted_event_stream_is_still_readable(tmp_path):
    with quiet(output_dir=tmp_path) as t:
        with t.track("Done"):
            pass
    lines = t.output_paths["jsonl"].read_text(encoding="utf-8").splitlines()
    # simulate a crash: the run was killed while a stage was running and a line was half written
    running = json.loads(lines[1])
    cut = tmp_path / "crashed.jsonl"
    cut.write_text("\n".join(lines[:2] + [json.dumps(running)[:40]]), encoding="utf-8")
    log = ExecutionLog.load(cut)
    assert not log.finished and log.outcome.startswith("INCOMPLETE")
    assert log.find("Done").status is Status.RUNNING


def test_failing_output_never_hides_the_application_exception(caplog):
    class Exploding(Output):
        def stage_finished(self, record):
            raise OSError("disk full")

    error = ValueError("the real problem")
    with pytest.raises(ValueError) as caught:
        with quiet(outputs=[Exploding()]) as t:
            with t.track("Step"):
                raise error
    assert caught.value is error
    assert t.output_errors and "disk full" in t.output_errors[0]
    assert len(t.output_errors) == 1  # the broken output is dropped after its first failure
    assert by_name(t, "Step").status is Status.FAILED  # records are still complete
    assert "execution log output failed" in caplog.text


def test_console_user_view_and_developer_view():
    lines = []
    with pytest.raises(ValueError):
        with ExecutionTracker("Run", console=ConsoleOutput(lines.append)):
            with stage("Step", secret_value="x"):
                raise ValueError("bad value")
    text = "\n".join(lines)
    assert "FAIL   Step" in text and "Invalid value or data: bad value" in text and "[at test_tracking.py:" in text
    assert "Traceback" not in text  # users don't get raw stack traces
    assert "EXECUTION SUMMARY" in text

    verbose = []
    with pytest.raises(ValueError):
        with ExecutionTracker("Run", console=ConsoleOutput(verbose.append, verbose=True)):
            with stage("Step"):
                raise ValueError("bad value")
    assert any("Traceback" in line for line in verbose)


def test_console_level_limit_still_shows_deep_failures():
    lines = []
    with pytest.raises(RuntimeError):
        with ExecutionTracker("Run", console=ConsoleOutput(lines.append, max_level=1)):
            with stage("Shallow"):
                with stage("Deep"):
                    raise RuntimeError("deep problem")
    text = "\n".join(lines)
    assert "START    Deep" not in text and "FAIL   Deep" in text


# ---------------------------------------------------------------- privacy
def test_sensitive_values_are_masked_and_data_is_never_logged():
    frame = pd.DataFrame({"a": [123456789]})
    with quiet() as t:
        with t.track("Connect", api_key="sk-123", user_password="pw", data=frame, values=np.arange(3),
                     settings={"token": "abc", "rows": 5}):
            pass
    meta = by_name(t, "Connect").metadata
    assert meta["api_key"] == meta["user_password"] == "***"
    assert meta["data"] == "<DataFrame shape=(1, 1)>" and "123456789" not in json.dumps(meta)
    assert meta["values"] == "<ndarray shape=(3,)>"
    assert meta["settings"] == {"token": "***", "rows": 5}
    assert redact_text("login failed: password=hunter2 for https://bob:pw@host/x") == \
        "login failed: password=*** for https://***@host/x"


def test_error_messages_are_masked():
    with pytest.raises(RuntimeError):
        with quiet() as t:
            with t.track("Connect"):
                raise RuntimeError("could not connect with token=abcdef")
    record = by_name(t, "Connect")
    assert "abcdef" not in record.error_message and "abcdef" not in record.reason


def test_explain_gives_plain_language():
    assert explain(MemoryError()) == "Not enough memory to finish this step."
    assert explain(FileNotFoundError("x.csv")).startswith("A required file or folder was not found: x.csv")


# ---------------------------------------------------------------- filtering and inspection
@pytest.fixture
def sample_log():
    with pytest.raises(TypeError):
        with quiet("Pipeline") as t:
            with t.track("Load", module="ml_pipe.config"):
                pass
            with t.track("Train", module="ml_pipe.base"):
                with t.track("Fold 1", module="ml_pipe.workers"):
                    time.sleep(0.005)  # timestamps have millisecond resolution
                with t.track("Fold 2", module="ml_pipe.workers"):
                    time.sleep(0.005)
                    raise TypeError("wrong type")
    return t.log()


def test_filtering(sample_log):
    log = sample_log
    assert [r.stage_name for r in log.filter(status="pass")] == ["Load", "Fold 1"]
    assert [r.stage_name for r in log.filter(status=["FAILED"], root_causes_only=True)] == ["Fold 2"]
    assert [r.stage_name for r in log.filter(module="workers")] == ["Fold 1", "Fold 2"]
    assert [r.stage_name for r in log.filter(level=2)] == ["Fold 1", "Fold 2"]
    assert [r.stage_name for r in log.filter(level=(None, 1))] == ["Pipeline", "Load", "Train"]
    assert [r.stage_name for r in log.filter(stage_name="fold")] == ["Fold 1", "Fold 2"]
    assert [r.stage_name for r in log.filter(error_type="typeerror", root_causes_only=True)] == ["Fold 2"]
    assert len(log.filter(execution_id=log.execution_ids[0][:4])) == len(log.records)
    assert log.filter(execution_id="zzz") == []
    start = log.find("Fold 1").start
    assert [r.stage_name for r in log.filter(since=start, level=2)] == ["Fold 1", "Fold 2"]
    assert log.filter(until=start.isoformat(), stage_name="Fold 2") == []
    with pytest.raises(ValueError, match="unknown status"):
        log.filter(status="maybe")


def test_expand_details_and_frame(sample_log):
    log = sample_log
    expanded = log.tree("Train", glyphs=False).splitlines()
    assert [line.split()[0] for line in expanded] == ["Train", "|--", "`--"]
    assert len(log.tree(max_depth=1, glyphs=False).splitlines()) == 3  # root + 2 children, folds folded
    details = log.details("Fold 2")
    assert "path         : Pipeline > Train > Fold 2" in details and "Traceback" in details
    assert "location" in details and "test_tracking.py" in details
    frame = log.to_frame()
    assert len(frame) == 5 and {"status", "path", "duration", "stage_id"} <= set(frame.columns)


def test_summary_text_matches_the_records(sample_log):
    text = sample_log.summary_text()
    assert "Total stages : 4" in text  # every stage below the root
    assert "PASSED:    2" in text and "FAILED:    2" in text and "SKIPPED:   0" in text
    assert "FAILED STAGES:\n1. Fold 2" in text
    assert "Where   : Pipeline > Train > Fold 2" in text and "Error   : TypeError" in text
    assert "Also stopped: Pipeline, Train" in text
    assert "Traceback" not in text and "Traceback" in sample_log.summary_text(details=True)


# ---------------------------------------------------------------- concurrency
def test_concurrent_executions_in_threads_do_not_mix():
    trackers, errors = {}, []

    def worker(i):
        try:
            with quiet(f"exec-{i}") as t:
                trackers[i] = t
                for j in range(20):
                    with stage(f"outer-{i}-{j}"):
                        with stage(f"inner-{i}-{j}"):
                            time.sleep(0.0005)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    for i, t in trackers.items():
        records = t.records()
        assert len(records) == 41
        assert {r.execution_id for r in records} == {t.execution_id}
        assert all(str(i) in r.stage_name for r in records)  # only this thread's stages
        for r in records:
            if r.stage_name.startswith("inner"):
                assert t.get(r.parent_stage_id).stage_name == r.stage_name.replace("inner", "outer")
        assert all(r.status is Status.PASSED for r in records)


def test_threads_sharing_one_tracker(tmp_path):
    with quiet(output_dir=tmp_path) as t:
        parent = t.root.stage_id

        def job(i):
            with t.track(f"job-{i}", parent=parent):
                with t.track(f"step-{i}"):  # nests under job-i: the context is per thread
                    pass

        threads = [threading.Thread(target=job, args=(i,)) for i in range(16)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    for i in range(16):
        assert t.get(by_name(t, f"step-{i}").parent_stage_id).stage_name == f"job-{i}"
    lines = t.output_paths["jsonl"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * 33 + 1 and all(json.loads(line) for line in lines)  # no interleaved writes


def test_no_ambiguous_fallback_with_two_active_trackers():
    a, b = quiet("a").start(), quiet("b").start()
    seen = []
    thread = threading.Thread(target=lambda: seen.append(current_tracker()))
    thread.start()
    thread.join()
    b.finish()
    a.finish()
    assert seen == [None]  # a thread without context never guesses between executions


# ---------------------------------------------------------------- multiprocessing
def test_process_pool_run_records_worker_processes(iris, monkeypatch):
    X, y = iris
    monkeypatch.setattr(BaseML, "_process_limit", 2)
    with quiet(task_records=True) as t, Classification() as run:
        run.set(X, y, [LogisticRegression(max_iter=300)])
        run.set_hyper_parameters({"C": [0.1, 1.0, 10.0]})
        run.k_folds(3)
    assert run.execution is t
    batch = next(r for r in t.records() if r.stage_name.startswith("Train batch"))
    assert batch.status is Status.PASSED and batch.metadata["tasks"] == 3
    assert os.getpid() not in batch.metadata["worker_pids"]
    tasks = [r for r in t.records() if r.stage_name.startswith("Task:")]
    assert len(tasks) == 3 and all(r.parent_stage_id == batch.stage_id for r in tasks)
    assert all(r.pid in batch.metadata["worker_pids"] and r.execution_id == t.execution_id for r in tasks)
    names = [r.stage_name for r in t.log().subtree(by_name(t, "k_folds"))]
    assert names[:4] == ["k_folds", "Share data with workers", "Compute folds", "Search (grid search)"]


def test_worker_failure_names_the_configuration_process_and_location(iris, monkeypatch):
    X, y = iris
    monkeypatch.setattr(BaseML, "_process_limit", 2)
    with pytest.raises(ValueError, match="invalid feature values"):
        with quiet() as t, Classification() as run:
            run.set(X, y, [LogisticRegression(max_iter=300), BrokenClassifier()])
            run.k_folds(3)
    (cause,) = t.log().root_causes()
    assert cause.stage_name == "Task: BrokenClassifier #1"
    assert cause.location.startswith("test_tracking.py:") and cause.location.endswith("in fit")
    assert cause.pid != os.getpid()
    task = cause.metadata["task"]
    assert task["execution_id"] == t.execution_id and task["model"] == "BrokenClassifier"
    assert task["params"] == {"level": 1}
    batch = t.get(cause.parent_stage_id)
    assert batch.stage_name.startswith("Train batch") and task["stage_id"] == batch.stage_id
    assert batch.status is Status.FAILED and batch.origin == cause.stage_id
    assert "worker" in cause.traceback.lower() or "RemoteTraceback" in cause.traceback


def test_untracked_run_gets_its_own_quiet_execution(iris, capsys):
    X, y = iris
    with Classification() as run:
        run.set(X, y, [LogisticRegression(max_iter=300)])
        run.train_test_split(0.7)
    assert run.execution.finished and run.execution.status is Status.PASSED
    assert run.execution.root.stage_name == "train_test_split"
    assert run.execution.root.metadata["models"] == ["LogisticRegression"]
    assert "EXECUTION SUMMARY" not in capsys.readouterr().err


def test_validation_error_of_a_run_is_recorded(iris):
    X, y = iris
    with Classification() as run:
        run.set(X, y, [LogisticRegression()])
        with pytest.raises(ValueError, match="n_splits"):
            run.k_folds(1)
    assert run.execution.status is Status.FAILED
    assert run.execution.root.error_type == "ValueError"


# ---------------------------------------------------------------- experiment pipeline
@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = data.data.assign(species=data.target.map(dict(enumerate(data.target_names))))
    df.to_csv(tmp_path / "iris.csv", index=False)
    return tmp_path / "iris.csv"


def experiment(tmp_path, **overrides):
    raw = {"data": {"path": "iris.csv", "target": "species", "task": "classification"},
           "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
           "models": [{"name": "LogisticRegression", "params": {"max_iter": 300}}],
           "output": {"dir": "runs", "keep": 1, "report": False}}
    raw.update(overrides)
    return Experiment.from_dict(raw, base=tmp_path)


def test_experiment_run_is_tracked_end_to_end(tmp_path, iris_csv):
    lines = []
    folder = experiment(tmp_path).run(log=lines.append)
    log = ExecutionLog.load(folder)
    assert log.outcome == "PASSED"
    top = [r.stage_name for r in log.children(log.root)]
    assert top == ["Prepare output folder", "Load data", "Configure run", "Train and validate",
                   "Rank results and write CSV", "Refit best models", "Experiment PDF report", "Save models",
                   "Finish PDF reports"]
    assert log.find("Experiment PDF report").status is Status.SKIPPED
    assert "turned off" in log.find("Experiment PDF report").reason
    assert log.find("Load data").metadata == {"file": "iris.csv", "rows": 150, "features": 4}
    assert any("EXECUTION SUMMARY" in line for line in lines)


def test_experiment_failure_is_explained_and_later_stages_skipped(tmp_path, iris_csv):
    exp = experiment(tmp_path, models=[{"name": "test_tracking.BrokenClassifier"}])
    with pytest.raises(ValueError):
        exp.run(log=lambda *_: None)
    (folder,) = (tmp_path / "runs").iterdir()
    log = ExecutionLog.load(folder)
    assert log.outcome == "FAILED"
    (cause,) = log.root_causes()
    assert cause.stage_name.startswith("Task: BrokenClassifier")
    assert log.path(cause).startswith("Train and validate > k_folds > Search (grid search) > Train batch")
    skipped = {r.stage_name: r.reason for r in log.filter(status="SKIPPED")}
    assert skipped["Save models"] == f"Not run because '{cause.stage_name}' failed."
    assert set(skipped) == {"Rank results and write CSV", "Refit best models", "Experiment PDF report",
                            "Save models", "Finish PDF reports"}


def test_experiment_pdf_has_execution_summary(tmp_path, iris_csv):
    exp = experiment(tmp_path, output={"dir": "runs", "keep": 1, "report": True})
    folder = exp.run(log=lambda *_: None)
    log = ExecutionLog.load(folder)
    assert log.find("Experiment PDF report").status is Status.PASSED
    assert log.find("PDF report: #1 LogisticRegression").status is Status.PASSED
    assert list(folder.glob("experiment_report_*.pdf"))

    from ml_pipe.visualization.pdf import PdfReport
    pdf = PdfReport("t")
    log.add_to_pdf(pdf)
    kinds = [kind for kind, _ in pdf._blocks["Execution Summary"]]
    assert kinds[:2] == ["paragraph", "kv"] and "table" in kinds
    assert pdf.build(tmp_path / "x.pdf").stat().st_size > 1000


def test_failed_plot_becomes_a_failed_sub_stage_without_failing_the_run(tmp_path, iris_csv, monkeypatch):
    from ml_pipe.visualization import core
    from ml_pipe.visualization.core import Visualization

    def broken(ctx, item):
        raise RuntimeError("cannot draw")

    extra = [Visualization("broken_plot", "Broken plot", core.SECTIONS[0], "always fails", broken)]
    monkeypatch.setattr(core, "_REGISTRY", core.registry() + extra)
    exp = experiment(tmp_path, output={"dir": "runs", "keep": 1, "report": True})
    folder = exp.run(log=lambda *_: None)
    log = ExecutionLog.load(folder)
    plot = log.find("Plot: Broken plot")
    assert plot.status is Status.FAILED and "cannot draw" in plot.reason
    assert log.parent(plot).stage_name == "Experiment PDF report"
    assert log.parent(plot).status is Status.PASSED
    assert log.outcome.startswith("PASSED WITH 1 CONTAINED FAILURE")


def test_cli_log_command(tmp_path, iris_csv, capsys):
    folder = experiment(tmp_path).run(log=lambda *_: None)
    capsys.readouterr()
    assert main(["log", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "EXECUTION SUMMARY" in out and "Train and validate" in out
    assert main(["log", str(folder), "--status", "skip"]) == 0
    out = capsys.readouterr().out
    assert "Experiment PDF report" in out and "turned off" in out and out.strip().endswith("stages")
    assert main(["log", str(folder), "--level", "2:", "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert records and all(r["level"] >= 2 for r in records)
    assert main(["log", str(folder), "--expand", "Train and validate", "--depth", "1"]) == 0
    expanded = capsys.readouterr().out.splitlines()
    assert expanded[0].startswith("Train and validate") and "k_folds  (+" in expanded[1] and len(expanded) == 2
    assert main(["log", str(folder), "--details", "Load data"]) == 0
    assert "iris.csv" in capsys.readouterr().out
    assert main(["log", str(folder), "--details", "no such stage"]) == 2
    assert main(["log", str(tmp_path / "nowhere")]) == 2


def test_cli_log_exit_code_for_a_failed_run(tmp_path, iris_csv, capsys):
    exp = experiment(tmp_path, data={"path": "missing.csv", "target": "species", "task": "classification"})
    with pytest.raises(Exception):
        exp.run(log=lambda *_: None)
    (folder,) = (tmp_path / "runs").iterdir()
    assert main(["log", str(folder)]) == 1
    out = capsys.readouterr().out
    assert "FAILED STAGES:\n1. Load data" in out and "data file not found" in out


# ---------------------------------------------------------------- reliability and performance
def test_finished_tracker_is_released():
    with quiet() as t:
        with t.track("x"):
            pass
    assert current_tracker() is None and t not in core._ACTIVE
    ref = weakref.ref(t)
    del t
    gc.collect()
    assert ref() is None


def test_memory_output_sees_every_transition_once():
    memory = MemoryOutput()
    with quiet(outputs=[memory]) as t:
        with t.track("a"):
            pass
    assert memory.events == [("stage_started", t.root.stage_id, "RUNNING"),
                             ("stage_started", by_name(t, "a").stage_id, "RUNNING"),
                             ("stage_finished", by_name(t, "a").stage_id, "PASSED"),
                             ("stage_finished", t.root.stage_id, "PASSED"),
                             ("execution_finished", None, "PASSED")]


def test_overhead_is_small():
    n = 2000

    def plain():
        return 1

    start = time.perf_counter()
    for _ in range(n):
        with stage("off"):  # tracking off: a ContextVar lookup
            plain()
    off = (time.perf_counter() - start) / n
    with quiet() as t:
        start = time.perf_counter()
        for _ in range(n):
            with t.track("on"):
                plain()
        on = (time.perf_counter() - start) / n
    assert off < 20e-6, f"{off * 1e6:.1f} us per untracked stage"
    assert on < 300e-6, f"{on * 1e6:.1f} us per tracked stage"


def test_cost_per_stage_does_not_grow_with_long_executions():
    def batch(t, n=1000):
        start = time.perf_counter()
        for _ in range(n):
            with t.track("step"):
                pass
        return time.perf_counter() - start

    with quiet() as t:
        t.plan("never started")
        first = batch(t)
        for _ in range(6):
            batch(t)
        last = batch(t)
    assert len(t.records()) == 8002
    assert last < 3 * first + 0.05, f"first 1000 stages {first:.3f}s, after 7000 stages {last:.3f}s"
