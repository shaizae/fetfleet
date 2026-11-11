"""Visualization subsystem: isolation, statuses, PDF completeness, and that no plotting or
reporting failure can fail an experiment."""
import base64
import copy
import dataclasses
import gc
import json
import re
import sys
import weakref
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml_pipe import Classification, FilteringCriteriaClassification, Regression
from ml_pipe.config import Experiment
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.results.regression import RegressionResults
from ml_pipe.utils import SharedMemory, ValidationType
from ml_pipe.visualization import (Rendered, SkipVisualization, Visualization, VisualizationContext,
                                   VisualizationManager, VisualizationReport, VisualizationResult, VizStatus, safe_call)
from ml_pipe.visualization import core
from ml_pipe.visualization.background import ReportWorker
from ml_pipe.visualization.pdf import PdfReport
from ml_pipe.visualization.style import new_figure

PNG = b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------------------ helpers
def pdf_images(path) -> int:
    """Number of image XObjects embedded in a PDF written by reportlab."""
    return len(re.findall(rb"/Subtype\s*/Image", open(path, "rb").read()))


def pdf_text(path) -> str:
    """Text drawn in the PDF (decompresses reportlab's Flate page streams)."""
    data = open(path, "rb").read()
    chunks = []
    for match in re.finditer(rb"stream\r?\n(.*?)\s*endstream", data, re.S):
        raw = match.group(1).strip()
        try:  # reportlab: ASCII85 over Flate
            raw = zlib.decompress(base64.a85decode(raw[:-2] if raw.endswith(b"~>") else raw))
        except (ValueError, zlib.error):
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass
        chunks += re.findall(rb"\((.*?)\)\s*Tj", raw)
    return " ".join(c.decode("latin-1") for c in chunks)


def boom(ctx, item):
    raise RuntimeError("plot exploded")


def ok_plot(ctx, item):
    fig, ax = new_figure(3, 2)
    ax.plot([0, 1], [0, 1])
    return Rendered(fig, metrics={"answer": 42})


def spec(key, render, **kw):
    return Visualization(key=key, title=key, section="Overall Results and Model Comparison",
                         description=f"{key} description", render=render, **kw)


def all_failing(monkeypatch):
    """Make every registered plot raise."""
    failing = [dataclasses.replace(v, render=boom) for v in core.registry()]
    monkeypatch.setattr(core, "_REGISTRY", failing)


def snapshot(results):
    return [(r.metrics, None if r.target_test is None else r.target_test.copy(),
             None if r._pred_test is None else r._pred_test.copy(),
             None if getattr(r, "_predict_score", None) is None else r._predict_score.copy()) for r in results]


def assert_same(results, before):
    for r, (metrics, y, pred, proba) in zip(results, before):
        assert r.metrics == metrics
        np.testing.assert_array_equal(r.target_test, y)
        np.testing.assert_array_equal(r._pred_test, pred)
        if proba is not None:
            np.testing.assert_array_equal(r._predict_score, proba)


# ------------------------------------------------------------------ fixtures (trained once per module)
@pytest.fixture(scope="module")
def clf_run():
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target_names[d.target])
    with Classification() as run:
        run.random_state = 0
        run.set(X, y, [DecisionTreeClassifier(random_state=0), LogisticRegression(max_iter=500), LinearSVC()])
        run.features.standard_scaler()
        run.set_hyper_parameters({"max_depth": [1, 3], "C": [0.1, 1.0], "class_weight": [None, "balanced"]})
        results = run.k_folds(3)
        run.filter_by(FilteringCriteriaClassification.accuracy)
        report = run.experiment_report()
        yield run, results, report


@pytest.fixture(scope="module")
def reg_run():
    X, y = make_regression(120, 4, noise=5, random_state=0)
    with Regression() as run:
        run.random_state = 0
        run.set(pd.DataFrame(X), y, [Ridge(), LinearRegression()])
        run.set_hyper_parameters({"alpha": [0.1, 10.0]})
        results = run.k_folds(4)
        yield run, results, run.experiment_report()


@pytest.fixture
def fitted_clf_result():
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X[:400], y[:400])
    r = ClassificationResults(model)
    r.validation = ValidationType.train_test_split
    r.set_test(X[400:], y[400:])
    r.predict()
    return r


# ------------------------------------------------------------------ 1. success
def test_every_applicable_plot_is_generated_for_classification(clf_run):
    _, _, report = clf_run
    viz = report.visualize()
    assert not viz.failed, [(r.key, r.reason) for r in viz.failed]
    keys = {r.key.split(":")[0] for r in viz.succeeded}
    assert {"model_ranking", "config_distribution", "cv_folds", "hp_sensitivity", "clf_metrics", "clf_confusion",
            "clf_per_class", "clf_roc", "clf_pr", "clf_roc_per_class", "runtime", "cost_vs_score"} <= keys
    for r in viz.succeeded:
        assert r.image.startswith(PNG) and r.size_px[0] > 100 and r.size_px[1] > 100
        assert r.description and r.section in core.SECTIONS
    # multiclass: per-class curves present, binary-only calibration skipped with a reason
    assert any(r.key == "clf_calibration" and "binary" in r.reason for r in viz.skipped)
    # memory is never estimated
    assert any(r.key == "memory" and "not measured" in r.reason for r in viz.skipped)


def test_every_applicable_plot_is_generated_for_regression(reg_run):
    _, _, report = reg_run
    viz = report.visualize()
    assert not viz.failed, [(r.key, r.reason) for r in viz.failed]
    keys = {r.key.split(":")[0] for r in viz.succeeded}
    assert {"reg_metrics", "reg_actual_vs_predicted", "reg_residuals", "reg_residual_distribution", "reg_qq",
            "reg_error_by_range", "reg_error_models", "cv_folds", "hp_sensitivity"} <= keys
    assert not any(r.key.startswith("clf_") for r in viz)


def test_plots_use_original_class_labels(clf_run):
    _, _, report = clf_run
    assert report.class_names == ["setosa", "versicolor", "virginica"]
    ctx = VisualizationContext.from_experiment(report)
    assert ctx.best.class_name(0) == "setosa"


def test_worker_records_fit_time_and_fold_sizes(clf_run, reg_run):
    _, results, report = clf_run
    for r in results:
        assert r.fit_time is not None and r.fit_time > 0
        assert r.fold_sizes is not None and len(r.fold_sizes) == 3 and r.fold_sizes.sum() == len(r.target_test)
    assert report.fit_times.notna().all()


def test_train_test_split_has_no_fold_plot_but_has_runtime():
    X, y = make_regression(80, 3, noise=5, random_state=0)
    with Regression() as run:
        run.set(pd.DataFrame(X), y, [Ridge(), LinearRegression()])
        results = run.train_test_split(0.75)
        viz = run.experiment_report().visualize()
    assert all(r.fold_sizes is None and r.fit_time > 0 for r in results)
    cv = next(r for r in viz if r.key == "cv_folds")
    assert cv.status is VizStatus.SKIPPED and "train / test split" in cv.reason
    assert next(r for r in viz if r.key == "runtime").ok


# ------------------------------------------------------------------ 2. failure / 4. one fails, others succeed
def test_failing_plot_is_recorded_and_logged(clf_run, caplog):
    ctx = VisualizationContext.from_experiment(clf_run[2])
    with caplog.at_level("WARNING", logger="ml_pipe.visualization"):
        viz = VisualizationManager([spec("bad", boom)]).run(ctx)
    (r,) = viz.results
    assert r.status is VizStatus.FAILED and r.error_type == "RuntimeError" and "plot exploded" in r.reason
    assert "RuntimeError" in r.traceback and r.image is None
    assert any("bad" in m and "plot exploded" in m for m in caplog.messages)


def test_one_failure_does_not_stop_the_others(clf_run):
    ctx = VisualizationContext.from_experiment(clf_run[2])
    viz = VisualizationManager([spec("a", ok_plot), spec("b", boom), spec("c", ok_plot)]).run(ctx)
    assert [r.status for r in viz] == [VizStatus.SUCCESS, VizStatus.FAILED, VizStatus.SUCCESS]
    assert viz.counts() == {"success": 2, "skipped": 0, "failed": 1}


def test_items_of_one_plot_type_are_isolated_from_each_other(clf_run):
    ctx = VisualizationContext.from_experiment(clf_run[2])

    def render(ctx, item):
        if item == 2:
            raise ValueError("item two is broken")
        return ok_plot(ctx, item)

    viz = VisualizationManager([spec("per", render, items=lambda ctx: [1, 2, 3])]).run(ctx)
    assert [r.key for r in viz] == ["per:1", "per:2", "per:3"]
    assert [r.status for r in viz] == [VizStatus.SUCCESS, VizStatus.FAILED, VizStatus.SUCCESS]


def test_failure_while_listing_items_is_contained(clf_run):
    ctx = VisualizationContext.from_experiment(clf_run[2])
    viz = VisualizationManager([spec("x", ok_plot, items=lambda ctx: 1 / 0), spec("y", ok_plot)]).run(ctx)
    assert [r.status for r in viz] == [VizStatus.FAILED, VizStatus.SUCCESS]


def test_memory_error_is_contained_but_keyboard_interrupt_is_not(clf_run):
    ctx = VisualizationContext.from_experiment(clf_run[2])

    def oom(ctx, item):
        raise MemoryError("too big")

    assert VisualizationManager([spec("m", oom)]).run(ctx).results[0].status is VizStatus.FAILED

    def interrupt(ctx, item):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        VisualizationManager([spec("k", interrupt)]).run(ctx)


def test_broken_registry_still_returns_a_report(clf_run, monkeypatch):
    def broken():
        raise RuntimeError("registry broken")

    manager = VisualizationManager()
    monkeypatch.setattr(type(manager), "visualizations", property(lambda self: broken()))
    viz = manager.run(VisualizationContext.from_experiment(clf_run[2]))
    assert viz.results[0].status is VizStatus.FAILED


# ------------------------------------------------------------------ 3. invalid or missing data
def test_missing_predictions_are_skipped_not_failed():
    r = RegressionResults(LinearRegression())  # never evaluated: no predictions
    viz = r.visualize()
    assert len(viz) > 0 and not viz.failed and not viz.succeeded


def test_degenerate_data_never_fails():
    r = ClassificationResults(DecisionTreeClassifier())
    r.evaluate(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), np.ones((4, 1)))  # one class, odd proba
    viz = r.visualize()
    assert not viz.failed, [(x.key, x.reason) for x in viz.failed]

    r = RegressionResults(LinearRegression())
    r.evaluate(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))  # constant target, 3 rows
    assert not r.visualize().failed


def test_mismatched_lengths_are_skipped():
    r = RegressionResults(LinearRegression())
    r.evaluate(np.arange(10.0), np.arange(10.0))
    r._pred_test = np.arange(5.0)  # corrupted on purpose
    viz = r.visualize()
    assert not viz.succeeded or all(x.status is not VizStatus.FAILED for x in viz)
    assert any("do not match" in x.reason for x in viz.skipped)


def test_nan_predictions_do_not_crash_the_report(tmp_path):
    r = RegressionResults(LinearRegression())
    y = np.linspace(0, 10, 50)
    r.evaluate(y, y + 1)
    r._pred_test = r._pred_test.copy()
    r._pred_test[3] = np.nan
    path = r.save_pdf_report(str(tmp_path))  # some plots may fail on NaN; the report must not
    assert path.exists()


# ------------------------------------------------------------------ 5. missing optional dependencies
def test_missing_scipy_skips_only_the_qq_plot(reg_run, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", None)  # import scipy.stats -> ImportError
    monkeypatch.setitem(sys.modules, "scipy", None)
    viz = reg_run[2].visualize()
    qq = next(r for r in viz if r.key == "reg_qq")
    assert qq.status is VizStatus.SKIPPED and "missing optional dependency" in qq.reason
    assert any(r.key == "reg_actual_vs_predicted" and r.ok for r in viz)


def test_missing_plotting_backend_skips_everything_but_pdf_is_still_written(reg_run, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ml_pipe.visualization.style", None)
    report = reg_run[2]
    path = report.save_pdf_report(tmp_path)
    viz = report.last_visualizations
    assert len(viz) > 0 and not viz.succeeded and not viz.failed
    assert path.exists() and pdf_images(path) == 0
    assert "Skipped visualizations" in pdf_text(path)


# ------------------------------------------------------------------ 6/7/8/12. PDF completeness
def test_pdf_contains_every_generated_plot(clf_run, tmp_path):
    report = clf_run[2]
    path = report.save_pdf_report(tmp_path)
    viz = report.last_visualizations
    assert len(viz.succeeded) >= 15
    assert pdf_images(path) == len(viz.succeeded)
    text = pdf_text(path)
    for heading in ("Executive Summary", "Experiment Configuration", "Dataset Overview",
                    "Overall Results and Model Comparison", "Classification Analysis", "Cross-Validation Analysis",
                    "Hyperparameter Optimization", "Runtime and Resource Analysis", "Visualization Index"):
        assert heading in text, heading
    assert "Regression Analysis" not in text  # irrelevant sections are left out


def test_pdf_with_some_failed_plots(clf_run, tmp_path, monkeypatch):
    extra = [spec("always_fails", boom), spec("also_fine", ok_plot)]
    monkeypatch.setattr(core, "_REGISTRY", core.registry() + extra)
    report = clf_run[2]
    path = report.save_pdf_report(tmp_path)
    viz = report.last_visualizations
    assert [r.key for r in viz.failed] == ["always_fails"]
    assert pdf_images(path) == len(viz.succeeded)
    text = pdf_text(path)
    assert "Failed visualizations" in text and "plot exploded" in text


def test_pdf_when_no_plot_can_be_generated(clf_run, tmp_path, monkeypatch):
    all_failing(monkeypatch)
    report = clf_run[2]
    path = report.save_pdf_report(tmp_path)
    viz = report.last_visualizations
    assert len(viz) > 0 and not viz.succeeded
    assert path.exists() and pdf_images(path) == 0
    text = pdf_text(path)
    assert "Executive Summary" in text and "Failed visualizations" in text


def test_pdf_falls_back_to_text_when_an_image_cannot_be_embedded(tmp_path):
    bad = VisualizationResult(key="corrupt", title="Corrupt", section="Overall Results and Model Comparison",
                              description="d", status=VizStatus.SUCCESS, image=b"not a png", size_px=(300, 200))
    pdf = PdfReport("t")
    path = pdf.build(tmp_path / "r.pdf", VisualizationReport([bad]))
    assert path.exists() and not (tmp_path / "r.pdf.part").exists()
    assert pdf.warnings and "rebuilt without figures" in pdf.warnings[0]
    assert "rebuilt without figures" in pdf_text(path)


def test_per_model_pdf_includes_its_plots(fitted_clf_result, tmp_path):
    r = fitted_clf_result
    path = r.save_pdf_report(str(tmp_path))
    viz = r.last_visualizations
    keys = {x.key.split(":")[0] for x in viz.succeeded}
    assert {"clf_confusion", "clf_roc", "clf_pr", "clf_calibration", "feature_importance"} <= keys
    assert pdf_images(path) == len(viz.succeeded)


def test_per_model_pdf_survives_failing_plots(fitted_clf_result, tmp_path, monkeypatch):
    all_failing(monkeypatch)
    path = fitted_clf_result.save_pdf_report(str(tmp_path))
    assert path.exists() and pdf_images(path) == 0


def test_save_results_keeps_the_model_when_the_pdf_fails(fitted_clf_result, tmp_path, monkeypatch):
    def broken(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ClassificationResults, "save_pdf_report", broken)
    fitted_clf_result.save_results(str(tmp_path))
    (folder,) = tmp_path.iterdir()
    assert any(p.suffix == ".joblib" for p in folder.iterdir())


def test_images_can_be_saved_separately(reg_run, tmp_path):
    viz = reg_run[2].visualize()
    paths = viz.save_images(tmp_path)
    assert len(paths) == len(viz.succeeded) and all(p.read_bytes().startswith(PNG) for p in paths)


# ------------------------------------------------------------------ 9. results are preserved
def test_visualization_does_not_modify_results(clf_run, tmp_path):
    _, results, report = clf_run
    before = snapshot(results)
    board = report.leaderboard.copy()
    summary = report.summary.copy()
    arrays = [(id(r.target_test), id(r._pred_test)) for r in results]
    report.save_pdf_report(tmp_path)
    best = [r for r in results if r.fold_sizes is not None][0]
    best_view = VisualizationContext.from_experiment(report).best
    assert best_view.y_true is report._best_per_model[best_view.name].target_test  # no copy
    assert_same(results, before)
    assert [(id(r.target_test), id(r._pred_test)) for r in results] == arrays
    pd.testing.assert_frame_equal(report.leaderboard, board)
    pd.testing.assert_frame_equal(report.summary, summary)
    assert best.fold_sizes.sum() == len(best.target_test)


# ------------------------------------------------------------------ 10. multiprocessing + visualization failures
def test_experiment_survives_failing_plots_and_pool_stays_usable(monkeypatch, tmp_path):
    all_failing(monkeypatch)
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    with Classification() as run:
        run.random_state = 0
        run.set(X, pd.Series(d.target), [DecisionTreeClassifier(random_state=0), LogisticRegression(max_iter=500)])
        first = run.k_folds(3)
        before = snapshot(first)
        report = run.experiment_report()
        assert report.save_pdf_report(tmp_path).exists()
        assert not report.last_visualizations.succeeded
        assert SharedMemory._shms == {}  # shared memory released as usual
        second = run.k_folds(3)  # the same pool runs again after the failures
        assert [r.metrics for r in second] == [r.metrics for r in first]
        assert_same(first, before)
        best = run.filter_by(FilteringCriteriaClassification.accuracy)
        assert best[0].save_pdf_report(str(tmp_path)).exists()


def _config(tmp_path, report=True):
    d = load_iris()
    frame = pd.DataFrame(d.data, columns=["a", "b", "c", "d"])
    frame["label"] = d.target_names[d.target]
    frame.to_csv(tmp_path / "data.csv", index=False)
    return Experiment.from_dict({
        "data": {"path": "data.csv", "target": "label", "task": "classification"},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0},
        "models": [{"name": "DecisionTreeClassifier", "params": {"random_state": 0}},
                   {"name": "LogisticRegression", "params": {"max_iter": 500}}],
        "hyper_parameters": {"max_depth": [2, 3]},
        "output": {"dir": "runs", "keep": 2, "report": report}}, base=tmp_path)


def test_cli_run_writes_every_report(tmp_path):
    lines = []
    folder = _config(tmp_path).run(log=lines.append)
    assert (folder / "leaderboard.csv").exists()
    reports = list(folder.glob("experiment_report_*.pdf"))
    assert len(reports) == 1 and pdf_images(reports[0]) > 10
    for i in (1, 2):
        (place,) = folder.glob(f"{i}_*")
        assert (place / "model.joblib").exists() and list(place.glob("report_*.pdf"))
    assert sum(line.startswith("report -> ") for line in lines) == 3


def test_cli_run_completes_when_every_plot_and_every_pdf_fails(tmp_path, monkeypatch):
    """The non-negotiable: a visualization or report exception cannot fail the experiment."""
    all_failing(monkeypatch)

    def broken_build(self, *a, **k):
        raise RuntimeError("PDF engine down")

    monkeypatch.setattr(PdfReport, "build", broken_build)
    lines = []
    folder = _config(tmp_path).run(log=lines.append)
    board = pd.read_csv(folder / "leaderboard.csv")
    assert len(board) == 3  # tree x max_depth [2, 3] + logistic regression
    for i in (1, 2):
        (place,) = folder.glob(f"{i}_*")
        assert (place / "model.joblib").exists()
    warnings = [line for line in lines if line.startswith("warning:")]
    assert len(warnings) == 3 and all("PDF engine down" in w for w in warnings)
    assert SharedMemory._shms == {}


def test_cli_run_without_reports_renders_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(VisualizationManager, "run", lambda *a: pytest.fail("rendered with report = false"))
    folder = _config(tmp_path, report=False).run(log=lambda s: None)
    assert not list(folder.rglob("*.pdf"))


# ------------------------------------------------------------------ 11. memory / resource cleanup
def test_figures_are_released_and_pyplot_is_untouched(reg_run, monkeypatch):
    from ml_pipe.visualization import style
    created = []
    original = style.new_figure

    def tracking(*a, **k):
        fig, axes = original(*a, **k)
        created.append(weakref.ref(fig))
        return fig, axes

    for module in ("comparison", "classification", "regression", "features"):
        monkeypatch.setattr(f"ml_pipe.visualization.plots.{module}.new_figure", tracking)
    open_before = plt.get_fignums()
    viz = reg_run[2].visualize()
    assert viz.succeeded and created
    assert plt.get_fignums() == open_before  # nothing registered with pyplot
    del viz
    gc.collect()
    assert all(ref() is None for ref in created), "a figure is still referenced after rendering"


def test_no_temporary_files_are_left(tmp_path, reg_run):
    reg_run[2].save_pdf_report(tmp_path)
    assert [p.suffix for p in tmp_path.iterdir()] == [".pdf"]


def test_images_are_released_after_the_pdf_is_written(tmp_path, reg_run):
    report = reg_run[2]
    report.save_pdf_report(tmp_path)
    kept = report.last_visualizations
    assert kept.succeeded and all(r.image is None for r in kept)  # statuses kept, PNG bytes dropped


# ------------------------------------------------------------------ error handling helpers
def test_safe_call_returns_error_and_logs():
    lines = []
    value, error = safe_call(lambda: 1 / 0, what="thing", log=lines.append)
    assert value is None and isinstance(error, ZeroDivisionError)
    assert lines and "thing failed" in lines[0]
    assert safe_call(lambda x: x + 1, 1) == (2, None)


def test_report_worker_isolates_jobs_and_keeps_order():
    order = []
    worker = ReportWorker()
    worker.submit(lambda: order.append(1) or "a")
    worker.submit(lambda: 1 / 0, what="broken job")
    worker.submit(lambda: order.append(3) or "c")
    outcomes = worker.wait(30)
    assert [v for v, _ in outcomes] == ["a", None, "c"]
    assert isinstance(outcomes[1][1], ZeroDivisionError) and order == [1, 3]


def test_report_worker_timeout_is_reported_not_raised():
    import threading
    release = threading.Event()
    lines = []
    worker = ReportWorker(lines.append)
    worker.submit(release.wait, 30)
    (value, error), = worker.wait(timeout=0.2)
    release.set()
    assert value is None and isinstance(error, TimeoutError) and lines


def test_visualization_section_must_be_known():
    with pytest.raises(ValueError):
        Visualization(key="k", title="t", section="Nowhere", description="", render=ok_plot)


# ------------------------------------------------------------------ feature research charts
def test_feature_research_pdf_survives_a_failing_chart(tmp_path, monkeypatch):
    from ml_pipe import FeatureResearch
    X, y = load_iris(return_X_y=True, as_frame=True)

    def broken(*a, **k):
        raise RuntimeError("chart broke")

    monkeypatch.setattr(FeatureResearch, "plot_importance", broken)
    research = FeatureResearch(X, y, cv=3)
    path = research.save_pdf_report(tmp_path, model=DecisionTreeClassifier(random_state=0), probe=False)
    assert path.exists() and "chart broke" in pdf_text(path)


def test_metrics_in_json_safe_form(clf_run):
    viz = clf_run[2].visualize()
    for r in viz.succeeded:
        json.dumps(r.metrics, default=str)  # printable / serializable
    assert copy.copy(viz).summary() == viz.summary()
