"""Read-only snapshots of collected results, the only input plots get.

A ResultView holds *references* to a result's arrays (no copies); plots must not
modify them. The context is built on the caller's thread before any drawing, so a
report rendered on a background thread does not depend on what the main thread
does to the result objects meanwhile (e.g. refit() replacing `model`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ml_pipe.utils import is_fitted

if TYPE_CHECKING:
    import pandas as pd

    from ml_pipe.results.base import BaseResults
    from ml_pipe.results.experiment import ExperimentReport


@dataclass(frozen=True, eq=False)  # eq=False: identity comparison (fields hold numpy arrays)
class ResultView:
    """What a plot may read about one configuration."""
    name: str                         # model class name
    label: str                        # how plots name it
    rank: int | None
    y_true: np.ndarray | None
    y_pred: np.ndarray | None
    proba: np.ndarray | None
    fold_sizes: np.ndarray | None
    fit_time: float | None
    metrics: dict[str, float]
    params: dict[str, Any]
    validation: str
    model: Any                        # the estimator (fitted only after refit)
    fitted: bool
    features_index: tuple[int, ...] | None
    feature_columns: tuple[str, ...] | None
    class_names: tuple[str, ...] | None

    @classmethod
    def of(cls, result: BaseResults, rank: int | None = None, label: str | None = None,
           class_names: Sequence | None = None) -> ResultView:
        """Snapshot `result` (array references, not copies)."""
        names = class_names if class_names is not None else getattr(result, "class_names", None)
        index = result.features_index
        columns = getattr(result, "feature_columns", None)
        model = result.model
        try:
            fitted = model is not None and is_fitted(model) and hasattr(model, "predict")
        except Exception:
            fitted = False
        try:
            params = dict(result.params)
        except Exception:
            params = {}
        fit_time = getattr(result, "fit_time", None)
        fold_sizes = getattr(result, "fold_sizes", None)
        return cls(
            name=result.name, label=label or result.name, rank=rank,
            y_true=_array(result.target_test), y_pred=_array(getattr(result, "_pred_test", None)),
            proba=_array(getattr(result, "_predict_score", None)),
            fold_sizes=None if fold_sizes is None else np.asarray(fold_sizes),
            fit_time=None if fit_time is None else float(fit_time),
            metrics={k: float(v) for k, v in result.metrics.items()}, params=params,
            validation=str(result.validation), model=model, fitted=fitted,
            features_index=None if index is None else tuple(int(i) for i in index),
            feature_columns=None if columns is None else tuple(str(c) for c in columns),
            class_names=None if names is None else tuple(str(c) for c in names))

    @property
    def residuals(self) -> np.ndarray:
        """y_true - y_pred (a new array)."""
        return np.asarray(self.y_true, dtype=float) - np.asarray(self.y_pred, dtype=float)

    def class_codes(self) -> np.ndarray:
        """Every class that appears in y_true or y_pred, sorted."""
        return np.unique(np.concatenate([np.asarray(self.y_true), np.asarray(self.y_pred)]))

    def proba_classes(self) -> np.ndarray | None:
        """The class each predict_proba column belongs to, or None if that can't be told."""
        if self.proba is None or self.proba.ndim != 2 or self.y_true is None:
            return None
        k = self.proba.shape[1]
        classes = getattr(self.model, "classes_", None)
        if classes is not None and len(classes) == k:
            return np.asarray(classes)
        seen = np.unique(self.y_true)
        if len(seen) == k:
            return seen
        if np.issubdtype(seen.dtype, np.integer) and seen.min() >= 0 and seen.max() < k:
            return np.arange(k)  # label-encoded targets: column i = class i
        return None

    def class_name(self, code) -> str:
        """Original label of an encoded class when known, else the value itself."""
        if self.class_names is not None and isinstance(code, (int, np.integer)) and 0 <= code < len(
                self.class_names):
            return self.class_names[int(code)]
        return str(code)


def _colors(models: list[str]) -> dict[str, str]:
    """Model -> color; empty (plots fall back to gray) if the plotting stack is unavailable."""
    try:
        from ml_pipe.visualization.style import model_colors
        return model_colors(models)
    except ImportError:
        return {}


def _array(value) -> np.ndarray | None:
    return None if value is None else np.asarray(value)


@dataclass(frozen=True)
class VisualizationContext:
    """Everything one report's plots can see.

    scope        "experiment" (every configuration of a run) or "result" (one model)
    best         the overall best configuration; best_per_model one view per model, best first
    leaderboard  the experiment's ranked DataFrame (experiment scope only; read-only)
    """
    scope: str
    task: str
    metric: str
    maximize: bool
    best: ResultView
    best_per_model: tuple[ResultView, ...]
    leaderboard: pd.DataFrame | None = None
    fit_times: pd.Series | None = None
    hyper_parameters: dict[str, list] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    preprocessing: tuple[str, ...] = ()
    n_rows: int | None = None
    n_columns: int | None = None
    search_log: tuple[dict, ...] = ()
    colors: dict[str, str] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Model names, best first."""
        return [v.name for v in self.best_per_model]

    @property
    def metric_names(self) -> list[str]:
        return list(self.best.metrics)

    @property
    def direction(self) -> str:
        return "higher is better" if self.maximize else "lower is better"

    def color(self, model: str) -> str:
        return self.colors.get(model, "#8c8b86")

    @classmethod
    def from_experiment(cls, report: ExperimentReport) -> VisualizationContext:
        """Snapshot an ExperimentReport (its best configuration per model and its leaderboard)."""
        board = report.leaderboard
        class_names = getattr(report, "class_names", None)
        views = []
        for name, result in report._best_per_model.items():
            rows = board.index[board["model"] == name]
            views.append(ResultView.of(result, rank=int(rows[0]) if len(rows) else None, label=name,
                                       class_names=class_names))
        views.sort(key=lambda v: (v.rank is None, v.rank))
        return cls(scope="experiment", task=report.task, metric=report.criteria.value,
                   maximize=bool(report.criteria.maximize), best=views[0], best_per_model=tuple(views),
                   leaderboard=board, fit_times=getattr(report, "fit_times", None),
                   hyper_parameters=dict(report.hyper_parameters), feature_names=tuple(report.feature_names),
                   preprocessing=tuple(report.preprocessing), n_rows=report.n_rows, n_columns=report.n_columns,
                   search_log=tuple(getattr(report, "search_log", ()) or ()),
                   colors=_colors([v.name for v in views]))

    @classmethod
    def from_result(cls, result: BaseResults) -> VisualizationContext:
        """Snapshot one result, for its own PDF report."""
        task = "classification" if getattr(result, "is_classification", False) else "regression"
        view = ResultView.of(result)
        metric = "accuracy" if task == "classification" else "r2"
        return cls(scope="result", task=task, metric=metric, maximize=True, best=view, best_per_model=(view,),
                   feature_names=view.feature_columns or (), colors=_colors([view.name]))
