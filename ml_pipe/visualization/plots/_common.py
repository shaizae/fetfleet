"""Helpers shared by the plot modules."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import numpy as np

from ml_pipe.visualization.core import SkipVisualization

if TYPE_CHECKING:
    import pandas as pd

    from ml_pipe.visualization.context import ResultView, VisualizationContext

MAX_SCATTER_POINTS = 5000
SEED = 0  # jitter / subsampling are seeded: the same results give the same figures


def need(condition, reason: str) -> None:
    """Skip the plot with `reason` unless `condition` holds."""
    if not condition:
        raise SkipVisualization(reason)


def predictions(view: ResultView) -> None:
    """Skip unless the view has matching, non-empty y_true / y_pred."""
    need(view.y_true is not None and view.y_pred is not None, f"{view.label}: no out-of-sample predictions")
    need(len(view.y_true) > 0 and len(view.y_true) == len(view.y_pred),
         f"{view.label}: predictions and targets do not match")


def leaderboard(ctx: VisualizationContext) -> pd.DataFrame:
    """The experiment leaderboard, or skip."""
    need(ctx.leaderboard is not None and len(ctx.leaderboard) > 0, "no leaderboard (experiment reports only)")
    return ctx.leaderboard


def score_function(task: str, metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """The metric as computed by the result classes, for re-scoring subsets (e.g. folds)."""
    from sklearn import metrics as m
    if task == "classification":
        weighted = {"average": "weighted", "zero_division": 0}
        functions = {"accuracy": m.accuracy_score,
                     "precision": lambda a, b: m.precision_score(a, b, **weighted),
                     "recall": lambda a, b: m.recall_score(a, b, **weighted),
                     "f1": lambda a, b: m.f1_score(a, b, **weighted)}
    else:
        functions = {"r2": m.r2_score, "mae": m.mean_absolute_error, "mse": m.mean_squared_error,
                     "rmse": lambda a, b: float(np.sqrt(m.mean_squared_error(a, b)))}
    need(metric in functions, f"unknown metric {metric!r}")
    return functions[metric]


def fold_scores(ctx: VisualizationContext, view: ResultView) -> np.ndarray | None:
    """The ranking metric on each fold of a k-fold result (folds are stored in order), or None."""
    sizes = view.fold_sizes
    if sizes is None or view.y_true is None or view.y_pred is None:
        return None
    sizes = np.asarray(sizes, dtype=int)
    if len(sizes) < 2 or sizes.sum() != len(view.y_true) or sizes.min() < 2:
        return None
    score = score_function(ctx.task, ctx.metric)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return np.array([score(view.y_true[a:b], view.y_pred[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])


def parsed_params(board: pd.DataFrame) -> list[dict]:
    """The `params` column of a leaderboard as dicts (it is stored as JSON text)."""
    out = []
    for text in board["params"]:
        try:
            out.append(json.loads(text) if text else {})
        except (TypeError, ValueError):
            out.append({})
    return out


def best_of(values, maximize: bool) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmax(values) if maximize else np.nanmin(values))


def fmt(value: float) -> str:
    """Compact number for captions and labels."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def jitter(n: int, width: float = 0.12) -> np.ndarray:
    return np.random.default_rng(SEED).uniform(-width, width, n)


def subsample(n: int, limit: int = MAX_SCATTER_POINTS) -> np.ndarray | None:
    """Row indexes to draw when n is too large to read as a scatter (None = draw all)."""
    if n <= limit:
        return None
    return np.sort(np.random.default_rng(SEED).choice(n, size=limit, replace=False))


def short(text: str, n: int = 28) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n - 1] + "…"
