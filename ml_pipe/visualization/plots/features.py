"""Feature analysis: model-intrinsic importance of refitted models."""
from __future__ import annotations

import numpy as np

from ml_pipe.visualization.core import Rendered, SkipVisualization, Visualization, register
from ml_pipe.visualization.plots._common import need, short
from ml_pipe.visualization.style import new_figure

TOP = 20


def _final(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _feature_names(ctx, view, n: int) -> tuple[list[str], bool]:
    """Names of the columns the final estimator sees, and whether they were standardized."""
    names = list(view.feature_columns or ctx.feature_names or [])
    scaled = False
    model = view.model
    steps = model.steps[:-1] if hasattr(model, "steps") else []
    try:
        for _, step in steps:
            kind = type(step).__name__
            scaled |= kind == "StandardScaler"
            if kind == "SelectColumns":
                if step.columns is not None and names:
                    names = [names[int(i)] for i in step.columns]
            elif hasattr(step, "get_feature_names_out") and names:
                names = [str(c) for c in step.get_feature_names_out(np.asarray(names, dtype=object))]
    except Exception:
        names = []
    if len(names) != n:
        names = [f"x{i}" for i in range(n)]
    return names, scaled


def _fitted_views(ctx):
    views = [v for v in ctx.best_per_model if v.fitted]
    need(views, "no refitted model (results of a run hold unfitted models; refit() / filter_by() / "
                "the CLI's saved models are fitted)")
    return views


def _importance(ctx, view):
    final = _final(view.model)
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
        kind = "impurity-based importance (feature_importances_)"
    elif hasattr(final, "coef_"):
        coef = np.atleast_2d(np.asarray(final.coef_, dtype=float))
        values = np.abs(coef).mean(axis=0)
        kind = "|coefficient|" + (" averaged over classes" if coef.shape[0] > 1 else "")
    else:
        raise SkipVisualization(f"{view.name} exposes neither feature_importances_ nor coef_; "
                                f"use FeatureResearch.permutation_importance()")
    need(values.ndim == 1 and len(values) > 0, "importance vector is empty")
    names, scaled = _feature_names(ctx, view, len(values))
    order = np.argsort(values)[::-1][:TOP]
    fig, ax = new_figure(7, 0.3 * len(order) + 1.5)
    ax.barh(range(len(order)), values[order], height=0.6, color=ctx.color(view.name))
    ax.set_yticks(range(len(order)), [short(names[i], 32) for i in order])
    ax.invert_yaxis()
    ax.set_xlabel(kind)
    ax.set_title(f"Feature importance - {view.name}" + (f" (top {TOP} of {len(values)})" if len(values) > TOP else ""))
    ax.grid(axis="y", visible=False)
    note = ""
    if "coefficient" in kind and not scaled:
        note = "Inputs were not standardized: coefficient sizes depend on each feature's units."
    elif "impurity" in kind:
        note = "Impurity importance favours continuous / high-cardinality features and is computed on training data."
    return Rendered(fig, models=(view.name,), note=note,
                    metrics={short(names[i], 40): f"{values[i]:.4f}" for i in order[:10]})


register(Visualization(
    key="feature_importance", title="Feature importance", section="Feature Analysis", render=_importance,
    scopes=("experiment", "result"), items=_fitted_views, item_label=lambda v: v.name,
    description="The importance the fitted model itself assigns to each input: feature_importances_ for tree "
                "ensembles, the absolute coefficient for linear models. Shows what the model relies on, not "
                "causal effects; correlated features share importance. For a model-agnostic, held-out measure "
                "use FeatureResearch.permutation_importance()."))
