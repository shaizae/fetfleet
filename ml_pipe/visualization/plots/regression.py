"""Regression plots: metrics, actual vs predicted, residual diagnostics, error analysis."""
from __future__ import annotations

import numpy as np

from ml_pipe.visualization.core import Rendered, Visualization, register
from ml_pipe.visualization.plots._common import fmt, leaderboard, need, predictions, short, subsample
from ml_pipe.visualization.style import INK, INK_SECONDARY, new_figure, reference_line

SECTION = "Regression Analysis"
REGRESSION = ("regression",)
BOTH = ("experiment", "result")
MAX_PANELS = 6
MAX_FLIER_ROWS = 20_000  # past this, outlier markers are an unreadable (and slow) smear


# ---------------------------------------------------------------- metric comparison
def _metric_comparison(ctx, _):
    leaderboard(ctx)
    views = ctx.best_per_model
    names = [("r2", "R² (higher is better)"), ("mae", "MAE (lower is better)"), ("rmse", "RMSE (lower is better)")]
    fig, axes = new_figure(10, 0.42 * len(views) + 1.5, 1, 3, sharey=True)
    for ax, (metric, title) in zip(axes, names):
        values = np.array([v.metrics.get(metric, np.nan) for v in views], dtype=float)
        ax.barh(range(len(views)), values, height=0.6, color=[ctx.color(v.name) for v in views])
        span = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        for y, value in enumerate(values):
            ax.annotate(fmt(value), (value, y), xytext=(3 if value >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=7, color=INK_SECONDARY)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_xlim(min(0, np.nanmin(values) * 1.25), max(0, np.nanmax(values)) + 0.3 * span)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(views)), [short(v.name, 22) for v in views])
    axes[0].invert_yaxis()
    fig.supxlabel("MAE and RMSE are in the target's units; R² is unitless (1 = perfect, 0 = predicting the mean)",
                  fontsize=8, color=INK_SECONDARY)
    metrics = {v.name: ", ".join(f"{m} {fmt(v.metrics.get(m, np.nan))}" for m in ("r2", "mae", "mse", "rmse"))
               for v in views}
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics)


register(Visualization(
    key="reg_metrics", title="Regression metrics by model", section="Overall Results and Model Comparison",
    render=_metric_comparison, tasks=REGRESSION,
    description="R², MAE and RMSE of each model's best configuration, each on its own axis because their scales "
                "differ. RMSE weighs large errors more than MAE: a model with a low MAE but a high RMSE makes "
                "occasional big mistakes. (MSE = RMSE² is listed below.)"))


# ---------------------------------------------------------------- actual vs predicted
def _actual_vs_predicted(ctx, _):
    views = [v for v in ctx.best_per_model if v.y_true is not None and v.y_pred is not None and len(v.y_true)]
    need(views, "no out-of-sample predictions")
    shown = views[:MAX_PANELS]
    cols = min(3, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = new_figure(3.4 * cols + 0.4, 3.4 * rows + 0.3, rows, cols, squeeze=False)
    low = min(min(np.nanmin(v.y_true), np.nanmin(v.y_pred)) for v in shown)
    high = max(max(np.nanmax(v.y_true), np.nanmax(v.y_pred)) for v in shown)
    pad = 0.03 * (high - low or 1)
    limits = (low - pad, high + pad)
    metrics, sampled = {}, False
    for ax, v in zip(axes.flat, shown):
        idx = subsample(len(v.y_true))
        sampled |= idx is not None
        yt, yp = (v.y_true, v.y_pred) if idx is None else (v.y_true[idx], v.y_pred[idx])
        ax.scatter(yt, yp, s=8, alpha=0.45, color=ctx.color(v.name), linewidths=0)
        reference_line(ax, limits, limits)
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect("equal")
        ax.set_title(f"{short(v.name, 24)}\nR² {v.metrics.get('r2', np.nan):.3f} · RMSE {fmt(v.metrics.get('rmse'))}",
                     fontsize=9)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        metrics[v.name] = f"R² {v.metrics.get('r2', np.nan):.4f}, RMSE {fmt(v.metrics.get('rmse'))}"
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    notes = []
    if len(views) > len(shown):
        notes.append(f"Best {len(shown)} of {len(views)} models shown.")
    if sampled:
        notes.append("A random 5,000 points per panel are drawn; metrics use every row.")
    return Rendered(fig, models=tuple(v.name for v in shown), metrics=metrics, note=" ".join(notes))


register(Visualization(
    key="reg_actual_vs_predicted", title="Actual vs predicted", section=SECTION, render=_actual_vs_predicted,
    tasks=REGRESSION, scopes=BOTH,
    description="Each out-of-sample prediction against its true value, one panel per model on identical axes. "
                "Points on the dashed identity line are perfect; a cloud flatter than the line shows the model "
                "shrinks toward the mean, and a widening cloud shows errors that grow with the target."))


# ---------------------------------------------------------------- residual diagnostics (best model)
def _binned_median(x, y, bins: int = 12):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None, None
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centers, medians = [], []
    for b in range(len(edges) - 1):
        mask = which == b
        if mask.sum() >= 3:
            centers.append(np.median(x[mask]))
            medians.append(np.median(y[mask]))
    return np.array(centers), np.array(medians)


def _residuals_vs_predicted(ctx, _):
    view = ctx.best
    predictions(view)
    residuals = view.residuals
    predicted = np.asarray(view.y_pred, dtype=float)
    fig, ax = new_figure(7, 4.2)
    idx = subsample(len(residuals))
    xs, ys = (predicted, residuals) if idx is None else (predicted[idx], residuals[idx])
    ax.scatter(xs, ys, s=8, alpha=0.45, color=ctx.color(view.name), linewidths=0, label="residual")
    ax.axhline(0, color=INK_SECONDARY, linewidth=1)
    centers, medians = _binned_median(predicted, residuals)
    if centers is not None and len(centers) > 1:
        ax.plot(centers, medians, color=INK, linewidth=1.6, label="median residual per bin")
    ax.set_xlabel("predicted value")
    ax.set_ylabel("residual (actual − predicted)")
    ax.set_title(f"Residuals vs predicted - {view.name}")
    ax.legend(loc="best")
    spread_low = np.std(residuals[predicted <= np.median(predicted)])
    spread_high = np.std(residuals[predicted > np.median(predicted)])
    return Rendered(fig, models=(view.name,), metrics={
        "mean residual (bias)": fmt(float(residuals.mean())),
        "residual SD, lower / upper half of predictions": f"{fmt(spread_low)} / {fmt(spread_high)}"},
        note="A random 5,000 points are drawn; statistics use every row." if idx is not None else "")


register(Visualization(
    key="reg_residuals", title="Residuals vs predicted values", section=SECTION, render=_residuals_vs_predicted,
    tasks=REGRESSION, scopes=BOTH,
    description="Residuals of the best model against its predictions, with the median residual per prediction "
                "bin. Ideally a flat, even band around zero. A curved median line means a missed non-linearity; "
                "a funnel shape means the error grows with the prediction (heteroscedasticity)."))


def _residual_distribution(ctx, _):
    view = ctx.best
    predictions(view)
    residuals = view.residuals
    fig, ax = new_figure(6.5, 4)
    bins = min(50, max(10, int(np.sqrt(len(residuals)))))
    ax.hist(residuals, bins=bins, color=ctx.color(view.name), alpha=0.85, edgecolor="white", linewidth=0.5)
    mean, sd = residuals.mean(), residuals.std(ddof=1) if len(residuals) > 1 else 0.0
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.axvline(mean, color=INK, linewidth=1.4, label=f"mean {fmt(mean)}")
    ax.set_xlabel("residual (actual − predicted)")
    ax.set_ylabel("rows")
    ax.set_title(f"Residual distribution - {view.name}")
    ax.legend(loc="upper right")
    skew = float(((residuals - mean) ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
    return Rendered(fig, models=(view.name,), metrics={"mean": fmt(mean), "standard deviation": fmt(sd),
                                                       "skewness": fmt(skew), "rows": len(residuals)})


register(Visualization(
    key="reg_residual_distribution", title="Residual distribution", section=SECTION, render=_residual_distribution,
    tasks=REGRESSION, scopes=BOTH,
    description="Histogram of the best model's residuals. Centred on zero means no overall bias; skew or a long "
                "tail shows the model systematically under- or over-predicts some rows."))


def _qq(ctx, _):
    from scipy import stats  # optional: a missing SciPy skips this plot only
    view = ctx.best
    predictions(view)
    residuals = view.residuals
    need(len(residuals) >= 8, "fewer than 8 residuals")
    (theoretical, ordered), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    fig, ax = new_figure(5, 4.6)
    idx = subsample(len(ordered))
    xs, ys = (theoretical, ordered) if idx is None else (theoretical[idx], ordered[idx])
    ax.scatter(xs, ys, s=8, color=ctx.color(view.name), alpha=0.6, linewidths=0)
    line = np.array([theoretical.min(), theoretical.max()])
    reference_line(ax, line, slope * line + intercept, "normal reference")
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("ordered residuals")
    ax.set_title(f"Normal Q-Q plot of residuals - {view.name}")
    ax.legend(loc="upper left")
    return Rendered(fig, models=(view.name,), metrics={"probability-plot correlation r": f"{r:.4f}"})


register(Visualization(
    key="reg_qq", title="Normal Q-Q plot of residuals", section=SECTION, render=_qq, tasks=REGRESSION, scopes=BOTH,
    description="Residual quantiles against those of a normal distribution. Points on the line = normally "
                "distributed errors; ends bending away = heavier tails than normal, so intervals that assume "
                "normal errors would be too narrow. A diagnostic, not a requirement of the models."))


def _error_by_range(ctx, _):
    view = ctx.best
    predictions(view)
    need(len(view.y_true) >= 20, "fewer than 20 rows")
    actual = np.asarray(view.y_true, dtype=float)
    error = np.abs(view.residuals)
    edges = np.unique(np.quantile(actual, np.linspace(0, 1, 11)))
    need(len(edges) >= 3, "the target takes too few distinct values to bin")
    which = np.clip(np.searchsorted(edges, actual, side="right") - 1, 0, len(edges) - 2)
    groups = [error[which == b] for b in range(len(edges) - 1)]
    labels = [f"{fmt(edges[b])}–{fmt(edges[b + 1])}" for b in range(len(edges) - 1)]
    keep = [i for i, g in enumerate(groups) if len(g)]
    fig, ax = new_figure(8, 4.2)
    ax.boxplot([groups[i] for i in keep], widths=0.55, showfliers=len(error) <= MAX_FLIER_ROWS,
               boxprops={"color": ctx.color(view.name)}, whiskerprops={"color": ctx.color(view.name)},
               capprops={"color": ctx.color(view.name)}, medianprops={"color": INK, "linewidth": 1.2},
               flierprops={"markersize": 2.5, "markeredgecolor": INK_SECONDARY, "alpha": 0.5})
    ax.set_xticks(range(1, len(keep) + 1), [labels[i] for i in keep], rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("actual value range (deciles of the evaluated rows)")
    ax.set_ylabel("absolute error |actual − predicted|")
    ax.set_title(f"Prediction error across the target range - {view.name}")
    ax.grid(axis="x", visible=False)
    medians = [float(np.median(groups[i])) for i in keep]
    worst = int(np.argmax(medians))
    return Rendered(fig, models=(view.name,), metrics={
        "MAE": fmt(view.metrics.get("mae")),
        "worst range (median |error|)": f"{labels[keep[worst]]} ({fmt(medians[worst])})"})


register(Visualization(
    key="reg_error_by_range", title="Error by target range", section=SECTION, render=_error_by_range,
    tasks=REGRESSION, scopes=BOTH,
    description="Absolute errors of the best model grouped by deciles of the true value. Shows whether the "
                "model is reliable across the whole range or fails at the extremes (often where training data "
                "is scarce)."))


def _error_across_models(ctx, _):
    views = [v for v in ctx.best_per_model if v.y_true is not None and v.y_pred is not None and len(v.y_true)]
    need(len(views) >= 2, "needs at least two models")
    groups = [np.abs(v.residuals) for v in views]
    fig, ax = new_figure(7.5, 0.55 * len(views) + 1.5)
    box = ax.boxplot(groups, orientation="horizontal", widths=0.5, patch_artist=True,
                     showfliers=max(len(g) for g in groups) <= MAX_FLIER_ROWS,
                     medianprops={"color": INK, "linewidth": 1.2},
                     flierprops={"markersize": 2.5, "markeredgecolor": INK_SECONDARY, "alpha": 0.4})
    for patch, v in zip(box["boxes"], views):
        patch.set_facecolor("none")
        patch.set_edgecolor(ctx.color(v.name))
    ax.set_yticks(range(1, len(views) + 1), [short(v.name) for v in views])
    ax.invert_yaxis()
    ax.set_xlabel("absolute error |actual − predicted| (target units)")
    ax.set_title("Distribution of absolute errors, best configuration per model")
    ax.grid(axis="y", visible=False)
    metrics = {v.name: f"median {fmt(float(np.median(g)))}, 90th percentile {fmt(float(np.quantile(g, 0.9)))}"
               for v, g in zip(views, groups)}
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics)


register(Visualization(
    key="reg_error_models", title="Error distribution across models", section=SECTION,
    render=_error_across_models, tasks=REGRESSION,
    description="Box plots of every out-of-sample absolute error for each model's best configuration. Two "
                "models with the same MAE can differ in their tails: the 90th percentile and the outliers show "
                "how bad the worst predictions get."))

