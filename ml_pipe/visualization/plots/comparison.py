"""Model comparison, cross-validation, hyper-parameter and cost plots (every task)."""
from __future__ import annotations

import numpy as np

from ml_pipe.visualization.core import Rendered, SkipVisualization, Visualization, register
from ml_pipe.visualization.plots._common import (best_of, fmt, fold_scores, jitter, leaderboard, need,
                                                 parsed_params, short)
from ml_pipe.visualization.style import INK, INK_SECONDARY, model_markers, new_figure

OVERALL = "Overall Results and Model Comparison"


# ---------------------------------------------------------------- ranking
def _ranking(ctx, _):
    board = leaderboard(ctx)
    metric = ctx.metric
    models = ctx.models
    fig, ax = new_figure(7.5, 0.45 * len(models) + 1.4)
    metrics = {}
    for y, model in enumerate(models):
        values = board.loc[board["model"] == model, metric].to_numpy(dtype=float)
        best = best_of(values, ctx.maximize)
        color = ctx.color(model)
        ax.plot([np.nanmin(values), np.nanmax(values)], [y, y], color=color, linewidth=2, alpha=0.35,
                solid_capstyle="round")
        ax.plot([best], [y], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{best:.4f}", (best, y), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7, color=INK_SECONDARY)
        metrics[f"{model} best {metric}"] = fmt(best)
        metrics[f"{model} configurations"] = len(values)
    ax.set_yticks(range(len(models)), [short(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} ({ctx.direction})")
    ax.set_title(f"Best {metric} per model (dot) and range over its configurations (line)")
    ax.grid(axis="y", visible=False)
    return Rendered(fig, models=tuple(models), metrics=metrics)


register(Visualization(
    key="model_ranking", title="Model ranking", section=OVERALL, render=_ranking,
    description="Each model's best configuration (dot) and the range of the ranking metric over all of its "
                "configurations (line), best model on top. A long line means the model is sensitive to its "
                "hyper-parameters / feature subset. The best of many tries is optimistically biased: compare "
                "with the spread, not only the dot."))


# ---------------------------------------------------------------- distribution over configurations
def _config_distribution(ctx, _):
    board = leaderboard(ctx)
    metric, models = ctx.metric, ctx.models
    groups = [board.loc[board["model"] == m, metric].to_numpy(dtype=float) for m in models]
    need(max(len(g) for g in groups) >= 2, "every model has a single configuration: no distribution to show")
    fig, ax = new_figure(7.5, 0.55 * len(models) + 1.5)
    box = ax.boxplot(groups, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 1.2})
    for patch, model in zip(box["boxes"], models):
        patch.set_facecolor("none")
        patch.set_edgecolor(ctx.color(model))
    metrics = {}
    for i, (values, model) in enumerate(zip(groups, models), start=1):
        ax.scatter(values, i + jitter(len(values)), s=14, color=ctx.color(model), alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)
        metrics[f"{model} median {metric}"] = fmt(float(np.nanmedian(values)))
    ax.set_yticks(range(1, len(models) + 1), [f"{short(m)} (n={len(g)})" for m, g in zip(models, groups)])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} ({ctx.direction})")
    ax.set_title(f"{metric} of every configuration, per model")
    ax.grid(axis="y", visible=False)
    return Rendered(fig, models=tuple(models), metrics=metrics)


register(Visualization(
    key="config_distribution", title="Score distribution across configurations", section=OVERALL,
    render=_config_distribution,
    description="Box plot of the ranking metric over every configuration (hyper-parameters x feature subset) of "
                "each model, with every configuration drawn as a point. Shows how robust a model family is to "
                "its settings - a narrow box near the top is a safer choice than a single lucky configuration."))


# ---------------------------------------------------------------- cross-validation folds
def _cv_folds(ctx, _):
    scores = {v.name: fold_scores(ctx, v) for v in ctx.best_per_model}
    scores = {m: s for m, s in scores.items() if s is not None}
    if not scores:
        validation = ctx.best.validation
        if validation == "train_test_split":
            raise SkipVisualization("single train / test split: one score per configuration, no fold distribution")
        if validation == "leave_one_out":
            raise SkipVisualization("leave-one-out: each fold is one row, per-fold scores are not meaningful")
        raise SkipVisualization("no per-fold boundaries recorded for these results")
    models = [m for m in ctx.models if m in scores]
    fig, ax = new_figure(max(4.5, 1.1 * len(models) + 2), 4.2)
    lengths = {len(s) for s in scores.values()}
    sizes = {tuple(v.fold_sizes) for v in ctx.best_per_model if v.name in scores}
    paired = len(models) > 1 and len(lengths) == 1 and len(sizes) == 1
    if paired:  # identical folds: connect fold i across models
        matrix = np.vstack([scores[m] for m in models])
        for fold in range(matrix.shape[1]):
            ax.plot(range(1, len(models) + 1), matrix[:, fold], color="#c9c8c3", linewidth=0.8, zorder=1)
    box = ax.boxplot([scores[m] for m in models], widths=0.45, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 1.2})
    metrics = {}
    for i, (patch, model) in enumerate(zip(box["boxes"], models), start=1):
        patch.set_facecolor("none")
        patch.set_edgecolor(ctx.color(model))
        values = scores[model]
        ax.scatter(np.full(len(values), i), values, s=18, color=ctx.color(model), zorder=3,
                   edgecolors="white", linewidths=0.6)
        metrics[f"{model} {ctx.metric} mean ± sd"] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    ax.set_xticks(range(1, len(models) + 1), [short(m, 18) for m in models], rotation=20 if len(models) > 4 else 0,
                  ha="right" if len(models) > 4 else "center")
    ax.set_ylabel(f"{ctx.metric} per fold ({ctx.direction})")
    ax.set_title(f"{ctx.metric} on each of the {max(lengths)} folds - best configuration per model")
    ax.grid(axis="x", visible=False)
    note = "Grey lines join the same fold across models (every model is scored on identical folds)." if paired else ""
    return Rendered(fig, models=tuple(models), metrics=metrics, note=note)


register(Visualization(
    key="cv_folds", title="Metric distribution across cross-validation folds", section="Cross-Validation Analysis",
    render=_cv_folds,
    description="The ranking metric computed separately on each validation fold for each model's best "
                "configuration. If the gap between two models is small compared with the fold-to-fold spread, "
                "the ranking is not reliable. Folds share training rows, so the spread understates the true "
                "variance and no significance test is implied."))


# ---------------------------------------------------------------- hyper-parameters
def _swept(ctx):
    board = leaderboard(ctx)
    params = parsed_params(board)
    swept = []
    for name, values in ctx.hyper_parameters.items():
        seen = {repr(p[name]) for p in params if name in p}
        if len(seen) > 1:
            swept.append(name)
    need(swept, "no hyper-parameter took more than one value")
    return swept


def _numeric(values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _sensitivity(ctx, name):
    board = leaderboard(ctx)
    params = parsed_params(board)
    metric = ctx.metric
    fig, ax = new_figure(7, 4.2)
    order = list(ctx.hyper_parameters.get(name, []))
    metrics, drawn = {}, []
    for model in ctx.models:
        rows = [(p[name], s) for p, s, m in zip(params, board[metric], board["model"]) if m == model and name in p]
        if len({repr(v) for v, _ in rows}) < 2:
            continue
        values = [v for v in order if any(repr(v) == repr(r) for r, _ in rows)] or sorted({repr(v) for v, _ in rows})
        best, mean = [], []
        for v in values:
            scores = np.array([s for r, s in rows if repr(r) == repr(v)], dtype=float)
            best.append(best_of(scores, ctx.maximize))
            mean.append(float(np.nanmean(scores)))
        numeric = _numeric(values)
        xs = np.asarray(values, dtype=float) if numeric else np.arange(len(values))
        idx = np.argsort(xs) if numeric else np.arange(len(xs))
        color = ctx.color(model)
        ax.plot(xs[idx], np.asarray(best)[idx], "-o", color=color, label=f"{short(model)} (best)", markersize=5)
        ax.plot(xs[idx], np.asarray(mean)[idx], "--", color=color, linewidth=1, alpha=0.7,
                label=f"{short(model)} (mean)")
        if numeric and min(xs) > 0 and max(xs) / min(xs) >= 100:
            ax.set_xscale("log")
        if not numeric:
            ax.set_xticks(range(len(values)), [short(str(v), 14) for v in values])
        top = int(np.argmax(best) if ctx.maximize else np.argmin(best))
        metrics[f"{model} best {name}"] = f"{values[top]} ({metric} {best[top]:.4f})"
        drawn.append(model)
    need(drawn, f"no model tried more than one value of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel(f"{metric} ({ctx.direction})")
    ax.set_title(f"{metric} vs {name}: best and mean over the other settings")
    ax.legend(loc="best", ncols=2 if len(drawn) > 2 else 1)
    return Rendered(fig, models=tuple(drawn), metrics=metrics)


register(Visualization(
    key="hp_sensitivity", title="Hyper-parameter sensitivity", section="Hyperparameter Optimization",
    render=_sensitivity, items=_swept, item_label=str,
    description="A validation curve built from the grid already trained: for each value of one hyper-parameter, "
                "the best (solid) and mean (dashed) ranking metric over every other setting, per model. A flat "
                "curve means the parameter barely matters; a best value at the edge of the grid suggests "
                "extending the grid. Marginal view - interactions are in the heatmap."))


def _heatmap_models(ctx):
    board = leaderboard(ctx)
    params = parsed_params(board)
    out = []
    for model in ctx.models:
        mine = [p for p, m in zip(params, board["model"]) if m == model]
        counts = {k: len({repr(p[k]) for p in mine if k in p}) for k in ctx.hyper_parameters}
        swept = sorted([k for k, c in counts.items() if c > 1], key=lambda k: -counts[k])
        if len(swept) >= 2:
            out.append((model, swept[0], swept[1]))
    need(out, "no model has two hyper-parameters with more than one value")
    return out


def _heatmap(ctx, item):
    from ml_pipe.visualization.style import SEQUENTIAL, annotate_heatmap
    model, row_param, col_param = item
    board = leaderboard(ctx)
    params = parsed_params(board)
    rows = [(p, s) for p, s, m in zip(params, board[ctx.metric], board["model"])
            if m == model and row_param in p and col_param in p]
    order = lambda k: [v for v in ctx.hyper_parameters[k] if any(repr(v) == repr(p[k]) for p, _ in rows)]
    r_values, c_values = order(row_param), order(col_param)
    need(r_values and c_values, "parameter values not found in the leaderboard")
    grid = np.full((len(r_values), len(c_values)), np.nan)
    for i, rv in enumerate(r_values):
        for j, cv in enumerate(c_values):
            scores = [s for p, s in rows if repr(p[row_param]) == repr(rv) and repr(p[col_param]) == repr(cv)]
            if scores:
                grid[i, j] = best_of(scores, ctx.maximize)
    fig, ax = new_figure(max(4.5, 0.75 * len(c_values) + 2.5), max(3.2, 0.5 * len(r_values) + 1.6))
    image = ax.imshow(grid, cmap=SEQUENTIAL, aspect="auto")
    annotate_heatmap(ax, grid, ".3f")
    ax.set_xticks(range(len(c_values)), [short(str(v), 12) for v in c_values])
    ax.set_yticks(range(len(r_values)), [short(str(v), 12) for v in r_values])
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=f"best {ctx.metric} ({ctx.direction})", shrink=0.85)
    ax.set_title(f"{model}: best {ctx.metric} for each {row_param} x {col_param}")
    i, j = np.unravel_index(np.nanargmax(grid) if ctx.maximize else np.nanargmin(grid), grid.shape)
    return Rendered(fig, models=(model,), metrics={
        "best cell": f"{row_param}={r_values[i]}, {col_param}={c_values[j]} ({ctx.metric} {grid[i, j]:.4f})"})


register(Visualization(
    key="hp_heatmap", title="Two-parameter interaction", section="Hyperparameter Optimization",
    render=_heatmap, items=_heatmap_models, item_label=lambda item: item[0],
    description="Best ranking metric for every combination of the model's two most-varied hyper-parameters "
                "(the best over any other settings in each cell). Diagonal ridges reveal interactions that the "
                "one-parameter curves cannot show. Darker = larger value; check the colour-bar direction note."))


# ---------------------------------------------------------------- feature subsets
def _n_features(ctx, _):
    board = leaderboard(ctx)
    need(board["n_features"].nunique() > 1, "every configuration used the same number of features")
    metric = ctx.metric
    fig, ax = new_figure(7, 4.2)
    metrics = {}
    for model in ctx.models:
        mine = board[board["model"] == model]
        if mine.empty:
            continue
        color = ctx.color(model)
        ax.scatter(mine["n_features"], mine[metric], s=10, color=color, alpha=0.25, linewidths=0)
        grouped = mine.groupby("n_features")[metric]
        best = grouped.max() if ctx.maximize else grouped.min()
        ax.plot(best.index, best.values, "-o", color=color, label=short(model), markersize=5)
        k = best.idxmax() if ctx.maximize else best.idxmin()
        metrics[f"{model} best subset size"] = f"{k} features ({metric} {best[k]:.4f})"
    ax.set_xlabel("number of features in the subset")
    ax.set_ylabel(f"{metric} ({ctx.direction})")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_title(f"{metric} vs number of features (line = best subset of each size, dots = every subset)")
    ax.legend(loc="best")
    return Rendered(fig, models=tuple(ctx.models), metrics=metrics)


register(Visualization(
    key="feature_count", title="Feature selection: score vs number of features", section="Feature Analysis",
    render=_n_features,
    description="Performance of the feature subsets tried (brute force, SelectKBest or forward / backward "
                "selection) against their size. Where the best-of-size line flattens, extra features stop "
                "paying off - prefer the smaller subset."))


# ---------------------------------------------------------------- runtime and cost
def _timed(ctx):
    board = leaderboard(ctx)
    times = ctx.fit_times
    need(times is not None and times.notna().any(), "fit time was not recorded for these results")
    frame = board.assign(fit_time=times.reindex(board.index))
    return frame[frame["fit_time"].notna()]


def _runtime(ctx, _):
    frame = _timed(ctx)
    models = [m for m in ctx.models if (frame["model"] == m).any()]
    groups = [frame.loc[frame["model"] == m, "fit_time"].to_numpy(dtype=float) for m in models]
    fig, ax = new_figure(7.5, 0.55 * len(models) + 1.5)
    box = ax.boxplot(groups, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 1.2})
    metrics = {}
    for i, (patch, values, model) in enumerate(zip(box["boxes"], groups, models), start=1):
        patch.set_facecolor("none")
        patch.set_edgecolor(ctx.color(model))
        ax.scatter(values, i + jitter(len(values)), s=14, color=ctx.color(model), alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.5)
        metrics[f"{model} median / total"] = f"{np.median(values):.3f} s / {values.sum():.2f} s"
    everything = np.concatenate(groups)
    if everything.min() > 0 and everything.max() / everything.min() > 50:
        ax.set_xscale("log")
    ax.set_yticks(range(1, len(models) + 1), [short(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("train + predict time per configuration, all folds (seconds, one worker core)")
    ax.set_title("Training cost per configuration, per model")
    ax.grid(axis="y", visible=False)
    return Rendered(fig, models=tuple(models), metrics=metrics)


register(Visualization(
    key="runtime", title="Runtime comparison", section="Runtime and Resource Analysis", render=_runtime,
    description="Wall-clock time of each configuration (fit and predict over every fold), measured inside the "
                "worker process on one core. Includes preprocessing inside the pipeline; excludes data transfer. "
                "Absolute values depend on the machine and its load; compare models within one run."))


def _cost(ctx, _):
    frame = _timed(ctx)
    metric = ctx.metric
    models = [m for m in ctx.models if (frame["model"] == m).any()]
    markers = model_markers(ctx.models)
    fig, ax = new_figure(7, 4.5)
    for model in models:
        mine = frame[frame["model"] == model]
        ax.scatter(mine["fit_time"], mine[metric], s=22, color=ctx.color(model), marker=markers[model],
                   alpha=0.8, edgecolors="white", linewidths=0.5, label=short(model), zorder=3)
    ordered = frame.sort_values(["fit_time", metric], ascending=[True, not ctx.maximize])
    front, best = [], -np.inf
    for t, s in zip(ordered["fit_time"], ordered[metric]):
        signed = s if ctx.maximize else -s
        if signed > best:
            best = signed
            front.append((t, s))
    front = np.array(front)
    ax.step(front[:, 0], front[:, 1], where="post", color=INK, linewidth=1, label="Pareto front", zorder=2)
    if frame["fit_time"].min() > 0 and frame["fit_time"].max() / frame["fit_time"].min() > 50:
        ax.set_xscale("log")
    ax.set_xlabel("train + predict time per configuration (seconds)")
    ax.set_ylabel(f"{metric} ({ctx.direction})")
    ax.set_title(f"{metric} vs computational cost (every configuration)")
    ax.legend(loc="best")
    cheapest = front[0]
    return Rendered(fig, models=tuple(models), metrics={
        "configurations on the Pareto front": len(front),
        "fastest configuration": f"{cheapest[0]:.3f} s ({metric} {cheapest[1]:.4f})",
        "best configuration": f"{front[-1][0]:.3f} s ({metric} {front[-1][1]:.4f})"})


register(Visualization(
    key="cost_vs_score", title="Performance vs computational cost", section="Runtime and Resource Analysis",
    render=_cost,
    description="Every configuration's score against its training time. The Pareto front joins the "
                "configurations that no other one beats on both score and time: a configuration near the top "
                "of the front but much further left is almost as good for a fraction of the cost."))


def _memory(ctx, _):
    raise SkipVisualization("memory consumption is not measured per configuration by the pipeline; "
                            "no values are estimated")


register(Visualization(
    key="memory", title="Memory consumption comparison", section="Runtime and Resource Analysis", render=_memory,
    description="Peak memory per configuration."))
