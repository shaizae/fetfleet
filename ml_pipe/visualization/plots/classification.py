"""Classification plots: metrics, confusion, per-class, ROC, precision-recall, calibration."""
from __future__ import annotations

import numpy as np

from ml_pipe.visualization.core import Rendered, Visualization, register
from ml_pipe.visualization.plots._common import fmt, leaderboard, need, predictions, short
from ml_pipe.visualization.style import (CATEGORICAL, INK_SECONDARY, SEQUENTIAL, annotate_heatmap, new_figure,
                                         reference_line)

SECTION = "Classification Analysis"
CLASSIFICATION = ("classification",)
BOTH = ("experiment", "result")
MAX_CLASSES = 30          # confusion matrix / heatmaps become unreadable past this
MAX_CLASS_CURVES = 8      # one categorical slot per class curve


def _labels(view, codes) -> list[str]:
    return [short(view.class_name(c), 16) for c in codes]


def _with_proba(ctx) -> list:
    views = [v for v in ctx.best_per_model
             if v.proba is not None and v.y_true is not None and v.proba_classes() is not None
             and len(v.proba) == len(v.y_true)]
    kept = {v.name for v in views}
    missing = [v.name for v in ctx.best_per_model if v.name not in kept]
    need(views, "no model produced class probabilities (predict_proba)")
    return views, missing


# ---------------------------------------------------------------- metric comparison
def _metric_comparison(ctx, _):
    leaderboard(ctx)
    names = ["accuracy", "precision", "recall", "f1"]
    views = ctx.best_per_model
    fig, axes = new_figure(10, 0.42 * len(views) + 1.5, 1, 4, sharey=True)
    y_true = ctx.best.y_true
    baseline = None
    if y_true is not None and len(y_true):
        _, counts = np.unique(y_true, return_counts=True)
        baseline = counts.max() / counts.sum()
    metrics = {}
    for ax, metric in zip(axes, names):
        values = [v.metrics.get(metric, np.nan) for v in views]
        ax.barh(range(len(views)), values, height=0.6, color=[ctx.color(v.name) for v in views])
        for y, value in enumerate(values):
            ax.text(min(value, 1) - 0.01, y, f"{value:.3f}", ha="right", va="center", fontsize=7, color="white")
        if metric == "accuracy" and baseline is not None:
            ax.axvline(baseline, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(baseline, len(views) - 0.4, " majority class", fontsize=7, color=INK_SECONDARY, va="bottom")
        ax.set_xlim(0, 1)
        ax.set_title(metric)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(views)), [short(v.name, 22) for v in views])
    axes[0].invert_yaxis()
    fig.supxlabel("score (weighted average over classes for precision / recall / F1; higher is better)",
                  fontsize=8, color=INK_SECONDARY)
    for v in views:
        metrics[v.name] = ", ".join(f"{m} {v.metrics.get(m, np.nan):.4f}" for m in names)
    if baseline is not None:
        metrics["majority-class accuracy (baseline)"] = f"{baseline:.4f}"
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics)


register(Visualization(
    key="clf_metrics", title="Classification metrics by model", section="Overall Results and Model Comparison",
    render=_metric_comparison, tasks=CLASSIFICATION,
    description="Accuracy, precision, recall and F1 of each model's best configuration, one panel per metric "
                "(all on the same 0-1 scale). The dashed line is the accuracy of always predicting the most "
                "frequent class: a model near it has learned little."))


# ---------------------------------------------------------------- class distribution
def _class_distribution(ctx, _):
    view = ctx.best
    need(view.y_true is not None and len(view.y_true), "no evaluated target values")
    codes, counts = np.unique(view.y_true, return_counts=True)
    need(len(codes) <= 60, f"{len(codes)} classes: too many to chart")
    fig, ax = new_figure(max(4.5, 0.45 * len(codes) + 2), 3.6)
    ax.bar(range(len(codes)), counts, width=0.6, color=CATEGORICAL[0])
    total = counts.sum()
    for x, c in enumerate(counts):
        ax.annotate(f"{c} ({c / total:.0%})", (x, c), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(range(len(codes)), _labels(view, codes), rotation=30 if len(codes) > 6 else 0,
                  ha="right" if len(codes) > 6 else "center")
    ax.set_ylabel("rows")
    ax.set_xlabel("class")
    ax.set_title(f"Class distribution of the {total} evaluated rows")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ratio = counts.max() / counts.min()
    return Rendered(fig, metrics={"classes": len(codes), "rows": int(total),
                                  "largest / smallest class": f"{ratio:.2f}"})


register(Visualization(
    key="clf_class_distribution", title="Class distribution", section="Classification Analysis",
    render=_class_distribution, tasks=CLASSIFICATION, scopes=BOTH,
    description="Number of rows per class among the rows the scores were computed on (all rows for k-fold / "
                "leave-one-out, the test rows for a train / test split). Strong imbalance makes accuracy "
                "misleading - read the per-class and precision-recall plots."))


# ---------------------------------------------------------------- confusion matrix
def _confusion(ctx, view):
    from sklearn.metrics import confusion_matrix
    predictions(view)
    codes = view.class_codes()
    need(len(codes) <= MAX_CLASSES, f"{len(codes)} classes: a confusion matrix would be unreadable")
    counts = confusion_matrix(view.y_true, view.y_pred, labels=codes)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = counts / counts.sum(axis=1, keepdims=True)
    size = max(3.4, 0.42 * len(codes) + 2.2)
    fig, axes = new_figure(2 * size + 0.6, size, 1, 2)
    labels = _labels(view, codes)
    for ax, matrix, fmt_, title, vmax in ((axes[0], counts, "d", "Counts", None),
                                          (axes[1], rates, ".2f", "Normalized by true class (row = recall)", 1)):
        ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        if len(codes) <= 15:
            annotate_heatmap(ax, matrix, fmt_ if fmt_ != "d" else ".0f", vmax)
        ax.set_xticks(range(len(codes)), labels, rotation=45 if len(codes) > 4 else 0,
                      ha="right" if len(codes) > 4 else "center")
        ax.set_yticks(range(len(codes)), labels)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(title)
        ax.grid(False)
    diagonal = np.diag(rates)
    worst = int(np.nanargmin(diagonal))
    return Rendered(fig, models=(view.name,), metrics={
        "accuracy": fmt(view.metrics.get("accuracy", np.nan)),
        "lowest recall": f"{labels[worst]} ({diagonal[worst]:.3f})",
        "rows": int(counts.sum())})


register(Visualization(
    key="clf_confusion", title="Confusion matrix", section=SECTION, render=_confusion, tasks=CLASSIFICATION,
    scopes=BOTH, items=lambda ctx: ctx.best_per_model,
    description="Rows are the true classes, columns the predictions. Left: counts. Right: each row divided by "
                "its total, so the diagonal is per-class recall and off-diagonal cells show which classes are "
                "mistaken for which - independent of class size."))


# ---------------------------------------------------------------- per-class metrics
def _per_class(ctx, _):
    from sklearn.metrics import precision_recall_fscore_support
    view = ctx.best
    predictions(view)
    codes = view.class_codes()
    need(len(codes) <= MAX_CLASSES, f"{len(codes)} classes: too many to chart")
    p, r, f, support = precision_recall_fscore_support(view.y_true, view.y_pred, labels=codes, zero_division=0)
    matrix = np.column_stack([p, r, f])
    fig, ax = new_figure(5.2, 0.38 * len(codes) + 1.5)
    ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    annotate_heatmap(ax, matrix, ".3f", 1)
    ax.set_xticks(range(3), ["precision", "recall", "F1"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(codes)), [f"{name} (n={n})" for name, n in zip(_labels(view, codes), support)])
    ax.grid(False)
    ax.set_title(f"Per-class scores - {view.name}", pad=22)
    worst = int(np.argmin(f))
    return Rendered(fig, models=(view.name,), metrics={
        "macro F1": f"{f.mean():.4f}", "weighted F1": fmt(view.metrics.get('f1', np.nan)),
        "worst class (F1)": f"{_labels(view, codes)[worst]} ({f[worst]:.3f})"})


register(Visualization(
    key="clf_per_class", title="Per-class precision, recall and F1", section=SECTION, render=_per_class,
    tasks=CLASSIFICATION, scopes=BOTH,
    description="Precision, recall and F1 of every class for the best model, with each class's support (n). "
                "The weighted averages in the leaderboard can hide a class the model almost never gets right; "
                "scores of small classes are noisy."))


def _per_class_models(ctx, _):
    from sklearn.metrics import f1_score
    views = [v for v in ctx.best_per_model if v.y_true is not None and v.y_pred is not None]
    need(len(views) >= 2, "needs at least two models")
    codes = np.unique(np.concatenate([v.class_codes() for v in views]))
    need(len(codes) <= MAX_CLASSES, f"{len(codes)} classes: too many to chart")
    matrix = np.vstack([f1_score(v.y_true, v.y_pred, labels=codes, average=None, zero_division=0) for v in views])
    fig, ax = new_figure(max(5, 0.6 * len(codes) + 3), 0.4 * len(views) + 1.6)
    image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    if matrix.size <= 300:
        annotate_heatmap(ax, matrix, ".2f", 1)
    ax.set_xticks(range(len(codes)), _labels(views[0], codes), rotation=30 if len(codes) > 6 else 0,
                  ha="right" if len(codes) > 6 else "center")
    ax.set_yticks(range(len(views)), [short(v.name, 24) for v in views])
    ax.set_xlabel("class")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="F1", shrink=0.85)
    ax.set_title("F1 per class and model (best configuration of each model)")
    best = {_labels(views[0], codes)[j]: views[int(np.argmax(matrix[:, j]))].name for j in range(len(codes))}
    return Rendered(fig, models=tuple(v.name for v in views),
                    metrics={f"best on {k}": v for k, v in list(best.items())[:12]})


register(Visualization(
    key="clf_per_class_models", title="Per-class F1 across models", section=SECTION, render=_per_class_models,
    tasks=CLASSIFICATION,
    description="F1 of every class for each model. Shows whether models fail on the same classes (a data "
                "problem) or on different ones (an ensemble may help)."))


# ---------------------------------------------------------------- ROC
def _roc_curve(view):
    """(fpr, tpr, auc, kind) - binary: positive = second class; multiclass: micro-average one-vs-rest."""
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize
    classes = view.proba_classes()
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(view.y_true, view.proba[:, 1], pos_label=classes[1])
        return fpr, tpr, auc(fpr, tpr), f"positive class {view.class_name(classes[1])}"
    y_bin = label_binarize(view.y_true, classes=classes)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), view.proba.ravel())
    return fpr, tpr, auc(fpr, tpr), "micro-average"


def _roc(ctx, _):
    views, missing = _with_proba(ctx)
    fig, ax = new_figure(5.6, 5)
    metrics, kind = {}, ""
    for v in views:
        fpr, tpr, roc_auc, kind = _roc_curve(v)
        ax.plot(fpr, tpr, color=ctx.color(v.name), label=f"{short(v.name)} (AUC {roc_auc:.3f})")
        metrics[f"{v.name} ROC AUC"] = f"{roc_auc:.4f}"
    reference_line(ax, [0, 1], [0, 1], "chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="false positive rate", ylabel="true positive rate")
    ax.set_aspect("equal")
    ax.set_title(f"ROC curves ({kind})" if len(views) == 1 or ctx.scope == "result" else f"ROC curves, {kind}")
    ax.legend(loc="lower right")
    note = f"No predict_proba (not shown): {', '.join(missing)}" if missing else ""
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics, note=note)


register(Visualization(
    key="clf_roc", title="ROC curves", section=SECTION, render=_roc, tasks=CLASSIFICATION, scopes=BOTH,
    description="True vs false positive rate over every decision threshold, for each model's best configuration "
                "(multiclass: micro-average over one-vs-rest classes). AUC is the probability that a random "
                "positive is ranked above a random negative; 0.5 is chance. Can look optimistic when positives "
                "are rare - see the precision-recall curves."))


def _per_class_curves(ctx, kind: str):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
    from sklearn.preprocessing import label_binarize
    view = ctx.best
    need(view.proba is not None and view.proba_classes() is not None and view.y_true is not None,
         f"{view.name}: no class probabilities (predict_proba)")
    classes = view.proba_classes()
    need(len(classes) > 2, "binary target: the single curve is in the model comparison plot")
    y_bin = label_binarize(view.y_true, classes=classes)
    shown = classes[:MAX_CLASS_CURVES]
    fig, ax = new_figure(6, 5)
    metrics = {}
    grid = np.linspace(0, 1, 201)
    interpolated = []
    for i, code in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        if kind == "roc":
            x, y, _ = roc_curve(y_bin[:, i], view.proba[:, i])
            score = auc(x, y)
            interpolated.append(np.interp(grid, x, y))
        else:
            y, x, _ = precision_recall_curve(y_bin[:, i], view.proba[:, i])
            score = average_precision_score(y_bin[:, i], view.proba[:, i])
        name = view.class_name(code)
        metrics[f"class {name} {'AUC' if kind == 'roc' else 'AP'}"] = f"{score:.4f}"
        if code in shown:
            ax.plot(x, y, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=1.5,
                    label=f"{short(name, 16)} ({score:.3f})")
    if kind == "roc":
        x, y, _ = roc_curve(y_bin.ravel(), view.proba.ravel())
        ax.plot(x, y, color="#0b0b0b", linewidth=2, label=f"micro-average ({auc(x, y):.3f})")
        macro = np.mean(interpolated, axis=0)
        macro[0] = 0
        ax.plot(grid, macro, color="#0b0b0b", linewidth=1.2, linestyle=(0, (1, 1.5)),
                label=f"macro-average ({auc(grid, macro):.3f})")
        reference_line(ax, [0, 1], [0, 1], "chance")
        ax.set(xlabel="false positive rate", ylabel="true positive rate")
        metrics["micro-average AUC"] = f"{auc(x, y):.4f}"
        metrics["macro-average AUC"] = f"{auc(grid, macro):.4f}"
    else:
        y, x, _ = precision_recall_curve(y_bin.ravel(), view.proba.ravel())
        micro = average_precision_score(y_bin, view.proba, average="micro")
        ax.plot(x, y, color="#0b0b0b", linewidth=2, label=f"micro-average ({micro:.3f})")
        ax.set(xlabel="recall", ylabel="precision")
        metrics["micro-average AP"] = f"{micro:.4f}"
    ax.set(xlim=(0, 1), ylim=(0, 1.01))
    ax.set_aspect("equal")
    ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=7)
    ax.set_title(f"{'ROC' if kind == 'roc' else 'Precision-recall'} per class (one-vs-rest) - {view.name}")
    note = (f"First {MAX_CLASS_CURVES} of {len(classes)} classes drawn; all are listed below."
            if len(classes) > MAX_CLASS_CURVES else "")
    return Rendered(fig, models=(view.name,), metrics=metrics, note=note)


register(Visualization(
    key="clf_roc_per_class", title="ROC per class", section=SECTION, tasks=CLASSIFICATION, scopes=BOTH,
    render=lambda ctx, _: _per_class_curves(ctx, "roc"),
    description="One-vs-rest ROC curve for each class of the best model, with the micro-average (every "
                "prediction pooled) and the macro-average (classes weighted equally). A class far below the "
                "others is the hard one to separate."))


# ---------------------------------------------------------------- precision-recall
def _pr(ctx, _):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize
    views, missing = _with_proba(ctx)
    fig, ax = new_figure(5.6, 5)
    metrics, prevalence, kind = {}, None, ""
    for v in views:
        classes = v.proba_classes()
        if len(classes) == 2:
            positive = v.y_true == classes[1]
            precision, recall, _ = precision_recall_curve(positive, v.proba[:, 1])
            ap = average_precision_score(positive, v.proba[:, 1])
            prevalence, kind = positive.mean(), f"positive class {v.class_name(classes[1])}"
        else:
            y_bin = label_binarize(v.y_true, classes=classes)
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), v.proba.ravel())
            ap = average_precision_score(y_bin, v.proba, average="micro")
            prevalence, kind = y_bin.mean(), "micro-average"
        ax.plot(recall, precision, color=ctx.color(v.name), label=f"{short(v.name)} (AP {ap:.3f})")
        metrics[f"{v.name} average precision"] = f"{ap:.4f}"
    reference_line(ax, [0, 1], [prevalence, prevalence], f"no skill ({prevalence:.2f})")
    metrics["no-skill precision (prevalence)"] = f"{prevalence:.4f}"
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="recall", ylabel="precision")
    ax.set_aspect("equal")
    ax.set_title(f"Precision-recall curves ({kind})")
    ax.legend(loc="lower left")
    note = f"No predict_proba (not shown): {', '.join(missing)}" if missing else ""
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics, note=note)


register(Visualization(
    key="clf_pr", title="Precision-recall curves", section=SECTION, render=_pr, tasks=CLASSIFICATION, scopes=BOTH,
    description="Precision against recall over every threshold (multiclass: micro-average). Average precision "
                "(AP) summarises the curve; the dashed line is a classifier with no skill (the positive rate). "
                "More informative than ROC when the positive class is rare."))

register(Visualization(
    key="clf_pr_per_class", title="Precision-recall per class", section=SECTION, tasks=CLASSIFICATION, scopes=BOTH,
    render=lambda ctx, _: _per_class_curves(ctx, "pr"),
    description="One-vs-rest precision-recall curve for each class of the best model, with average precision "
                "per class and the micro-average."))


# ---------------------------------------------------------------- calibration
def _calibration(ctx, _):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    views, missing = _with_proba(ctx)
    views = [v for v in views if len(v.proba_classes()) == 2]
    need(views, "calibration diagram is drawn for binary targets only")
    need(min(len(v.y_true) for v in views) >= 20, "fewer than 20 evaluated rows")
    fig, (ax, hist) = new_figure(5.6, 6, 2, 1, height_ratios=[3, 1], sharex=True)
    metrics = {}
    for v in views:
        positive = v.y_true == v.proba_classes()[1]
        p = v.proba[:, 1]
        true_rate, mean_pred = calibration_curve(positive, p, n_bins=10, strategy="uniform")
        brier = brier_score_loss(positive, p)
        color = ctx.color(v.name)
        ax.plot(mean_pred, true_rate, "-o", color=color, markersize=4, label=f"{short(v.name)} (Brier {brier:.3f})")
        hist.hist(p, bins=np.linspace(0, 1, 21), histtype="step", color=color, linewidth=1.5)
        metrics[f"{v.name} Brier score"] = f"{brier:.4f}"
    reference_line(ax, [0, 1], [0, 1], "perfectly calibrated")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), ylabel="observed fraction of positives")
    ax.legend(loc="upper left")
    ax.set_title("Calibration (reliability) diagram")
    hist.set(xlabel="predicted probability of the positive class", ylabel="rows")
    return Rendered(fig, models=tuple(v.name for v in views), metrics=metrics)


register(Visualization(
    key="clf_calibration", title="Calibration", section=SECTION, render=_calibration, tasks=CLASSIFICATION,
    scopes=BOTH,
    description="Whether predicted probabilities can be read as probabilities: in each probability bin, the "
                "observed share of positives against the mean prediction (diagonal = perfect). Below the diagonal "
                "the model is over-confident. The histogram shows how many rows fall in each bin; the Brier score "
                "(lower is better) combines calibration and discrimination."))
