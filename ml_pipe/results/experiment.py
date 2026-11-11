"""Summary of every configuration an experiment (classification or regression run) trained."""
from __future__ import annotations

import datetime
import json
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from ml_pipe.search import rank

if TYPE_CHECKING:
    from ml_pipe.features import Features
    from ml_pipe.results.base import BaseResults
    from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression


def leaderboard(ranked: list[BaseResults], grid: dict[str, list] | None, features: Features) -> pd.DataFrame:
    """One row per configuration, in the given order; the index is the rank (1 = best).
    grid: only these hyper-parameters are listed (None = every parameter of the model).
    Features are named after the columns only while the matrix still has one column per name
    (no polynomial expansion); otherwise they are listed as x<index>."""
    names = list(features.columns) if features.columns is not None else None
    n_columns = features.data.shape[1]
    same_columns = names is not None and n_columns == len(names)
    rows = []
    for r in ranked:
        index = r.features_index
        if index is None:
            used = "all"
        elif same_columns:
            used = ", ".join(str(names[i]) for i in index)
        else:
            used = ", ".join(f"x{i}" for i in index)
        params = r.params if grid is None else {k: v for k, v in r.params.items() if k in grid}
        rows.append({"model": r.name, "params": json.dumps(params, default=str) if params else "",
                     "n_features": len(index) if index is not None else n_columns,
                     "features": used, "validation": str(r.validation), **r.metrics})
    board = pd.DataFrame(rows)
    board.index = range(1, len(board) + 1)
    return board


class ExperimentReport:
    """Every configuration of one run, ranked by `criteria`, plus a per-model summary.

    leaderboard  one row per configuration, best first (index = rank)
    summary      one row per model: configurations tried, best / mean / std / worst of
                 `criteria`, the mean of every other metric, and the best one's details
    best         the leaderboard row of the best configuration
    """

    def __init__(self, task: str, results: list[BaseResults],
                 criteria: FilteringCriteriaClassification | FilteringCriteriaRegression, features: Features,
                 grid: dict[str, list] | None = None, hyper_parameters: dict[str, list] | None = None,
                 *, class_names=None, search_log: list[dict] | None = None) -> None:
        """grid: hyper-parameters listed per configuration (None = all of each model's parameters).
        hyper_parameters: the grid the experiment searched, shown in the setup (default: grid).
        class_names: original class labels in encoded order (classification), used to label plots.
        search_log: the run's search_log (halving rounds / greedy feature steps), shown in the PDF.
        Each model's best configuration is kept for the ROC curves and the report's plots."""
        if not results:
            raise ValueError("no results to report: run k_folds(), train_test_split() or leave_one_out() first")
        self.task = task
        self.criteria = criteria
        self.n_rows = features.data.shape[0]
        self.n_columns = len(features.columns) if features.columns is not None else features.data.shape[1]
        self.created = datetime.datetime.now()
        self.feature_names = ([str(c) for c in features.columns] if features.columns is not None
                              else [f"x{i}" for i in range(self.n_columns)])
        self.preprocessing = [repr(step) for step in getattr(features, "preprocessing", [])]
        self.hyper_parameters = dict(hyper_parameters if hyper_parameters is not None else (grid or {}))
        ranked = rank(results, criteria, features.data.shape[1])
        self.leaderboard = leaderboard(ranked, grid, features)
        self._best_per_model = {}
        for result in ranked:
            self._best_per_model.setdefault(result.name, result)
        self.metric_names = list(results[0].metrics)
        self.summary = self._summarize()
        self.feature_subsets = int(self.leaderboard["features"].nunique())
        self.class_names = None if class_names is None else [str(c) for c in class_names]
        self.search_log = [dict(step) for step in (search_log or [])]
        # seconds per configuration (fit + predict, all folds), aligned with the leaderboard; NaN = not measured
        self.fit_times = pd.Series([np.nan if getattr(r, "fit_time", None) is None else float(r.fit_time)
                                    for r in ranked], index=self.leaderboard.index, name="fit_time", dtype=float)
        self.last_visualizations = None

    @property
    def best(self) -> pd.Series:
        """The leaderboard row of the best configuration."""
        return self.leaderboard.iloc[0]

    @property
    def n_configurations(self) -> int:
        """How many configurations were trained."""
        return len(self.leaderboard)

    def _summarize(self) -> pd.DataFrame:
        """Build the per-model summary, ordered by each model's best rank.

        The leaderboard is ranked, so the first row of each model's group is that model's best.
        """
        metric = self.criteria.value
        rows = []
        for model, group in self.leaderboard.groupby("model", sort=False):
            values = group[metric]
            top = group.iloc[0]
            row = {"model": model, "configurations": len(group),
                   f"best_{metric}": top[metric], f"mean_{metric}": values.mean(),
                   f"std_{metric}": values.std(ddof=0), f"worst_{metric}": group.iloc[-1][metric]}
            row.update({f"mean_{m}": group[m].mean() for m in self.metric_names if m != metric})
            row.update({"best_rank": int(group.index[0]), "best_params": top["params"],
                        "best_n_features": top["n_features"]})
            rows.append(row)
        summary = pd.DataFrame(rows).set_index("model")
        return summary.sort_values("best_rank")

    def roc_curves(self) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        """{model: (fpr, tpr, auc)} for each model's best configuration (classification only).
        Multiclass: micro-average over the classes (one-vs-rest). Models without predict_proba are left out.
        Binary targets are encoded 0 / 1, so class 1 is the positive label."""
        curves = {}
        if self.task != "classification":
            return curves
        for name, result in self._best_per_model.items():
            proba, y = getattr(result, "_predict_score", None), result.target_test
            if proba is None or y is None:
                continue
            proba = np.asarray(proba)
            if proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y, proba[:, 1], pos_label=1)
            else:
                y_bin = label_binarize(y, classes=np.arange(proba.shape[1]))
                fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
            curves[name] = (fpr, tpr, float(auc(fpr, tpr)))
        return curves

    def plot_roc_curves(self, show: bool = True) -> bool:
        """One ROC curve per model (its best configuration). False if there is nothing to plot."""
        curves = self.roc_curves()
        if not curves:
            return False
        multiclass = any(np.asarray(r._predict_score).shape[1] > 2 for r in self._best_per_model.values()
                         if getattr(r, "_predict_score", None) is not None)
        plt.figure(figsize=(7, 6))
        for name, (fpr, tpr, roc_auc) in sorted(curves.items(), key=lambda c: -c[1][2]):
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - best configuration per model" + (" (micro-average)" if multiclass else ""))
        plt.legend(loc="lower right")
        if show:
            plt.show()
        return True

    def report_lines(self) -> Iterator[str]:
        """Yield the overview: data, validation, ranking metric, setup and the best configuration."""
        best = self.best
        direction = "higher" if self.criteria.maximize else "lower"
        yield f"Experiment Report - {self.task}"
        yield ""
        yield f"Date            : {self.created:%Y-%m-%d %H:%M:%S}"
        yield f"Data            : {self.n_rows} rows, {self.n_columns} features"
        yield f"Validation      : {best['validation']}"
        yield f"Ranked by       : {self.criteria.value} ({direction} is better)"
        yield f"Models          : {len(self.summary)}"
        yield f"Configurations  : {self.n_configurations}"
        yield ""
        yield from self.setup_lines()
        yield ""
        yield f"Best: {best['model']}  {self.criteria.value}={best[self.criteria.value]:.4f}"
        yield f"  params  : {best['params'] or '(defaults)'}"
        yield f"  features: {best['n_features']} ({best['features']})"
        yield "  " + "  ".join(f"{m}={best[m]:.4f}" for m in self.metric_names)

    def setup_lines(self) -> Iterator[str]:
        """The features, preprocessing and hyper-parameters the experiment used."""
        yield "Experiment setup"
        yield f"Features ({len(self.feature_names)}):"
        for line in textwrap.wrap(", ".join(self.feature_names), 110):
            yield "  " + line
        yield f"Feature subsets tried: {self.feature_subsets}"
        yield "Preprocessing: " + (", ".join(self.preprocessing) if self.preprocessing else "none")
        yield "Hyper-parameters:" + ("" if self.hyper_parameters else " none (model defaults)")
        for name, values in self.hyper_parameters.items():
            for i, line in enumerate(textwrap.wrap(json.dumps(list(values), default=str), 90)):
                yield f"  {name if i == 0 else '':<20}{line}"
        yield "Models: " + ", ".join(self.summary.index)

    def summary_table(self) -> str:
        """The per-model summary as text, without the best_params column."""
        table = self.summary.drop(columns=["best_params"])
        return table.to_string(float_format=lambda v: f"{v:.4f}")

    def leaderboard_table(self, top: int = 10) -> str:
        """The `top` best configurations as text, without the features column."""
        board = self.leaderboard.head(top).drop(columns=["features"])
        return board.to_string(float_format=lambda v: f"{v:.4f}")

    def __str__(self) -> str:
        """The overview, the per-model summary and the top 10 configurations."""
        return ("\n".join(self.report_lines())
                + "\n\nPer model\n" + self.summary_table()
                + f"\n\nTop {min(10, self.n_configurations)} configurations\n" + self.leaderboard_table())

    def __repr__(self) -> str:
        """Task, number of configurations and the best model's score."""
        return (f"ExperimentReport({self.task}, {self.n_configurations} configurations, "
                f"best {self.best['model']} {self.criteria.value}={self.best[self.criteria.value]:.4f})")

    def save_csv(self, folder: str | Path) -> tuple[Path, Path]:
        """Write leaderboard.csv and summary.csv into `folder`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        board, summary = folder / "leaderboard.csv", folder / "summary.csv"
        self.leaderboard.to_csv(board, index_label="rank")
        self.summary.to_csv(summary)
        return board, summary

    def visualize(self):
        """Every plot that applies to this experiment, each generated in isolation.

        Returns a VisualizationReport (one SUCCESS / SKIPPED / FAILED record per plot, PNG bytes
        for the successful ones; .save_images(folder) writes them). Never raises; the results
        are only read."""
        from ml_pipe.visualization import VisualizationContext, VisualizationManager
        return VisualizationManager().run_for(lambda: VisualizationContext.from_experiment(self))

    def save_pdf_report(self, folder: str | Path, top: int = 20) -> Path:
        """Write experiment_report_<date>.pdf into `folder` and return its path.

        Sections (only those with content): executive summary, configuration, dataset, model
        comparison, classification or regression analysis, feature analysis, cross-validation,
        hyper-parameter optimization, runtime, then an index of every visualization and the
        skipped / failed ones. Each plot is generated in isolation, so one failing plot never
        removes the others; `top` = configurations listed in the leaderboard table."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"experiment_report_{self.created:%Y-%m-%d_%H-%M-%S}.pdf"
        visualizations = self.visualize()
        pdf = self._pdf(visualizations, top)
        ranks = {model: f"rank #{int(row['best_rank'])}" for model, row in self.summary.iterrows()}
        try:
            return pdf.build(path, visualizations, identifiers=ranks)
        finally:  # statuses stay inspectable; the images are released
            self.last_visualizations = visualizations.without_images()

    def _pdf(self, visualizations, top: int):
        """The text part of the PDF (the figures are added when it is built)."""
        from ml_pipe.visualization.pdf import PdfReport
        best, metric = self.best, self.criteria.value
        direction = "higher is better" if self.criteria.maximize else "lower is better"
        pdf = PdfReport(f"Experiment Report - {self.task}",
                        f"{self.n_configurations} configurations of {len(self.summary)} model(s), ranked by "
                        f"{metric} ({direction})  ·  {self.created:%Y-%m-%d %H:%M}")

        section = "Executive Summary"
        pdf.key_values(section, [
            ("best configuration", f"{best['model']}  (rank #1)"),
            (f"{metric}", f"{best[metric]:.4f}  ({direction})"),
            ("all metrics", ", ".join(f"{m} {best[m]:.4f}" for m in self.metric_names)),
            ("hyper-parameters", best["params"] or "(defaults)"),
            ("features", f"{best['n_features']}: {textwrap.shorten(str(best['features']), 300)}"),
            ("validation", str(best["validation"])),
            ("configurations / models", f"{self.n_configurations} / {len(self.summary)}"),
            ("visualizations", visualizations.summary()),
        ])
        pdf.bullets(section, self._findings())

        section = "Experiment Configuration"
        grid = [(f"grid: {name}", json.dumps(list(values), default=str))
                for name, values in self.hyper_parameters.items()]
        pdf.key_values(section, [
            ("created", f"{self.created:%Y-%m-%d %H:%M:%S}"),
            ("task", self.task), ("validation", str(best["validation"])),
            ("ranking metric", f"{metric} ({direction})"),
            ("models", ", ".join(self.summary.index)),
            ("preprocessing", ", ".join(self.preprocessing) or "none"),
            ("feature subsets tried", str(self.feature_subsets)),
        ] + (grid or [("hyper-parameters", "none (model defaults)")]))
        if self.search_log:
            pdf.heading(section, "Search rounds")
            pdf.table(section, pd.DataFrame(self.search_log), index=False)

        section = "Dataset Overview"
        pdf.key_values(section, [("rows", str(self.n_rows)), ("features", str(self.n_columns))])
        pdf.paragraph(section, "Feature columns: " + ", ".join(self.feature_names))

        section = "Overall Results and Model Comparison"
        pdf.heading(section, "Per model")
        pdf.paragraph(section, f"Best, mean, spread and worst {metric} over each model's configurations, and the "
                               f"mean of every other metric.")
        pdf.table(section, self.summary.drop(columns=["best_params"]))
        pdf.heading(section, f"Top {min(top, self.n_configurations)} configurations")
        pdf.table(section, self.leaderboard.drop(columns=["features"]).rename_axis("rank"), max_rows=top)
        return pdf

    def _findings(self) -> list[str]:
        """Plain statements read off the results (nothing estimated)."""
        metric = self.criteria.value
        out = []
        if len(self.summary) > 1:
            first, second = self.summary.index[0], self.summary.index[1]
            gap = abs(self.summary.iloc[0][f"best_{metric}"] - self.summary.iloc[1][f"best_{metric}"])
            out.append(f"{first} leads {second} by {gap:.4f} {metric} (best configuration of each).")
        spread = self.summary[f"std_{metric}"]
        if spread.notna().any() and (self.summary["configurations"] > 1).any():
            model = spread.idxmax()
            out.append(f"{model} varies most across its configurations (std of {metric} {spread.max():.4f}).")
        if self.task == "classification":
            result = self._best_per_model.get(self.best["model"])
            y = None if result is None else result.target_test
            if y is not None and len(y):
                _, counts = np.unique(y, return_counts=True)
                out.append(f"Always predicting the most frequent class would score accuracy "
                           f"{counts.max() / counts.sum():.4f}; the best model scores {self.best['accuracy']:.4f}.")
        times = self.fit_times.dropna()
        if len(times):
            out.append(f"Training time per configuration: median {times.median():.3f} s, "
                       f"total {times.sum():.1f} s of worker time.")
        out.append("The best of many configurations is optimistically biased; confirm on held-out data.")
        return out
