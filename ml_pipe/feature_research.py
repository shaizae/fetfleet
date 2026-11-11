"""Understand the features before (or after) training: quality, signal, redundancy and importance.

Stand-alone: give it a feature table and a target, it does not need a Classification / Regression run.

    research = FeatureResearch(X, y)            # task is guessed from y, or task="classification"
    research.missing_report()
    research.mutual_information()
    research.high_correlation_pairs(0.9)
    research.permutation_importance()
    research.random_probe()
    print(research.report())
    X = X.drop(columns=research.recommend().index)
    research.save_pdf_report("reports")         # everything above in one PDF, with charts
"""
from __future__ import annotations

import datetime
import logging
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_pipe.features import Features
from ml_pipe.target import Target

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASKS = ("auto", CLASSIFICATION, REGRESSION)

# An integer target with at most this many distinct values is treated as classes when task="auto".
MAX_CLASSES_AUTO = 20

# Chart colors (light, print-friendly): one hue for magnitude, blue <-> gray <-> red for correlation.
BAR_COLOR = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
CORRELATION_CMAP = LinearSegmentedColormap.from_list("correlation", ["#e34948", "#f0efec", "#2a78d6"])
# The heatmap writes the value in each cell up to this many features (beyond that the cells get too small).
MAX_ANNOTATED_FEATURES = 15


class FeatureResearch:
    """Per-feature diagnostics for one feature table and its target.

    X             DataFrame (or 2-D array; columns are then named x0, x1, ...)
    y             target, one value per row (no missing values)
    task          "classification", "regression" or "auto" (strings / bools / categories / integers with
                  at most 20 distinct values are classes, anything else is regression)
    cv            folds for every cross-validated method (stratified for classification when possible)
    random_state  seed for the folds, the default models, mutual information and the noise probes

    The data is kept the same way a Classification / Regression run keeps it: the features in a
    `Features` object (`research.features`) and the target in a `Target` object (`research.targets`,
    label-encoded for classification; `targets.classes` maps codes back to labels). Every method reads
    `features.data`, so a step applied there (e.g. `research.features.standard_scaler()`) is seen by
    the methods that follow.

    Statistical methods use the numeric columns only (bools count as numeric) and fill their missing
    values with the column median; `skipped` lists the columns they leave out. The data-quality methods
    (missing_report, constant_features, cardinality) look at every column.
    """

    def __init__(self, X: pd.DataFrame | np.ndarray, y, task: str = "auto", cv: int = 5,
                 random_state: int | None = 42) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if cv < 2:
            raise ValueError("cv must be at least 2")
        if not isinstance(X, pd.DataFrame):
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X must be 2-D")
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(np.asarray(y).ravel() if not isinstance(y, pd.Series) else y.to_numpy())
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
        if y.isna().any():
            raise ValueError("y has missing values: drop those rows first")
        if X.shape[1] == 0:
            raise ValueError("X has no columns")

        self.task = self._guess_task(y) if task == "auto" else task
        self.cv = cv
        self.random_state = random_state

        X = X.reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        # Features stores one numpy array, so mixed column types become `object`: keep each column's
        # dtype to rebuild the table with the right types (numeric / text / bool decide what runs).
        self._dtypes = X.dtypes.to_dict()
        self.features = Features()
        self.features.load_new_data(X)

        self.targets = Target(encode=self.is_classification)
        self.targets.load_new_data(y.to_numpy(copy=True))

    # ------------------------------------------------------------------ data
    @property
    def X(self) -> pd.DataFrame:
        """The current `features.data` as a DataFrame with the original column names and types.
        After a step that changes the number of columns (polynomial_features) the columns are
        named x0, x1, ... and keep the types numpy gives them."""
        data = self.features.data
        columns = list(self.features.columns)
        if data.shape[1] != len(columns):
            return pd.DataFrame(data, columns=[f"x{i}" for i in range(data.shape[1])]).infer_objects()
        frame = pd.DataFrame(data, columns=columns)
        if self.features.preprocessing:  # scalers return floats: the original types no longer apply
            return frame.infer_objects()
        return frame.astype(self._dtypes)

    @property
    def y(self) -> pd.Series:
        """`targets.data` as a Series: class codes 0..k-1 (classification) or floats (regression)."""
        return pd.Series(self.targets.data, name="target")

    @property
    def _numeric(self) -> pd.DataFrame:
        """Numeric columns as floats, missing values filled with the column median
        (a column with no values at all is filled with 0)."""
        numeric = self.X.select_dtypes(include=["number", "bool"]).astype(float)
        return numeric.fillna(numeric.median()).fillna(0.0)

    @property
    def skipped(self) -> list[str]:
        """Columns the statistical methods leave out (not numeric)."""
        X = self.X
        numeric = set(X.select_dtypes(include=["number", "bool"]).columns)
        return [c for c in X.columns if c not in numeric]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _guess_task(y: pd.Series) -> str:
        if (y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y)
                or pd.api.types.is_string_dtype(y)):
            return CLASSIFICATION
        if pd.api.types.is_integer_dtype(y) and y.nunique() <= MAX_CLASSES_AUTO:
            return CLASSIFICATION
        if pd.api.types.is_float_dtype(y) and (y == y.round()).all() and y.nunique() <= MAX_CLASSES_AUTO:
            return CLASSIFICATION
        return REGRESSION

    @property
    def is_classification(self) -> bool:
        return self.task == CLASSIFICATION

    @property
    def numeric_features(self) -> list[str]:
        """The numeric columns the statistical methods use."""
        return list(self.X.select_dtypes(include=["number", "bool"]).columns)

    def _require_numeric(self) -> pd.DataFrame:
        if self._numeric.shape[1] == 0:
            raise ValueError("X has no numeric columns: encode the categorical ones first")
        return self._numeric

    def _splitter(self):
        if self.is_classification and self.y.value_counts().min() >= self.cv:
            return StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        return KFold(self.cv, shuffle=True, random_state=self.random_state)

    def _default_model(self) -> BaseEstimator:
        cls = RandomForestClassifier if self.is_classification else RandomForestRegressor
        return cls(n_estimators=100, random_state=self.random_state, n_jobs=-1)

    def _default_scoring(self) -> str:
        return "accuracy" if self.is_classification else "r2"

    def _cv_permutation(self, X: pd.DataFrame, model: BaseEstimator | None, n_repeats: int,
                        scoring: str | None) -> pd.DataFrame:
        """Permutation importance on the held-out part of every fold, averaged over the folds."""
        model = model if model is not None else self._default_model()
        scoring = scoring or self._default_scoring()
        y = self.y
        per_fold = []
        for train, test in self._splitter().split(X, y):
            fitted = clone(model).fit(X.iloc[train], y.iloc[train])
            result = sk_permutation_importance(fitted, X.iloc[test], y.iloc[test], scoring=scoring,
                                               n_repeats=n_repeats, random_state=self.random_state)
            per_fold.append(result.importances_mean)
        per_fold = np.array(per_fold)
        return pd.DataFrame({"importance": per_fold.mean(axis=0), "std_across_folds": per_fold.std(axis=0)},
                            index=X.columns).sort_values("importance", ascending=False)

    # ------------------------------------------------------------- data quality
    def missing_report(self) -> pd.DataFrame:
        """Missing values per column, most missing first."""
        X = self.X
        count = X.isna().sum()
        return pd.DataFrame({"missing": count, "missing_pct": count / len(X) * 100}).sort_values(
            "missing", ascending=False)

    def constant_features(self, min_share: float = 1.0) -> pd.DataFrame:
        """Columns whose most common value covers at least `min_share` of the rows
        (1.0 = constant, 0.99 = quasi-constant). Missing values count as a value."""
        if not 0 < min_share <= 1:
            raise ValueError("min_share must be in (0, 1]")
        X = self.X
        rows = {}
        for col in X.columns:
            counts = X[col].value_counts(dropna=False)
            share = counts.iloc[0] / len(X)
            if share >= min_share:
                rows[col] = {"top_value": counts.index[0], "top_share": share}
        return pd.DataFrame(rows, index=["top_value", "top_share"]).T.astype({"top_share": float})

    def cardinality(self, id_ratio: float = 0.95) -> pd.DataFrame:
        """Distinct values per column. `id_like` flags non-float columns where almost every row is
        unique (customer ids, row numbers, timestamps): they rarely generalize."""
        X = self.X
        n_unique = X.nunique(dropna=True)
        ratio = n_unique / len(X)
        id_like = [(ratio[c] >= id_ratio) and not pd.api.types.is_float_dtype(X[c]) for c in X.columns]
        return pd.DataFrame({"dtype": X.dtypes.astype(str), "n_unique": n_unique, "unique_ratio": ratio,
                             "id_like": id_like}).sort_values("n_unique", ascending=False)

    def distribution_summary(self) -> pd.DataFrame:
        """Skew, kurtosis and share of IQR outliers (outside Q1 - 1.5 IQR .. Q3 + 1.5 IQR) per numeric column.
        |skew| > 1 usually means a log / power transform is worth trying."""
        X = self.X.select_dtypes(include=["number", "bool"]).astype(float)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        outliers = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
        return pd.DataFrame({"mean": X.mean(),"min":X.min(),"max":X.max(),"median":X.median(), "std": X.std(), "skew": X.skew(), "kurtosis": X.kurt(),
                             "outlier_pct": outliers / X.notna().sum().replace(0, np.nan) * 100})

    # ------------------------------------------------- feature vs target (univariate)
    def univariate_scores(self) -> pd.DataFrame:
        """ANOVA F-test (classification) or F-test + Pearson / Spearman correlation (regression)
        of each feature against the target, strongest first. Linear relations only."""
        X = self._require_numeric()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant columns give a divide-by-zero warning and NaN
            y = self.y
            f, p = (f_classif if self.is_classification else f_regression)(X, y)
        table = pd.DataFrame({"f_score": f, "p_value": p}, index=X.columns)
        if not self.is_classification:
            table["pearson"] = X.corrwith(y)
            table["spearman"] = X.corrwith(y, method="spearman")
        return table.sort_values("f_score", ascending=False)

    def mutual_information(self) -> pd.Series:
        """Mutual information between each feature and the target, highest first.
        0 = independent; also catches non-linear relations the F-test misses."""
        X = self._require_numeric()
        fn = mutual_info_classif if self.is_classification else mutual_info_regression
        mi = fn(X, self.y, random_state=self.random_state)
        return pd.Series(mi, index=X.columns, name="mutual_information").sort_values(ascending=False)

    def single_feature_score(self, model: BaseEstimator | None = None, scoring: str | None = None) -> pd.DataFrame:
        """Cross-validated score of a model trained on one feature at a time, best first.
        Default model: a depth-4 decision tree; default scoring: accuracy / r2."""
        X = self._require_numeric()
        if model is None:
            cls = DecisionTreeClassifier if self.is_classification else DecisionTreeRegressor
            model = cls(max_depth=4, random_state=self.random_state)
        scoring = scoring or self._default_scoring()
        splitter = self._splitter()
        y = self.y
        rows = {}
        for col in X.columns:
            scores = cross_val_score(clone(model), X[[col]], y, cv=splitter, scoring=scoring)
            rows[col] = {"score": scores.mean(), "std": scores.std()}
        return pd.DataFrame(rows).T.sort_values("score", ascending=False)

    # ------------------------------------------------------------- redundancy
    def correlation_matrix(self, method: str = "spearman") -> pd.DataFrame:
        """Feature-to-feature correlation ("pearson", "spearman" or "kendall"), on the raw values
        (missing values are skipped pairwise, not filled)."""
        return self.X.select_dtypes(include=["number", "bool"]).astype(float).corr(method=method)

    def high_correlation_pairs(self, threshold: float = 0.9, method: str = "spearman") -> pd.DataFrame:
        """Pairs with |correlation| >= threshold, strongest first. `drop` names the member with the
        lower mutual information with the target, so dropping every `drop` keeps the most informative
        feature of each correlated group."""
        corr = self.correlation_matrix(method)
        mi = self.mutual_information()
        cols = corr.columns
        rows = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iat[i, j]
                if pd.notna(value) and abs(value) >= threshold:
                    a, b = cols[i], cols[j]
                    rows.append({"feature_a": a, "feature_b": b, "correlation": value,
                                 "drop": b if mi[a] >= mi[b] else a})
        table = pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation", "drop"])
        return table.reindex(table["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def vif(self) -> pd.Series:
        """Variance inflation factor: 1 / (1 - R²) of each feature regressed on all the others, highest first.
        Above 5-10 means strong multicollinearity (matters for linear models, much less for trees)."""
        X = self._require_numeric()
        if X.shape[1] < 2:
            raise ValueError("vif needs at least two numeric features")
        values = {}
        for col in X.columns:
            others = X.drop(columns=col)
            r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
            values[col] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
        return pd.Series(values, name="vif").sort_values(ascending=False)

    # -------------------------------------------------------- model importance
    def model_importance(self, model: BaseEstimator | None = None) -> pd.Series:
        """`feature_importances_` (trees) or |coef_| (linear models, averaged over classes) of the model
        fitted on all rows. Fast, but biased (trees favour high-cardinality features, coefficients
        depend on scale); prefer permutation_importance."""
        X = self._require_numeric()
        fitted = clone(model if model is not None else self._default_model()).fit(X, self.y)
        if hasattr(fitted, "feature_importances_"):
            values = fitted.feature_importances_
        elif hasattr(fitted, "coef_"):
            coef = np.abs(np.asarray(fitted.coef_))
            values = coef.mean(axis=0) if coef.ndim == 2 else coef
        else:
            raise ValueError(f"{type(fitted).__name__} has neither feature_importances_ nor coef_: "
                             "use permutation_importance")
        return pd.Series(values, index=X.columns, name="importance").sort_values(ascending=False)

    def permutation_importance(self, model: BaseEstimator | None = None, n_repeats: int = 5,
                               scoring: str | None = None) -> pd.DataFrame:
        """Score drop when a feature is shuffled, measured on held-out folds (any model works).
        `importance` near or below 0 = the model does not need the feature.
        `std_across_folds` large next to `importance` = the ranking is not stable."""
        return self._cv_permutation(self._require_numeric(), model, n_repeats, scoring)

    # ---------------------------------------------------------- sanity checks
    def random_probe(self, model: BaseEstimator | None = None, n_probes: int = 5, n_repeats: int = 5,
                     scoring: str | None = None) -> pd.DataFrame:
        """Add `n_probes` pure-noise columns and compare each feature with the best of them, twice:

        beats_noise     permutation importance above the best noise column's
        mi_beats_noise  mutual information with the target above the best noise column's

        Permutation importance alone under-rates correlated features: when several features carry the
        same information, shuffling one costs the model nothing (the others stand in), so each looks
        useless. Mutual information looks at one feature at a time and does not have that blind spot.
        A feature that beats neither (`is_noise`) is probably useless; one that only fails
        `beats_noise` is more likely redundant - see high_correlation_pairs()."""
        if n_probes < 1:
            raise ValueError("n_probes must be at least 1")
        X = self._require_numeric().copy()
        rng = np.random.default_rng(self.random_state)
        probes = [f"__probe_{i}" for i in range(n_probes)]
        for name in probes:
            X[name] = rng.standard_normal(len(X))
        table = self._cv_permutation(X, model, n_repeats, scoring)
        mi_fn = mutual_info_classif if self.is_classification else mutual_info_regression
        table["mutual_information"] = pd.Series(mi_fn(X, self.y, random_state=self.random_state), index=X.columns)
        noise_level = table.loc[probes, "importance"].max()
        mi_noise_level = table.loc[probes, "mutual_information"].max()
        table = table.drop(index=probes)
        table["noise_level"] = noise_level
        table["mi_noise_level"] = mi_noise_level
        table["beats_noise"] = table["importance"] > noise_level
        table["mi_beats_noise"] = table["mutual_information"] > mi_noise_level
        table["is_noise"] = ~table["beats_noise"] & ~table["mi_beats_noise"]
        return table

    def leakage_check(self, threshold: float = 0.98, model: BaseEstimator | None = None,
                      scoring: str | None = None) -> pd.DataFrame:
        """Features that alone reach a cross-validated score >= threshold. That is rarely real signal:
        check that they are not the target in disguise or only known after the outcome."""
        scores = self.single_feature_score(model, scoring)
        return scores[scores["score"] >= threshold]

    # ------------------------------------------------------------- reporting
    def report(self, model: BaseEstimator | None = None, importance: bool = True) -> pd.DataFrame:
        """One row per numeric feature: missing %, distinct values, F-score, mutual information,
        highest |correlation| with another feature, permutation importance (when `importance`),
        a rank per score and their mean (`mean_rank`, 1 = best). Sorted by `mean_rank`."""
        X = self._require_numeric()
        cols = X.columns
        table = pd.DataFrame(index=cols)
        table["missing_pct"] = self.missing_report().loc[cols, "missing_pct"]
        table["n_unique"] = self.X[cols].nunique()
        table["f_score"] = self.univariate_scores()["f_score"]
        table["mutual_information"] = self.mutual_information()
        if len(cols) > 1:
            corr = self.correlation_matrix().abs()
            corr = corr.mask(np.eye(len(corr), dtype=bool))  # ignore each feature's correlation with itself
            table["max_abs_corr"] = corr.max()
        ranked = ["f_score", "mutual_information"]
        if importance:
            table["permutation_importance"] = self.permutation_importance(model)["importance"]
            ranked.append("permutation_importance")
        for col in ranked:
            table[f"rank_{col}"] = table[col].rank(ascending=False, na_option="bottom")
        table["mean_rank"] = table[[f"rank_{c}" for c in ranked]].mean(axis=1)
        return table.sort_values("mean_rank")

    def recommend(self, missing_threshold: float = 0.5, correlation_threshold: float = 0.95,
                  probe: bool = False, model: BaseEstimator | None = None) -> pd.DataFrame:
        """Columns worth dropping, with the reason(s). Index = column name, so
        `X.drop(columns=research.recommend().index)` applies it.

        constant      one value in every row
        id_like       almost every row unique (non-float)
        missing       more than `missing_threshold` (share of rows) missing
        redundant     |correlation| >= correlation_threshold with a more informative feature
        noise         beats random noise columns on neither permutation importance nor mutual
                      information (only with probe=True; trains models; see random_probe)

        Leakage is not a drop reason: run leakage_check() and decide yourself.
        """
        probed = self.random_probe(model) if probe else None
        return self._recommend(missing_threshold, correlation_threshold, probed)

    def _recommend(self, missing_threshold: float, correlation_threshold: float,
                   probed: pd.DataFrame | None) -> pd.DataFrame:
        """recommend() with the random_probe() table already computed (None = no noise check)."""
        reasons: dict[str, list[str]] = {}

        def add(cols, reason):
            for c in cols:
                reasons.setdefault(c, []).append(reason)

        add(self.constant_features(1.0).index, "constant")
        card = self.cardinality()
        add(card.index[card["id_like"]], "id_like")
        missing = self.missing_report()
        add(missing.index[missing["missing_pct"] > missing_threshold * 100], "missing")
        if len(self.numeric_features) > 1:
            add(self.high_correlation_pairs(correlation_threshold)["drop"].unique(), "redundant")
        if probed is not None:
            add(probed.index[probed["is_noise"]], "noise")
        return pd.DataFrame({"reasons": [", ".join(r) for r in reasons.values()]},
                            index=pd.Index(list(reasons), name="feature"))

    # ------------------------------------------------------------- charts
    def plot_importance(self, importance: pd.DataFrame | None = None, top: int = 25,
                        model: BaseEstimator | None = None) -> plt.Figure:
        """Horizontal bars of permutation importance, most important on top (at most `top` features), with
        the spread across folds. `importance`: a permutation_importance() or random_probe() table (computed
        with `model` when None); a random_probe() table also draws the best noise column as a dashed line."""
        table = importance if importance is not None else self.permutation_importance(model)
        table = table.sort_values("importance", ascending=False).head(top).iloc[::-1]  # barh draws bottom-up
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(table) + 1.4))
        ax.barh(table.index, table["importance"], height=0.6, color=BAR_COLOR, zorder=2)
        ax.errorbar(table["importance"], table.index, xerr=table["std_across_folds"], fmt="none",
                    ecolor=INK_MUTED, elinewidth=1, capsize=2, zorder=3)
        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
        if "noise_level" in table:
            level = table["noise_level"].iloc[0]
            ax.axvline(level, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=4)
            ax.annotate("best noise column", (level, 1), xycoords=("data", "axes fraction"), xytext=(4, -10),
                        textcoords="offset points", fontsize=8, color=INK_MUTED)
        scoring = self._default_scoring()
        ax.set_title(f"Permutation importance (drop in {scoring} when the feature is shuffled)",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"{scoring} drop (mean over {self.cv} folds, bar = spread across folds)",
                      fontsize=8, color=INK_MUTED)
        self._style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        fig.tight_layout()
        return fig

    def plot_correlation(self, method: str = "spearman") -> plt.Figure:
        """Heatmap of the feature-to-feature correlation (blue = positive, red = negative, gray = none)."""
        corr = self.correlation_matrix(method)
        n = len(corr)
        size = min(max(0.45 * n + 2, 4), 11)
        fig, ax = plt.subplots(figsize=(size + 1, size))
        image = ax.imshow(corr.values, cmap=CORRELATION_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=7, color=INK)
        ax.set_yticks(range(n), corr.columns, fontsize=7, color=INK)
        if n <= MAX_ANNOTATED_FEATURES:
            for i in range(n):
                for j in range(n):
                    value = corr.iat[i, j]
                    if pd.notna(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if abs(value) > 0.6 else INK)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(f"{method} correlation", fontsize=8, color=INK_MUTED)
        bar.ax.tick_params(labelsize=7, colors=INK_MUTED)
        bar.outline.set_visible(False)
        ax.set_title(f"Feature correlation ({method})", loc="left", fontsize=10, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        fig.tight_layout()
        return fig

    @staticmethod
    def _style_axes(ax) -> None:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.tick_params(axis="y", length=0, labelsize=8, labelcolor=INK)
        ax.tick_params(axis="x", labelsize=7, colors=INK_MUTED)

    # ------------------------------------------------------------- PDF report
    def overview_lines(self) -> list[str]:
        """Task, data size, preprocessing and target summary, one line each."""
        rows, columns = self.features.data.shape
        skipped = self.skipped
        steps = [type(step).__name__ for step in self.features.preprocessing]
        lines = [f"Task:           {self.task}",
                 f"Rows:           {rows}",
                 f"Columns:        {columns} ({len(self.numeric_features)} numeric)",
                 f"Skipped:        {', '.join(skipped) if skipped else 'none'} (not numeric)",
                 f"Preprocessing:  {', '.join(steps) if steps else 'none'}",
                 f"Validation:     {self.cv}-fold cross-validation, random_state={self.random_state}"]
        y = self.y
        if self.is_classification:
            labels = self.targets.classes
            counts = y.value_counts().sort_index()
            parts = [f"{labels[code] if labels is not None else code}: {n} ({n / len(y):.0%})"
                     for code, n in counts.items()]
            lines.append(f"Target classes: {', '.join(parts)}")
        else:
            lines.append(f"Target:         mean {y.mean():.4g}, std {y.std():.4g}, "
                         f"min {y.min():.4g}, max {y.max():.4g}")
        return lines

    def save_pdf_report(self, folder: str | Path, model: BaseEstimator | None = None, probe: bool = True,
                        correlation_threshold: float = 0.9, missing_threshold: float = 0.5,
                        leakage_threshold: float = 0.98) -> Path:
        """Write feature_research_<date>.pdf into `folder` and return its path.

        Pages: overview and recommendations (columns to drop, possible leakage); the feature ranking table;
        the importance chart; data quality (missing, constant, id-like, distributions); correlation (pairs
        >= correlation_threshold, VIF, heatmap).
        probe=True compares importances with random noise columns (adds the `noise` drop reason and the
        noise line on the chart); False uses plain permutation importance. Either way models are trained
        (`model`, default random forest) once per fold."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        created = datetime.datetime.now()
        path = folder / f"feature_research_{created:%Y-%m-%d_%H-%M-%S}.pdf"
        numeric = self._require_numeric()

        importance = self.random_probe(model) if probe else self.permutation_importance(model)
        ranking = self.report(importance=False)
        ranking.insert(ranking.columns.get_loc("rank_f_score"), "permutation_importance",
                       importance["importance"])
        ranking["rank_permutation_importance"] = ranking["permutation_importance"].rank(ascending=False,
                                                                                        na_option="bottom")
        rank_cols = [c for c in ranking.columns if c.startswith("rank_")]
        ranking["mean_rank"] = ranking[rank_cols].mean(axis=1)
        ranking = ranking.drop(columns=rank_cols).sort_values("mean_rank")
        recommended = self._recommend(missing_threshold, correlation_threshold, importance if probe else None)
        leaks = self.leakage_check(leakage_threshold)

        styles = getSampleStyleSheet()
        code = styles["Code"]
        small = code.clone("SmallCode", fontSize=7, leading=8.5)
        body, h2, h3 = styles["BodyText"], styles["Heading2"], styles["Heading3"]

        def table(frame: pd.DataFrame | pd.Series, empty: str):
            if len(frame) == 0:
                return Paragraph(empty, body)
            return Preformatted(frame.to_string(float_format=lambda v: f"{v:.4f}"), small)

        quality = self.missing_report()
        quality = quality[quality["missing"] > 0]
        card = self.cardinality()
        several = numeric.shape[1] > 1

        with TemporaryDirectory() as tmpdir:
            def chart(draw, name: str):
                """The chart drawn by draw() as an Image; if drawing fails, a note instead (the
                report goes on without it)."""
                fig = None
                try:
                    fig = draw()
                    image_path = Path(tmpdir) / name
                    width, height = (v * 72 for v in fig.get_size_inches())
                    fig.savefig(image_path, dpi=150, bbox_inches="tight")
                    scale = min(680 / width, 430 / height, 1.4)  # fit the landscape A4 frame (~686 x 439 pt)
                    return Image(str(image_path), width=width * scale, height=height * scale)
                except Exception as e:
                    logging.getLogger("ml_pipe.visualization").warning("chart %s failed: %s", name, e)
                    return Paragraph(f"Chart {name} could not be drawn ({type(e).__name__}: {e}).", body)
                finally:
                    if fig is not None:
                        plt.close(fig)

            elements = [Paragraph("Feature Research Report", styles["Title"]),
                        Paragraph(f"Created {created:%Y-%m-%d %H:%M}", body), Spacer(1, 8)]
            elements += [Preformatted(line, code) for line in self.overview_lines()]
            elements += [
                Spacer(1, 12), Paragraph("Recommended to drop", h2),
                Paragraph(f"constant / id-like / more than {missing_threshold:.0%} missing / |correlation| "
                          f">= {correlation_threshold} with a more informative feature"
                          + (" / beats random noise columns on neither permutation importance nor mutual information"
                             if probe else ""), body),
                table(recommended, "Nothing to drop."),
                Spacer(1, 12), Paragraph("Possible leakage", h2),
                Paragraph(f"Features that alone reach a cross-validated {self._default_scoring()} of "
                          f"{leakage_threshold} or more. Check they are not the target in disguise.", body),
                table(leaks, "None found."),

                PageBreak(), Paragraph("Feature ranking", h2),
                Paragraph("mean_rank = average rank of f_score, mutual_information and "
                          "permutation_importance (1 = best).", body),
                table(ranking, "No numeric features."),

                PageBreak(), chart(lambda: self.plot_importance(importance), "importance.png"),

                PageBreak(), Paragraph("Data quality", h2),
                Paragraph("Missing values", h3), table(quality, "No missing values."),
                Paragraph("Constant / quasi-constant (top value in 99% of rows or more)", h3),
                table(self.constant_features(0.99), "None."),
                Paragraph("Id-like columns (almost every row unique)", h3), table(card[card["id_like"]], "None."),
                Paragraph("Distributions (|skew| > 1: consider a log / power transform)", h3),
                table(self.distribution_summary(), "No numeric features."),
            ]
            if several:
                elements += [
                    PageBreak(), Paragraph("Correlation", h2),
                    Paragraph(f"Pairs with |correlation| >= {correlation_threshold}", h3),
                    table(self.high_correlation_pairs(correlation_threshold), "None."),
                    Paragraph("Variance inflation factor (> 5-10 = strong multicollinearity)", h3),
                    table(self.vif().to_frame(), ""),
                    PageBreak(), chart(self.plot_correlation, "correlation.png"),
                ]
            SimpleDocTemplate(str(path), pagesize=landscape(A4), title="Feature Research Report").build(elements)
        return path

    def __repr__(self) -> str:
        rows, columns = self.features.data.shape
        return (f"FeatureResearch(task={self.task!r}, rows={rows}, columns={columns}, "
                f"numeric={len(self.numeric_features)}, cv={self.cv})")
