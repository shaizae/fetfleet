"""Run an experiment described in a TOML or JSON file (see `ml-pipe init`)."""
from __future__ import annotations

import datetime
import difflib
import importlib
import json
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd
import sklearn.feature_selection

from ml_pipe.base import BaseML
from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.results.experiment import ExperimentReport, leaderboard as _leaderboard
from ml_pipe.search import DIRECTIONS, STRATEGIES, rank as _rank
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, Sound
from ml_pipe.visualization.background import ReportWorker

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from ml_pipe.features import Features
    from ml_pipe.results.base import BaseResults

TASKS = {
    "classification": (Classification, FilteringCriteriaClassification, "accuracy"),
    "regression": (Regression, FilteringCriteriaRegression, "r2"),
}
VALIDATIONS = ("train_test_split", "k_folds", "leave_one_out")
PREPROCESSING = ("standard_scaler", "min_max_scaler", "polynomial_features")

EXAMPLE_CONFIG = '''\
# ml_pipe experiment. Run with:  ml-pipe run experiment.toml
# Relative paths are relative to this file.

[data]
path = "data.csv"            # .csv, .parquet, .xlsx (needs openpyxl) or .json
target = "label"             # the column to predict
# features = ["a", "b"]      # default: every other column
task = "classification"      # or "regression"

[run]
validation = "k_folds"       # train_test_split | k_folds | leave_one_out
n_splits = 5                 # k_folds only
# train_ratio = 0.8          # train_test_split only: fraction of rows used for training
select_by = "accuracy"       # classification: accuracy f1 precision recall | regression: r2 mae mse rmse
random_state = 0
# processes = 4              # default: half the CPU cores

# applied in order, refitted on the training rows of every split
preprocessing = ["standard_scaler"]
# preprocessing = ["min_max_scaler", { polynomial_features = { degree = 2 } }]

# [feature_selection]        # optional
# method = "f_classif"       # a sklearn.feature_selection score function, "brute_force" (every subset),
#                            # "forward" or "backward" (greedy: add / remove one feature at a time)
# k = [2, 4, 8]              # numbers of features to try (score functions only)
# max_features = 10          # forward: stop at this many features
# min_features = 1           # backward: stop at this many features
# tol = 0.0                  # forward: a feature must improve the score by more than this;
#                            # backward: a removal may lower it by at most this

# [search]                   # optional: try fewer configurations than every combination
# strategy = "halving"       # grid (all, default) | random | halving
# n_iter = 200               # random: how many to try; halving: optional random sample to start from
# factor = 3                 # halving: keep the best 1/3 each round, on 3x more rows
# min_rows = 100             # halving: rows in the first round (default: from the data)

# a scikit-learn class name, or a full import path such as "xgboost.XGBClassifier"
[[models]]
name = "RandomForestClassifier"
params = { random_state = 0 }

[[models]]
name = "LogisticRegression"
params = { max_iter = 1000 }

# every combination is tried; each value is applied only to models that have that parameter
[hyper_parameters]
n_estimators = [100, 300]
C = [0.1, 1, 10]

[output]
dir = "runs"                 # each run writes runs/<date_time>/
keep = 1                     # how many of the top configurations to refit and save
report = true                # PDF report for the experiment and each saved model
sound = "none"               # played when the whole experiment ends:
                             # none | ping | double | chime | success | sparkle | alarm | bell
'''


class ConfigError(ValueError):
    """An experiment config (or its data file) is invalid."""
    pass


@dataclass
class Experiment:
    """One experiment: the data, how to validate, what to search and where to write the results.

    Usually built from a config file with from_file(); the fields mirror its sections.
    """
    data_path: Path
    target: str
    task: str
    features: list[str] | None = None
    validation: str = "k_folds"
    n_splits: int = 5
    train_ratio: float = 0.8
    select_by: str | None = None
    random_state: int | None = None
    processes: int | None = None
    preprocessing: list = field(default_factory=list)
    selection_method: str | None = None
    selection_k: list[int] = field(default_factory=list)
    selection_max_features: int | None = None
    selection_min_features: int = 1
    selection_tol: float = 0.0
    search_strategy: str = "grid"
    search_n_iter: int | None = None
    search_factor: float = 3
    search_min_rows: int | None = None
    models: list[dict] = field(default_factory=list)
    hyper_parameters: dict[str, list] = field(default_factory=dict)
    output_dir: Path = Path("runs")
    keep: int = 1
    report: bool = True
    sound: Sound = Sound.none
    source: Path | None = None
    report_timeout: float = 900.0  # seconds to wait for the PDF reports before returning

    @classmethod
    def from_file(cls, path: str | Path) -> Experiment:
        """Read a .json config (any other suffix is parsed as TOML); relative paths in it are relative to the file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as e:
            raise ConfigError(f"{path}: {e}") from e
        return cls.from_dict(raw, base=path.parent, source=path)

    @classmethod
    def from_dict(cls, raw: dict, base: Path = Path("."), source: Path | None = None) -> Experiment:
        """Build and validate an experiment from a parsed config.

        `base` is the folder relative data and output paths are resolved against; `source` is
        the config file, copied into each run folder (None = nothing to copy)."""
        unknown = set(raw) - {"data", "run", "preprocessing", "feature_selection", "search", "models",
                              "hyper_parameters", "output"}
        if unknown:
            raise ConfigError(f"unknown section(s): {sorted(unknown)}")
        data, run, out = raw.get("data", {}), raw.get("run", {}), raw.get("output", {})
        for key in ("path", "target", "task"):
            if key not in data:
                raise ConfigError(f"[data] needs '{key}'")
        selection, search = raw.get("feature_selection", {}), raw.get("search", {})
        unknown = set(search) - {"strategy", "n_iter", "factor", "min_rows"}
        if unknown:
            raise ConfigError(f"[search] unknown keys {sorted(unknown)}")
        exp = cls(
            data_path=base / data["path"],
            target=data["target"],
            task=data["task"],
            features=data.get("features"),
            validation=run.get("validation", "k_folds"),
            n_splits=run.get("n_splits", 5),
            train_ratio=run.get("train_ratio", 0.8),
            select_by=run.get("select_by"),
            random_state=run.get("random_state"),
            processes=run.get("processes"),
            preprocessing=raw.get("preprocessing", []),
            selection_method=selection.get("method"),
            selection_k=selection.get("k", []),
            selection_max_features=selection.get("max_features"),
            selection_min_features=selection.get("min_features", 1),
            selection_tol=selection.get("tol", 0.0),
            search_strategy=search.get("strategy", "grid"),
            search_n_iter=search.get("n_iter"),
            search_factor=search.get("factor", 3),
            search_min_rows=search.get("min_rows"),
            models=raw.get("models", []),
            hyper_parameters=raw.get("hyper_parameters", {}),
            output_dir=base / out.get("dir", "runs"),
            keep=out.get("keep", 1),
            report=out.get("report", True),
            sound=_read_sound(out),
            source=source,
        )
        exp.validate()
        return exp

    def validate(self) -> None:
        """Raise ConfigError for the first invalid setting; also turns `sound` into a Sound.

        Every model is built here too, so a bad model spec fails now, not after loading the data."""
        if self.task not in TASKS:
            raise ConfigError(f"task must be one of {list(TASKS)}, not {self.task!r}")
        if self.validation not in VALIDATIONS:
            raise ConfigError(f"validation must be one of {list(VALIDATIONS)}, not {self.validation!r}")
        criteria = TASKS[self.task][1]
        if self.select_by is not None and self.select_by not in criteria.__members__:
            raise ConfigError(f"select_by for {self.task} must be one of {list(criteria.__members__)}")
        if not self.models:
            raise ConfigError("add at least one [[models]] entry")
        for step in self.preprocessing:
            name = step if isinstance(step, str) else next(iter(step), None)
            if name not in PREPROCESSING or not (isinstance(step, str) or len(step) == 1):
                raise ConfigError(f"preprocessing steps are {list(PREPROCESSING)}, got {step!r}")
        if self.search_strategy not in STRATEGIES:
            raise ConfigError(f"[search] strategy must be one of {list(STRATEGIES)}, not {self.search_strategy!r}")
        if self.search_strategy == "random" and self.search_n_iter is None:
            raise ConfigError("[search] random needs n_iter (how many configurations to try)")
        if self.search_n_iter is not None and (not isinstance(self.search_n_iter, int) or self.search_n_iter < 1):
            raise ConfigError("[search] n_iter must be a whole number >= 1")
        if not isinstance(self.search_factor, (int, float)) or self.search_factor <= 1:
            raise ConfigError("[search] factor must be a number > 1")
        if self.search_min_rows is not None and (not isinstance(self.search_min_rows, int)
                                                 or self.search_min_rows < 2):
            raise ConfigError("[search] min_rows must be a whole number >= 2")
        if self.search_strategy == "halving" and self.validation == "leave_one_out":
            raise ConfigError("[search] halving needs k_folds or train_test_split validation")
        if self.selection_method in DIRECTIONS and self.search_strategy != "grid":
            raise ConfigError(f"feature_selection {self.selection_method!r} already searches step by step; "
                              f"use it with the default grid search")
        if self.selection_method not in (None, "brute_force", *DIRECTIONS):
            if not callable(getattr(sklearn.feature_selection, self.selection_method, None)):
                raise ConfigError(f"feature_selection method {self.selection_method!r} is not in "
                                  f"sklearn.feature_selection")
            if not self.selection_k:
                raise ConfigError("feature_selection needs k, e.g. k = [2, 4]")
        try:
            self.sound = Sound(self.sound)
        except ValueError:
            raise ConfigError(f"[output] sound must be one of {[s.value for s in Sound]}, not {self.sound!r}") from None
        if not isinstance(self.keep, int) or self.keep < 1:
            raise ConfigError("[output] keep must be a whole number >= 1")
        for name, values in self.hyper_parameters.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"hyper_parameters.{name} must be a non-empty list")
        for spec in self.models:
            build_model(spec)

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """The feature columns and the target, dropping rows with no target.

        Raises ConfigError for missing columns, non-numeric features or missing feature values."""
        df = read_table(self.data_path)
        if self.target not in df.columns:
            raise ConfigError(f"target column {self.target!r} is not in {self.data_path.name} "
                              f"(columns: {list(df.columns)})")
        columns = self.features or [c for c in df.columns if c != self.target]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ConfigError(f"feature columns not in the data: {missing}")
        df = df.dropna(subset=[self.target])
        X = df[columns]
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ConfigError(f"non-numeric feature columns (encode or drop them): {non_numeric}")
        if X.isna().any().any():
            raise ConfigError(f"missing values in: {list(X.columns[X.isna().any()])} (fill or drop them)")
        return X, df[self.target]

    def run(self, output_dir: Path | None = None, log: Callable[[str], None] = print) -> Path:
        """Train every configuration, then write leaderboard.csv, summary.csv and the best model(s) to a new folder."""
        task_cls, criteria, default_metric = TASKS[self.task]
        metric = criteria[self.select_by or default_metric]
        X, y = self.load_data()
        log(f"{len(X)} rows, {X.shape[1]} features, target {self.target!r} ({self.task})")
        folder = (output_dir or self.output_dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder.mkdir(parents=True, exist_ok=False)
        if self.source is not None:
            shutil.copy(self.source, folder / f"config{self.source.suffix}")

        previous_limit = task_cls._process_limit
        if self.processes:
            task_cls.process_limit(self.processes)
        try:
            return self._train(task_cls, metric, X, y, folder, log)
        finally:
            BaseML._process_limit = previous_limit

    def _train(self, task_cls: type[BaseML],
               metric: FilteringCriteriaClassification | FilteringCriteriaRegression,
               X: pd.DataFrame, y: pd.Series, folder: Path, log: Callable[[str], None]) -> Path:
        """Train, rank and report with a `task_cls` run, then save the best `keep` models into `folder`.

        The configured sound plays as the training block ends, before the results are logged.
        Returns `folder`."""
        with task_cls(sound=self.sound) as run:
            run.random_state = self.random_state
            run.set(X, y, [build_model(spec) for spec in self.models])
            for step in self.preprocessing:
                name, kwargs = (step, {}) if isinstance(step, str) else next(iter(step.items()))
                getattr(run.features, name)(**kwargs)
            if self.selection_method == "brute_force":
                run.brute_force_features()
            elif self.selection_method == "forward":
                run.forward_selection(self.selection_max_features, self.selection_tol)
            elif self.selection_method == "backward":
                run.backward_elimination(self.selection_min_features, self.selection_tol)
            elif self.selection_method:
                run.feature_selection(getattr(sklearn.feature_selection, self.selection_method), self.selection_k)
            run.set_hyper_parameters(self.hyper_parameters)
            run.search(self.search_strategy, n_iter=self.search_n_iter, factor=self.search_factor,
                       min_rows=self.search_min_rows, select_by=metric)

            if self.validation == "train_test_split":
                results = run.train_test_split(self.train_ratio)
            elif self.validation == "leave_one_out":
                results = run.leave_one_out()
            else:
                results = run.k_folds(self.n_splits)

            ranked = rank(results, metric, run.features.data.shape[1])
            report = ExperimentReport(self.task, results, metric, run.features, self.hyper_parameters,
                                      class_names=run.targets.classes, search_log=run.search_log)
            board = report.leaderboard
            report.save_csv(folder)
            best = run.refit(ranked[:self.keep])
            search_log = run.search_log

        # Reports are rendered on a background thread while the models are saved, and every
        # report job is isolated: a plotting or PDF failure is logged, never raised, so it
        # cannot cost the run its leaderboard or its saved models.
        reports = ReportWorker(log) if self.report else None
        if reports is not None:
            reports.submit(report.save_pdf_report, folder, what="experiment PDF report")
        for step in search_log:
            log("  " + ", ".join(f"{k} {v}" for k, v in step.items()))
        log(f"{len(results)} configurations -> {folder / 'leaderboard.csv'}, {folder / 'summary.csv'}")
        log(report.summary_table())
        log(board.head(10).to_string())
        for i, result in enumerate(best, start=1):
            place = folder / f"{i}_{result.name}"
            place.mkdir()
            model_path = result.to_model().save(place / "model.joblib")
            if reports is not None:
                reports.submit(result.save_pdf_report, str(place), what=f"PDF report of #{i} {result.name}")
            log(f"#{i}: {result.name} {metric.value}={getattr(result, metric.value):.4f} -> {model_path}")
        if reports is not None:
            for path, error in reports.wait(self.report_timeout):
                if error is None and path is not None:
                    log(f"report -> {path}")
            viz = report.last_visualizations
            if viz is not None and (viz.failed or viz.skipped):
                log(f"experiment report plots: {viz.summary()} (see the report's last section)")
        return folder


def _read_sound(out: dict) -> Sound | str:
    """[output] sound = "<name>"; the older `ping = true/false` still works.

    The sound name is returned as written; Experiment.validate() checks it."""
    if "ping" in out:
        if "sound" in out:
            raise ConfigError("[output] use either sound or ping, not both")
        if not isinstance(out["ping"], bool):
            raise ConfigError("[output] ping must be true or false (or use sound = \"ping\")")
        return Sound.ping if out["ping"] else Sound.none
    return out.get("sound", Sound.none)


def build_model(spec: dict) -> BaseEstimator:
    """An unfitted estimator from a [[models]] entry: {"name": ..., "params": {...}}."""
    if not isinstance(spec, dict) or "name" not in spec:
        raise ConfigError(f"each [[models]] entry needs a name, got {spec!r}")
    unknown = set(spec) - {"name", "params"}
    if unknown:
        raise ConfigError(f"model {spec['name']}: unknown keys {sorted(unknown)} (put parameters under params)")
    cls = find_estimator(spec["name"])
    try:
        return cls(**spec.get("params", {}))
    except TypeError as e:
        raise ConfigError(f"model {spec['name']}: {e}") from e


def find_estimator(name: str) -> type[BaseEstimator]:
    """The estimator class for a scikit-learn class name or a full import path such as "xgboost.XGBClassifier"."""
    if "." in name:
        module, _, attr = name.rpartition(".")
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as e:
            raise ConfigError(f"cannot import {name}: {e}") from e
    from sklearn.utils.discovery import all_estimators
    found = dict(all_estimators())
    if name not in found:
        close = difflib.get_close_matches(name, found, n=1, cutoff=0.8)
        hint = f" Did you mean {close[0]}?" if close else ""
        raise ConfigError(f"{name!r} is not a scikit-learn estimator.{hint} For other libraries use the full "
                          f"import path, e.g. xgboost.XGBClassifier")
    return found[name]


def read_table(path: Path) -> pd.DataFrame:
    """Read a .csv, .parquet, .xlsx / .xls or .json table (chosen by the file suffix)."""
    if not path.exists():
        raise ConfigError(f"data file not found: {path}")
    readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".xlsx": pd.read_excel, ".xls": pd.read_excel,
               ".json": pd.read_json}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ConfigError(f"unsupported data file type {path.suffix!r} (use {', '.join(readers)})")
    return reader(path)


def rank(results: list[BaseResults], metric: FilteringCriteriaClassification | FilteringCriteriaRegression,
         n_columns: int) -> list[BaseResults]:
    """Best first; ties go to the configuration with fewer features (the simpler model)."""
    return _rank(results, metric, n_columns)


def leaderboard(ranked: list[BaseResults], grid: dict[str, list], features: Features) -> pd.DataFrame:
    """One row per configuration, in the given order; the index is the rank (1 = best)."""
    return _leaderboard(ranked, grid, features)
