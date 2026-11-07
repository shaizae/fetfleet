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
from typing import Any

import pandas as pd
import sklearn.feature_selection

from ml_pipe.base import BaseML
from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression

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
# method = "f_classif"       # a sklearn.feature_selection score function, or "brute_force"
# k = [2, 4, 8]              # numbers of features to try (not used by brute_force)

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
report = true                # PDF report for each saved model
'''


class ConfigError(ValueError):
    pass


@dataclass
class Experiment:
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
    models: list[dict] = field(default_factory=list)
    hyper_parameters: dict[str, list] = field(default_factory=dict)
    output_dir: Path = Path("runs")
    keep: int = 1
    report: bool = True
    source: Path | None = None

    # ------------------------------------------------------------------ reading
    @classmethod
    def from_file(cls, path: str | Path) -> Experiment:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as e:
            raise ConfigError(f"{path}: {e}") from e
        return cls.from_dict(raw, base=path.parent, source=path)

    @classmethod
    def from_dict(cls, raw: dict, base: Path = Path("."), source: Path | None = None) -> Experiment:
        unknown = set(raw) - {"data", "run", "preprocessing", "feature_selection", "models", "hyper_parameters",
                              "output"}
        if unknown:
            raise ConfigError(f"unknown section(s): {sorted(unknown)}")
        data, run, out = raw.get("data", {}), raw.get("run", {}), raw.get("output", {})
        for key in ("path", "target", "task"):
            if key not in data:
                raise ConfigError(f"[data] needs '{key}'")
        selection = raw.get("feature_selection", {})
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
            models=raw.get("models", []),
            hyper_parameters=raw.get("hyper_parameters", {}),
            output_dir=base / out.get("dir", "runs"),
            keep=out.get("keep", 1),
            report=out.get("report", True),
            source=source,
        )
        exp.validate()
        return exp

    def validate(self):
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
        if self.selection_method not in (None, "brute_force"):
            if not callable(getattr(sklearn.feature_selection, self.selection_method, None)):
                raise ConfigError(f"feature_selection method {self.selection_method!r} is not in "
                                  f"sklearn.feature_selection")
            if not self.selection_k:
                raise ConfigError("feature_selection needs k, e.g. k = [2, 4]")
        if not isinstance(self.keep, int) or self.keep < 1:
            raise ConfigError("[output] keep must be a whole number >= 1")
        for name, values in self.hyper_parameters.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"hyper_parameters.{name} must be a non-empty list")
        for spec in self.models:
            build_model(spec)  # fail now, not after loading the data

    # ------------------------------------------------------------------ running
    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
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

    def run(self, output_dir: Path | None = None, log=print) -> Path:
        """Train every configuration, then write leaderboard.csv and the best model(s) to a new folder."""
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

    def _train(self, task_cls, metric, X, y, folder: Path, log) -> Path:
        with task_cls() as run:
            run.random_state = self.random_state
            run.set(X, y, [build_model(spec) for spec in self.models])
            for step in self.preprocessing:
                name, kwargs = (step, {}) if isinstance(step, str) else next(iter(step.items()))
                getattr(run.features, name)(**kwargs)
            if self.selection_method == "brute_force":
                run.brute_force_features()
            elif self.selection_method:
                run.feature_selection(getattr(sklearn.feature_selection, self.selection_method), self.selection_k)
            run.set_hyper_parameters(self.hyper_parameters)

            if self.validation == "train_test_split":
                results = run.train_test_split(self.train_ratio)
            elif self.validation == "leave_one_out":
                results = run.leave_one_out()
            else:
                results = run.k_folds(self.n_splits)

            ranked = rank(results, metric, run.features.data.shape[1])
            board = leaderboard(ranked, self.hyper_parameters, run.features)
            board.to_csv(folder / "leaderboard.csv", index_label="rank")
            best = run.refit(ranked[:self.keep])

        log(f"{len(results)} configurations -> {folder / 'leaderboard.csv'}")
        log(board.head(10).to_string())
        for i, result in enumerate(best, start=1):
            place = folder / f"{i}_{result.name}"
            place.mkdir()
            model_path = result.to_model().save(place / "model.joblib")
            if self.report:
                result.save_pdf_report(str(place))
            log(f"#{i}: {result.name} {metric.value}={getattr(result, metric.value):.4f} -> {model_path}")
        return folder


def build_model(spec: dict):
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


def find_estimator(name: str):
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
    if not path.exists():
        raise ConfigError(f"data file not found: {path}")
    readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".xlsx": pd.read_excel, ".xls": pd.read_excel,
               ".json": pd.read_json}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ConfigError(f"unsupported data file type {path.suffix!r} (use {', '.join(readers)})")
    return reader(path)


def rank(results, metric, n_columns: int) -> list:
    """Best first; ties go to the configuration with fewer features (the simpler model)."""
    sign = -1 if metric.maximize else 1

    def key(r):
        n = len(r.features_index) if r.features_index is not None else n_columns
        return sign * getattr(r, metric.value), n

    return sorted(results, key=key)


def leaderboard(ranked, grid: dict[str, list], features) -> pd.DataFrame:
    """One row per configuration, in the given order; the index is the rank (1 = best)."""
    names = list(features.columns) if features.columns is not None else None
    same_columns = names is not None and features.data.shape[1] == len(names)  # no polynomial expansion
    rows = []
    for r in ranked:
        index = r.features_index
        if index is None:
            used = "all"
        elif same_columns:
            used = ", ".join(str(names[i]) for i in index)
        else:
            used = ", ".join(f"x{i}" for i in index)
        params = {k: v for k, v in r.params.items() if k in grid}
        rows.append({"model": r.name, "params": json.dumps(params, default=str) if params else "",
                     "n_features": len(index) if index is not None else features.data.shape[1],
                     "features": used, "validation": str(r.validation), **r.metrics})
    board = pd.DataFrame(rows)
    board.index = range(1, len(board) + 1)
    return board
