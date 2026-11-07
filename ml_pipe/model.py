"""A trained model packaged for use on new data: save it, load it, predict."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier


@dataclass
class SavedModel:
    """A fitted estimator plus what is needed to use it on new rows.

    `estimator` takes raw rows: when the run used scaling, polynomial features or a feature
    subset, it is a scikit-learn Pipeline that applies them itself. For classification,
    predictions are decoded back to the original labels.
    """
    estimator: Any
    task: str  # "classification" | "regression"
    feature_columns: list[str] | None = None  # columns the estimator expects, in order
    class_names: list | None = None  # original labels; index i = encoded class i
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    validation: str = ""
    created: str = field(default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds"))
    ml_pipe_version: str = ""

    @property
    def name(self) -> str:
        model = self.estimator
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        return type(model).__name__

    # ------------------------------------------------------------------ predicting
    def _matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_columns is None:
                return X.to_numpy()
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise KeyError(f"columns missing from the data: {missing}")
            return X[self.feature_columns].to_numpy()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_columns is not None and X.shape[1] != len(self.feature_columns):
            raise ValueError(f"expected {len(self.feature_columns)} columns {self.feature_columns}, got {X.shape[1]}")
        return X

    def predict(self, X) -> np.ndarray:
        """Predictions for a DataFrame (columns picked by name, extra columns ignored) or an array."""
        pred = self.estimator.predict(self._matrix(X))
        if self.class_names is not None:
            return np.asarray(self.class_names)[np.asarray(pred, dtype=np.int64)]
        return pred

    def predict_proba(self, X) -> pd.DataFrame:
        """Class probabilities, one column per class (classification only)."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.name} has no predict_proba (for SVC, train it with probability=True)")
        proba = self.estimator.predict_proba(self._matrix(X))
        classes = getattr(self.estimator, "classes_", np.arange(proba.shape[1]))
        if self.class_names is not None:
            classes = np.asarray(self.class_names)[np.asarray(classes, dtype=np.int64)]
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(proba, columns=list(classes), index=index)

    # ------------------------------------------------------------------ files
    def save(self, path: str | Path) -> Path:
        """Write to `path` (a .joblib file, or a folder to create a named file in). Returns the file."""
        path = Path(path)
        if path.suffix != ".joblib":
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"model_{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.joblib"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    def __str__(self):
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"{self.name} ({self.task}, {self.validation}) {metrics}"


def load(path: str | Path) -> SavedModel:
    """Load a model saved by SavedModel.save() / result.save_model().

    Only load files you trust: joblib files can run code when loaded."""
    obj = joblib.load(path)
    if isinstance(obj, SavedModel):
        return obj
    # a bare estimator (files saved before ml_pipe 0.2)
    return SavedModel(estimator=obj, task="classification" if is_classifier(obj) else "regression")
