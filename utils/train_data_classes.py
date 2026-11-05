from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from results import BaseResults
from utils.utils import SharedMemory, ValidationType


class SelectColumns(TransformerMixin, BaseEstimator):
    """Keeps the given columns. Used after preprocessing, because feature indexes refer
    to the transformed matrix (e.g. the columns PolynomialFeatures creates)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        return X if self.columns is None else np.asarray(X)[:, self.columns]


def build_estimator(model, preprocessing: list | None, features_index=None):
    """The estimator to fit on *raw* rows: preprocessing -> column selection -> model.

    Every step is fitted on the training rows only, so test rows never leak into the
    scaling. Without preprocessing or column selection this is just the model.
    """
    steps = [(f"pre{i}", clone(step)) for i, step in enumerate(preprocessing or [])]
    if features_index is not None:
        steps.append(("select", SelectColumns(np.asarray(features_index))))
    if not steps:
        return model
    return Pipeline(steps + [("model", model)])


@dataclass(slots=True)
class TrainData:
    model: Any
    results_type: BaseResults
    features: SharedMemory = None
    target: SharedMemory = None
    train_index: list[int] = None
    test_index: list[int] = None
    featuresIndex: list[int] = None
    preprocessing: list = None  # unfitted transformers, refitted inside every split

    @property
    def name(self):
        return self.model.__class__.__name__

    def get_model_params(self) -> dict[str, Any]:
        if hasattr(self.model, "get_params"):
            return self.model.get_params(deep=False)
        return {k: v for k, v in self.model.__dict__.items() if not k.startswith("_")}

    def copy(self):
        # Fresh unfitted model with the same parameters; everything else (shared memory,
        # index lists) is shared, never mutated in place, so a shallow copy is enough.
        return replace(self, model=clone(self.model, safe=False))

    def set_features_and_targets(self, features: SharedMemory, target: SharedMemory):
        self.features = features
        self.target = target

    def set_indexes(self, train_index: list[int], test_index: list[int]):
        self.train_index = train_index
        self.test_index = test_index

    def set_features_selection(self, indexes: list[int]):
        self.featuresIndex = indexes

    def fit(self, train_features: np.ndarray, train_target: np.ndarray):
        self.model.fit(train_features, train_target)
        return self.model

    def cost(self) -> float:
        """Rough relative cost, used to start the most expensive tasks first."""
        params = self.get_model_params()
        trees = params.get("n_estimators") or 1
        n_features = len(self.featuresIndex) if self.featuresIndex is not None else 1
        return float(trees) * max(1, n_features)

    def __iter__(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)


@dataclass(slots=True)
class KFoldsTrainData(TrainData):
    number_of_folds: int = 0
    shuffle: bool = True
    randon_state: int = None
    # fold id of every row, computed once in the parent so every task uses the same
    # folds. None -> the worker builds a KFold itself (direct calls, older code).
    folds: SharedMemory = None
    validation: ValidationType = None

    @staticmethod
    def set_from_train_data(train_data: TrainData, number_of_folds: int, shuffle: bool,
                            randon_state: int, folds: SharedMemory = None,
                            validation: ValidationType = None) -> KFoldsTrainData:
        return KFoldsTrainData(**dict(train_data), number_of_folds=number_of_folds, shuffle=shuffle,
                               randon_state=randon_state, folds=folds, validation=validation)

    def cost(self) -> float:
        return TrainData.cost(self) * max(1, self.number_of_folds)
