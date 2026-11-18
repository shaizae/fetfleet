from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from fitfleet.encoding import build_encoder
from fitfleet.results.base import BaseResults
from fitfleet.schema import ColumnSchema
from fitfleet.utils import SharedMemory, ValidationType


class SelectColumns(TransformerMixin, BaseEstimator):
    """Keeps the given columns (None = all). Numeric tables: used after preprocessing, because
    feature indexes refer to the transformed matrix (e.g. the columns PolynomialFeatures creates).
    Mixed tables: used first, on the encoded matrix; `names` then names the kept matrix columns
    (a text column takes several), for reports."""

    def __init__(self, columns=None, names=None):
        self.columns = columns
        self.names = names

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        return X if self.columns is None else np.asarray(X)[:, self.columns]

    def get_feature_names_out(self, input_features=None):
        if self.names is not None:
            return np.asarray(self.names, dtype=object)
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        input_features = np.asarray(input_features, dtype=object)
        return input_features if self.columns is None else input_features[np.asarray(self.columns)]


def build_estimator(model, preprocessing: list | None, features_index=None, schema: ColumnSchema | None = None,
                    encoding: str = "auto", preselected: bool = False, encoder_options: dict | None = None):
    """The estimator to fit on *raw* rows.

    Purely numeric data without gaps (schema None or simple): preprocessing -> column
    selection -> model; feature indexes refer to the columns preprocessing produces.

    Mixed data (a schema with categorical columns or missing values): column selection ->
    encoding (one-hot / codes / imputation, chosen per model, with the user's preprocessing
    on the numeric part) -> model; feature indexes refer to the original columns.
    `preselected`: the rows passed in already hold only the selected columns (workers cut
    them straight out of shared memory), so no selection step is added. `encoder_options` go to
    fitfleet.encoding.build_encoder (max_onehot, regression, random_state).

    Every step is fitted on the training rows only, so test rows never leak into the
    scaling or encoding. Without any step this is just the model.
    """
    if schema is not None and not schema.simple:
        steps = []
        if not preselected and (features_index is not None or schema.text_index):
            names = schema.matrix_names(features_index)
            matrix = None if features_index is None else np.asarray(schema.matrix_index(features_index))
            steps.append(("select", SelectColumns(matrix, names)))
        encoder, model = build_encoder(schema, features_index, model, preprocessing, encoding,
                                       **(encoder_options or {}))
        if encoder is not None:
            steps.append(("encode", encoder))
        return Pipeline(steps + [("model", model)]) if steps else model

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
    features_index: list[int] = None
    preprocessing: list = None  # unfitted transformers, refitted inside every split
    candidate_id: int = None  # position in the batch it was sent with; copied onto its result
    schema: ColumnSchema = None  # column kinds of a mixed table (None: purely numeric, nothing missing)
    encoding: str = "auto"  # categorical encoding: auto (per model) | onehot | ordinal | target
    encoder_options: dict = None  # max_onehot, regression, random_state (see fitfleet.encoding)
    encoding_searched: bool = False  # the encoding came from the hyper-parameter grid: report it

    @property
    def mixed(self) -> bool:
        """The features need encoding (categorical columns or missing values)."""
        return self.schema is not None and not self.schema.simple

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
        self.features_index = indexes

    def fit(self, train_features: np.ndarray, train_target: np.ndarray):
        self.model.fit(train_features, train_target)
        return self.model

    def cost(self) -> float:
        """Rough relative cost, used to start the most expensive tasks first."""
        params = self.get_model_params()
        trees = params.get("n_estimators") or 1
        n_features = len(self.features_index) if self.features_index is not None else 1
        return float(trees) * max(1, n_features)

    def __iter__(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)


@dataclass(slots=True)
class KFoldsTrainData(TrainData):
    number_of_folds: int = 0
    shuffle: bool = True
    random_state: int = None
    # fold id of every row, computed once in the parent so every task uses the same
    # folds. None -> the worker builds a KFold itself (direct calls, older code).
    folds: SharedMemory = None
    validation: ValidationType = None

    @staticmethod
    def set_from_train_data(train_data: TrainData, number_of_folds: int, shuffle: bool,
                            random_state: int, folds: SharedMemory = None,
                            validation: ValidationType = None) -> KFoldsTrainData:
        return KFoldsTrainData(**dict(train_data), number_of_folds=number_of_folds, shuffle=shuffle,
                               random_state=random_state, folds=folds, validation=validation)

    def cost(self) -> float:
        return TrainData.cost(self) * max(1, self.number_of_folds)
