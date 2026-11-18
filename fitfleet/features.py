
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas import Index
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from tqdm import trange

from fitfleet.encoding import allows_nan
from fitfleet.schema import ColumnSchema


class Features:
    """Feature matrix plus the preprocessing applied to it.

    The scaler / polynomial methods transform `data` right away so you can inspect
    the result, and they also record each step. Training never uses `data`: it starts
    from `raw_data` and refits the recorded steps on the training rows of every
    split, so test rows never influence the scaling (no data leakage).
    """

    def __init__(self):
        self._data: np.ndarray = None
        self._raw: np.ndarray = None
        self._steps: list = []
        self._columns: Index[str] = None
        self._schema: ColumnSchema | None = None

    def load_new_data(self, data: pd.DataFrame, schema: ColumnSchema | None = None):
        """Keep `data`. With a schema of a mixed table (categorical columns or missing values) the
        table is stored as the schema's float matrix: categories as codes, missing values as NaN."""
        self._schema = schema
        if self.mixed:
            self._raw = schema.encode(data)
            self._data = self._raw
        else:
            self._data = data.values
            self._raw = self._data
        self._steps = []
        self._columns = data.columns

    @property
    def schema(self) -> ColumnSchema | None:
        return self._schema

    @property
    def mixed(self) -> bool:
        """The table has categorical columns or missing values (see fitfleet.schema)."""
        return self._schema is not None and not self._schema.simple

    @property
    def n_selectable(self) -> int:
        """How many columns feature indexes choose from: the original columns of a mixed table,
        otherwise the columns of the (preprocessed) matrix."""
        return len(self._columns) if self.mixed else self._data.shape[1]

    def index_names(self) -> list[str] | None:
        """Column names for feature indexes, or None when the indexes refer to generated columns
        (e.g. after polynomial_features on a numeric table)."""
        if self._columns is None:
            return None
        if self.mixed or self._data.shape[1] == len(self._columns):
            return [str(c) for c in self._columns]
        return None

    def pop_index(self, index: list[int]):
        val = self._data[index, ...]
        self._data = np.delete(self._data, index, axis=0)
        if self._raw is not None:
            self._raw = np.delete(self._raw, index, axis=0)
        return val

    @property
    def raw_data(self) -> np.ndarray:
        """The features before any recorded preprocessing step."""
        return self._raw

    @property
    def preprocessing(self) -> list:
        """Unfitted copies of the recorded steps, in order."""
        return [clone(step) for step in self._steps]

    def _apply(self, step):
        if self.mixed:
            self._steps.append(clone(step))
            self._data = self._mixed_preview()
            return self.data
        self._data = step.fit_transform(self._data)
        self._steps.append(clone(step))
        return self.data

    def _mixed_preview(self) -> np.ndarray:
        """`data` of a mixed table: the recorded steps applied to the numeric columns (missing
        values filled with the column median first when a step can't take NaN), followed by the
        other matrix columns (codes, dates, text) as they are. A preview only: training refits
        everything per split."""
        number_cols = self._schema.matrix_index(self._schema.numeric_index)
        numeric = self._raw[:, number_cols]
        others = self._raw[:, [j for j in range(self._raw.shape[1]) if j not in set(number_cols)]]
        if np.isnan(numeric).any() and not all(allows_nan(s) for s in self._steps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
                medians = np.nan_to_num(np.nanmedian(numeric, axis=0))
            numeric = np.where(np.isnan(numeric), medians, numeric)
        for step in self._steps:
            numeric = clone(step).fit_transform(numeric)
        return np.hstack([numeric, others])

    @property
    def data(self):
        return self._data

    def update_data(self, new_data: np.ndarray):
        if new_data.shape != self._data.shape:  # also rejects extra dimensions
            raise ValueError("new data must have same shape as existing data")
        self._data = new_data
        self._raw = new_data  # replaced data becomes the new starting point
        self._steps = []

    @property
    def columns(self):
        return self._columns

    def standard_scaler(self, with_mean: bool = True, with_std: bool = True):
        return self._apply(StandardScaler(with_mean=with_mean, with_std=with_std))

    def min_max_scaler(self, min_val=0, max_val=1):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        return self._apply(MinMaxScaler(feature_range=(min_val, max_val)))

    def __iter__(self):
        for row in trange(self._data.shape[0], desc="feature rows"):
            yield self._data[row, ...]

    def min(self):
        return np.min(self._data)

    def max(self):
        return np.max(self._data)

    def mean(self):
        return np.mean(self._data)

    def std(self):
        return np.std(self._data)

    def median(self):
        return np.median(self._data)

    def polynomial_features(self, degree: int = 2, include_bias: bool = False):
        return self._apply(PolynomialFeatures(degree=degree, include_bias=include_bias))
