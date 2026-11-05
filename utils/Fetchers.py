
import numpy as np
import pandas as pd
from pandas import Index
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from tqdm import trange


class Fetchers:
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

    def load_new_data(self, data: pd.DataFrame):
        self._data = data.values
        self._raw = self._data
        self._steps = []
        self._columns = data.columns

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
        self._data = step.fit_transform(self._data)
        self._steps.append(clone(step))
        return self.data

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
        for row in trange(self._data.shape[0]):
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
