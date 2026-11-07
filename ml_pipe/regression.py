from sklearn.model_selection import KFold

from ml_pipe.base import BaseML
from ml_pipe.results.regression import RegressionResults
from ml_pipe.utils import FilteringCriteriaRegression


class Regression(BaseML):
    results_type = RegressionResults
    criteria_type = FilteringCriteriaRegression
    default_criteria = FilteringCriteriaRegression.r2
    _encode_target = False  # keep target values numeric; label-encoding turned them into ranks

    def _make_splitter(self, n_splits: int, shuffle: bool, seed, y=None):
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
