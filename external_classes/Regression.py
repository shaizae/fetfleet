from sklearn.model_selection import KFold

from external_classes.BaseML import BaseML
from results.RegressionResults import RegressionResults
from utils.utils import FilteringCriteriaRegression


class Regression(BaseML):
    results_type = RegressionResults
    criteria_type = FilteringCriteriaRegression
    _encode_target = False  # keep target values numeric; label-encoding turned them into ranks

    def _make_splitter(self, n_splits: int, shuffle: bool, seed):
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
