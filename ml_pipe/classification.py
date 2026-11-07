import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from ml_pipe.base import BaseML
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.utils import FilteringCriteriaClassification


class Classification(BaseML):
    results_type = ClassificationResults
    criteria_type = FilteringCriteriaClassification
    default_criteria = FilteringCriteriaClassification.accuracy
    _encode_target = True

    def _stratify_labels(self, y: np.ndarray):
        return y

    def _make_splitter(self, n_splits: int, shuffle: bool, seed, y: np.ndarray = None):
        """Stratified folds keep each class's share the same in every fold. They need
        at least n_splits rows per class; otherwise fall back to plain KFold."""
        y = self.targets.data if y is None else y
        counts = np.bincount(np.asarray(y, dtype=np.int64))
        smallest_class = counts[counts > 0].min()
        if smallest_class >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        warnings.warn(f"smallest class has {smallest_class} rows (< {n_splits} folds): using KFold, "
                      f"not StratifiedKFold", stacklevel=3)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
