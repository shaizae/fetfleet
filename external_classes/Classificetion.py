import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from external_classes.BaseML import BaseML
from results.ClassificationResults import ClassificationResults
from utils.utils import FilteringCriteriaClassification


class Classification(BaseML):
    results_type = ClassificationResults
    criteria_type = FilteringCriteriaClassification
    _encode_target = True

    def _stratify_labels(self, y: np.ndarray):
        return y

    def _make_splitter(self, n_splits: int, shuffle: bool, seed):
        """Stratified folds keep each class's share the same in every fold. They need
        at least n_splits rows per class; otherwise fall back to plain KFold."""
        smallest_class = np.bincount(np.asarray(self.targets.data, dtype=np.int64)).min()
        if smallest_class >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        warnings.warn(f"smallest class has {smallest_class} rows (< {n_splits} folds): using KFold, "
                      f"not StratifiedKFold", stacklevel=3)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
