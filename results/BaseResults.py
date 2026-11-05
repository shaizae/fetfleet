from abc import ABC, abstractmethod
import datetime
import os

import numpy as np
from joblib import dump

from utils.utils import ValidationType, create_path, is_fitted


class BaseResults(ABC):
    def __init__(self, model):
        self.model = model
        self._features_test = None
        self._target_test = None
        self._pred_test = None
        self._validation = ""
        self.features_index = None  # column indexes this run was trained on

    @property
    def name(self):
        model = self.model
        if hasattr(model, "steps"):  # sklearn Pipeline: name it after the final estimator
            model = model.steps[-1][1]
        return model.__class__.__name__

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, validation):
        if validation not in ValidationType:
            raise ValueError(f"{validation} is not valid")
        self._validation = validation

    @property
    def features_test(self):
        if self._features_test is None:
            return None
        return self._features_test


    @property
    def target_test(self):
        if self._target_test is None:
            return None
        return self._target_test

    def set_test(self, features_test, target_test):
        self._features_test = features_test
        self._target_test = target_test

    def append_results(self, result: "BaseResults"):
        if result.features_test is not None:
            if self.features_test is None:
                self._features_test = result.features_test.copy()
            else:
                self._features_test = np.concatenate(
                    [self.features_test, result.features_test],
                    axis=0,
                )

        if result.target_test is not None:
            if self.target_test is None:
                self._target_test = result.target_test.copy()
            else:
                self._target_test = np.concatenate(
                    [self.target_test, result.target_test],
                    axis=0,
                )

        if result._pred_test is not None:
            if self._pred_test is None:
                self._pred_test = result._pred_test.copy()
            else:
                self._pred_test = np.concatenate(
                    [self._pred_test, result._pred_test],
                    axis=0,
                )

        self.model = result.model
        return self

    def save_model(self, path: str):
        if not is_fitted(self.model):
            raise ValueError(f"{self.name} is not fitted: results from a run hold only the model's "
                             f"parameters. Use filter_by() or refit() on the pipeline before saving.")
        filename = os.path.join(
            path,
            f"model_{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.joblib",
        )
        dump(self.model, filename)

    def save_results(self, path: str):
        name = f"{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        folder = os.path.join(path, name)

        create_path(folder)

        self.save_pdf_report(folder)
        self.save_model(folder)

    @abstractmethod
    def predict(self):
        """Predict features_test with self.model, then evaluate()."""
        ...

    @abstractmethod
    def evaluate(self, y_true, y_pred, proba=None):
        """Store out-of-sample predictions and compute the metrics from them."""
        ...

    @abstractmethod
    def report_lines(self):
        ...

    @abstractmethod
    def save_pdf_report(self, filename: str):
        ...

    @abstractmethod
    def __str__(self):
        ...