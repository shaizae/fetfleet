from __future__ import annotations

from abc import ABC, abstractmethod
import datetime
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from ml_pipe.model import SavedModel
from ml_pipe.utils import ValidationType, create_path, is_fitted

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class BaseResults(ABC):
    """Out-of-sample predictions and metrics of one model configuration.

    Besides the model, a result records `features_index` (the column indexes this run
    was trained on), `feature_columns` (the names of the raw input columns) and
    `class_names` (the original class labels, classification only), the last two set
    by refit(), and `candidate_id` (which task of its batch produced this result).
    Workers also record `fit_time` (seconds to fit and predict every split of this
    configuration, on one worker core) and, for k-fold runs, `fold_sizes` (rows per
    fold, in the order the pooled predictions are stored); both stay None otherwise.
    """

    def __init__(self, model: BaseEstimator) -> None:
        """Wrap `model` (an estimator or Pipeline) with no test data or predictions yet."""
        self.model = model
        self._features_test = None
        self._target_test = None
        self._pred_test = None
        self._validation = ""
        self.features_index = None
        self.feature_columns = None
        self.class_names = None
        self.candidate_id = None
        self.fit_time: float | None = None
        self.fold_sizes: np.ndarray | None = None

    @property
    def name(self) -> str:
        """The model's class name; for an sklearn Pipeline, that of its final estimator."""
        model = self.model
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        return model.__class__.__name__

    @property
    def validation(self) -> ValidationType | str:
        """How the result was validated ("" until set)."""
        return self._validation

    @validation.setter
    def validation(self, validation: ValidationType | str) -> None:
        """Set the validation type; raises ValueError if it is not a ValidationType value."""
        if validation not in ValidationType:
            raise ValueError(f"{validation} is not valid")
        self._validation = validation

    @property
    def features_test(self) -> np.ndarray | None:
        """The test feature rows, or None when not set (evaluate() clears them)."""
        if self._features_test is None:
            return None
        return self._features_test


    @property
    def target_test(self) -> np.ndarray | None:
        """The true targets of the test rows, or None when not set."""
        if self._target_test is None:
            return None
        return self._target_test

    def set_test(self, features_test: np.ndarray, target_test: np.ndarray) -> None:
        """Store the test features and targets for predict()."""
        self._features_test = features_test
        self._target_test = target_test

    def append_results(self, result: BaseResults) -> Self:
        """Concatenate `result`'s test features, targets and predictions onto this one's.

        Used to pool the folds of one configuration; the model is replaced by `result`'s.
        Returns self.
        """
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

    @property
    def params(self) -> dict:
        """The model's own parameters (of the final step when the model is a Pipeline)."""
        model = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        return model.get_params(deep=False) if hasattr(model, "get_params") else {}

    @property
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Every metric of this result, by name."""
        ...

    def to_model(self) -> SavedModel:
        """The fitted model with its column names, labels and scores, ready to save or predict."""
        if not is_fitted(self.model):
            raise ValueError(f"{self.name} is not fitted: results from a run hold only the model's "
                             f"parameters. Use filter_by() or refit() on the pipeline before saving.")
        from ml_pipe import __version__
        return SavedModel(
            estimator=self.model,
            task="classification" if getattr(self, "is_classification", False) else "regression",
            feature_columns=None if self.feature_columns is None else list(self.feature_columns),
            class_names=None if self.class_names is None else list(self.class_names),
            metrics={k: float(v) for k, v in self.metrics.items()},
            params={k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool, type(None)))},
            validation=str(self.validation),
            ml_pipe_version=__version__,
        )

    def save_model(self, path: str) -> Path:
        """Save to a new file in the folder `path`; load it with ml_pipe.load(file)."""
        return self.to_model().save(path)

    def save_results(self, path: str) -> None:
        """Save the model and the PDF report into a new timestamped subfolder of `path`.

        The model is saved first; a failure while writing the report is logged and does
        not affect the saved model."""
        from ml_pipe.visualization.core import safe_call
        name = f"{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        folder = os.path.join(path, name)

        create_path(folder)

        self.save_model(folder)
        safe_call(self.save_pdf_report, folder, what=f"PDF report of {self.name}")

    def visualize(self):
        """Every plot that applies to this result, each generated in isolation.

        Returns a VisualizationReport (one SUCCESS / SKIPPED / FAILED record per plot); never raises."""
        from ml_pipe.visualization import VisualizationContext, VisualizationManager
        return VisualizationManager().run_for(lambda: VisualizationContext.from_result(self))

    def _build_pdf_report(self, filename: str, title: str) -> Path:
        """The per-model PDF: metrics, then every applicable plot, then the visualization appendix."""
        from ml_pipe.visualization.pdf import PdfReport
        visualizations = self.visualize()
        self.last_visualizations = visualizations.without_images()  # statuses only; images go to the PDF
        pdf = PdfReport(title, f"Validation: {self.validation or 'n/a'}  ·  generated by ml_pipe",
                        landscape_pages=False)
        pdf.key_values("Executive Summary", [("model", self.name), ("validation", str(self.validation) or "n/a")]
                       + [(k, f"{v:.4f}") for k, v in self.metrics.items()]
                       + ([("fit + predict time", f"{self.fit_time:.3f} s")] if self.fit_time is not None else []))
        params = {k: v for k, v in self.params.items()}
        if params:
            pdf.key_values("Experiment Configuration", [(k, repr(v)) for k, v in sorted(params.items())])
        self._report_details(pdf)
        return pdf.build(filename, visualizations)

    def _report_details(self, pdf) -> None:
        """Hook: add task-specific text to the per-model PDF."""

    @abstractmethod
    def predict(self) -> None:
        """Predict features_test with self.model, then evaluate()."""
        ...

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> None:
        """Store out-of-sample predictions and compute the metrics from them."""
        ...

    @abstractmethod
    def report_lines(self) -> Iterator[str]:
        """Yield the lines of the text report ("" marks a blank line)."""
        ...

    @abstractmethod
    def save_pdf_report(self, filename: str) -> None:
        """Write a PDF report into the folder `filename`."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """The text report."""
        ...
