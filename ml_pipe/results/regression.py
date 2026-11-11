from __future__ import annotations

import datetime
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from ml_pipe.results.base import BaseResults

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class RegressionResults(BaseResults):
    """Results of a regressor: R², MAE, MSE and RMSE."""

    def __init__(self, model: BaseEstimator) -> None:
        """Wrap the regressor `model`; all metrics start at 0."""
        super().__init__(model)
        self._r2 = 0.0
        self._mae = 0.0
        self._mse = 0.0
        self._rmse = 0.0

    @property
    def metrics(self) -> dict[str, float]:
        """R², MAE, MSE and RMSE, by name."""
        return {"r2": self.r2, "mae": self.mae, "mse": self.mse, "rmse": self.rmse}

    @property
    def r2(self) -> float:
        """Coefficient of determination."""
        return self._r2

    @property
    def mae(self) -> float:
        """Mean absolute error."""
        return self._mae

    @property
    def mse(self) -> float:
        """Mean squared error."""
        return self._mse

    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        return self._rmse

    def predict(self) -> None:
        """Predict features_test with self.model, then evaluate()."""
        self.evaluate(self.target_test, self.model.predict(self.features_test))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> None:
        """Store the predictions, compute the metrics and drop features_test; `proba` is ignored."""
        self._target_test = y_true
        self._pred_test = y_pred
        self._r2 = r2_score(y_true, y_pred)
        self._mae = mean_absolute_error(y_true, y_pred)
        self._mse = mean_squared_error(y_true, y_pred)
        self._rmse = np.sqrt(self._mse)

        self._features_test = None

    def plot_prediction(self, show: bool = True) -> None:
        """Scatter predicted against actual values, with the identity line for reference."""
        plt.figure(figsize=(6, 6))

        plt.scatter(self.target_test, self._pred_test, alpha=0.7)

        minimum = min(self.target_test.min(), self._pred_test.min())
        maximum = max(self.target_test.max(), self._pred_test.max())

        plt.plot([minimum, maximum], [minimum, maximum], "r--")

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs Actual - {self.name}")

        if show:
            plt.show()

    def plot_residuals(self, show: bool = True) -> None:
        """Scatter the residuals (actual - predicted) against the predictions."""
        residuals = self.target_test - self._pred_test

        plt.figure(figsize=(6, 4))
        plt.scatter(self._pred_test, residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot - {self.name}")

        if show:
            plt.show()

    def report_lines(self) -> Iterator[str]:
        """Yield all report lines."""
        yield f"Regression Report - {self.name}"
        yield ""

        yield f"R² Score : {self.r2:.4f}"
        yield f"MAE      : {self.mae:.4f}"
        yield f"MSE      : {self.mse:.4f}"
        yield f"RMSE     : {self.rmse:.4f}"
        yield f"Validation type: {self.validation}"

    def report_matrix(self) -> str:
        """The metrics as a two-column text table."""
        return (
            f"{'Metric':<20}{'Value'}\n"
            f"{'-' * 35}\n"
            f"{'R²':<20}{self.r2:.6f}\n"
            f"{'MAE':<20}{self.mae:.6f}\n"
            f"{'MSE':<20}{self.mse:.6f}\n"
            f"{'RMSE':<20}{self.rmse:.6f}"
        )

    def save_pdf_report(self, filename: str) -> Path:
        """Write a timestamped PDF report into the folder `filename` and return its path.

        Metrics, then every applicable plot (actual vs predicted, residual diagnostics, error by
        target range, feature importance when the model is fitted), each generated in isolation:
        a plot that fails or does not apply is listed in the appendix instead of stopping the report."""
        if self._pred_test is None:
            raise ValueError("No predictions available")
        path = os.path.join(filename, f"report_{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pdf")
        return self._build_pdf_report(path, f"Regression Report - {self.name}")

    def __str__(self) -> str:
        """The report lines followed by the metrics table."""
        return (
                "\n".join(self.report_lines())
                + "\n\n"
                + self.report_matrix()
        )
