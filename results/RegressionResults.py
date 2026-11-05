import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Image
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from results.BaseResults import BaseResults


class RegressionResults(BaseResults):

    def __init__(self, model):
        super().__init__(model)
        self._r2 = 0.0
        self._mae = 0.0
        self._mse = 0.0
        self._rmse = 0.0

    @property
    def r2(self):
        return self._r2

    @property
    def mae(self):
        return self._mae

    @property
    def mse(self):
        return self._mse

    @property
    def rmse(self):
        return self._rmse

    def predict(self):
        self.evaluate(self.target_test, self.model.predict(self.features_test))

    def evaluate(self, y_true, y_pred, proba=None):
        self._target_test = y_true
        self._pred_test = y_pred
        self._r2 = r2_score(y_true, y_pred)
        self._mae = mean_absolute_error(y_true, y_pred)
        self._mse = mean_squared_error(y_true, y_pred)
        self._rmse = np.sqrt(self._mse)

        self._features_test = None

    def plot_prediction(self, show: bool = True):
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

    def plot_residuals(self, show: bool = True):
        residuals = self.target_test - self._pred_test

        plt.figure(figsize=(6, 4))
        plt.scatter(self._pred_test, residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot - {self.name}")

        if show:
            plt.show()

    def report_lines(self):
        yield f"Regression Report - {self.name}"
        yield ""

        yield f"R² Score : {self.r2:.4f}"
        yield f"MAE      : {self.mae:.4f}"
        yield f"MSE      : {self.mse:.4f}"
        yield f"RMSE     : {self.rmse:.4f}"
        yield f"Validation type: {self.validation}"

    def report_matrix(self):
        return (
            f"{'Metric':<20}{'Value'}\n"
            f"{'-' * 35}\n"
            f"{'R²':<20}{self.r2:.6f}\n"
            f"{'MAE':<20}{self.mae:.6f}\n"
            f"{'MSE':<20}{self.mse:.6f}\n"
            f"{'RMSE':<20}{self.rmse:.6f}"
        )

    def save_pdf_report(self, filename: str):
        filename = os.path.join(
            filename,
            f"report_{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pdf",
        )

        if self._pred_test is None:
            raise ValueError("No predictions available")

        styles = getSampleStyleSheet()

        doc = SimpleDocTemplate(filename)

        elements = []

        for line in self.report_lines():
            if line == "":
                elements.append(Spacer(1, 12))
            else:
                elements.append(Paragraph(line, styles["BodyText"]))

        elements.append(Spacer(1, 12))
        elements.append(Preformatted(self.report_matrix(), styles["Code"]))
        elements.append(Spacer(1, 0.5 * cm))

        with TemporaryDirectory() as tmpdir:

            prediction_path = Path(tmpdir) / "prediction.png"

            self.plot_prediction(show=False)
            plt.savefig(prediction_path, bbox_inches="tight")
            plt.close()

            elements.append(
                Paragraph("Predicted vs Actual", styles["Heading2"])
            )
            elements.append(
                Image(str(prediction_path), width=400, height=300)
            )

            residual_path = Path(tmpdir) / "residuals.png"

            self.plot_residuals(show=False)
            plt.savefig(residual_path, bbox_inches="tight")
            plt.close()

            elements.append(Spacer(1, 20))
            elements.append(
                Paragraph("Residual Plot", styles["Heading2"])
            )
            elements.append(
                Image(str(residual_path), width=400, height=300)
            )

            doc.build(elements)

    def __str__(self):
        return (
                "\n".join(self.report_lines())
                + "\n\n"
                + self.report_matrix()
        )
