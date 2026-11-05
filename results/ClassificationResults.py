import datetime
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Preformatted
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from results.BaseResults import BaseResults

mono_style = ParagraphStyle("Mono", fontName="Courier", fontSize=8, leading=10, )


class ClassificationResults(BaseResults):
    is_classification = True  # workers align predict_proba columns across folds

    def __init__(self, model):
        super().__init__(model)
        self._predict_score = None
        self._accuracy: float = 0
        self._precision: float = 0
        self._recall: float = 0
        self._f1: float = 0

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f1(self):
        return self._f1

    def predict(self):
        pred = self.model.predict(self.features_test)
        proba = self.model.predict_proba(self.features_test) if hasattr(self.model, "predict_proba") else None
        self.evaluate(self.target_test, pred, proba)

    def evaluate(self, y_true, y_pred, proba=None):
        self._target_test = y_true
        self._pred_test = y_pred
        self._predict_score = proba
        self._accuracy = accuracy_score(y_true, y_pred)
        self._precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        self._recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        self._f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        self._features_test = None

    def plot_confusion_matrix(self, labels=None, normalize=None, show: bool = True):
        """
        normalize: None, 'true', 'pred', 'all'
        """

        cm = confusion_matrix(self.target_test, self._pred_test, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
        plt.title(f"Confusion Matrix - {self.name}")
        if show:
            plt.show()

    def plot_roc_curve(self, show: bool = True):
        """
        Works for binary classification.
        For multiclass, you’d need binarization (handled below).
        """
        if self._predict_score is None:
            return
        classes = np.unique(self.target_test)

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(self.target_test, self._predict_score[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.name}")
            plt.legend()
            if show:
                plt.show()
            return

        y_test_bin = label_binarize(self.target_test, classes=classes)

        plt.figure()

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self._predict_score[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multiclass ROC - {self.name}")
        plt.legend()
        if show:
            plt.show()

    def save_pdf_report(self, filename: str):
        filename = os.path.join(filename,
                                f"report_{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
        if self._pred_test is None:
            raise ValueError("No predictions available")

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(filename)
        elements = []

        for line in self.report_lines():
            if line == "":
                elements.append(Spacer(1, 12))
            else:
                if isinstance(line, str):
                    elements.append(Paragraph(line.replace("\n", "<br/>"), styles["BodyText"]))
                else:
                    elements.append(Paragraph(line, styles["BodyText"]))

        elements.append(Spacer(1, 12))

        elements.append(Preformatted(self.report_matrix(), styles["BodyText"]))
        elements.append(Spacer(1, 0.5 * cm))

        with TemporaryDirectory() as tmpdir:

            cm_path = Path(tmpdir) / "cm.png"

            self.plot_confusion_matrix(show=False)
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()

            elements.append(
                Paragraph("Confusion Matrix", styles["Heading2"])
            )
            elements.append(Image(str(cm_path), width=400, height=300))

            if self._predict_score is not None:
                roc_path = Path(tmpdir) / "roc.png"

                self.plot_roc_curve(show=False)
                plt.savefig(roc_path, bbox_inches="tight")
                plt.close()

                elements.append(Spacer(1, 20))
                elements.append(
                    Paragraph("ROC Curve", styles["Heading2"])
                )
                elements.append(
                    Image(str(roc_path), width=400, height=300)
                )

            doc.build(elements)

    def __str__(self):
        lines = list(self.report_lines())
        return "\n".join(lines) + "\n\n" + self.report_matrix()

    def report_lines(self):
        """Yield all report lines."""
        yield f"Classification Report - {self.name}"
        yield ""

        yield f"Accuracy : {self.accuracy:.4f}"
        yield f"Precision: {self.precision:.4f}"
        yield f"Recall   : {self.recall:.4f}"
        yield f"F1 Score : {self.f1:.4f}"
        yield f" validation type: {self.validation}"

    def report_matrix(self) -> str:
        return classification_report(
            self.target_test,
            self._pred_test,
            zero_division=0,
            output_dict=False,
        )
