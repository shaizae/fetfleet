import numpy as np
import pytest

from tests.conftest import DummyModel
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.utils import FilteringCriteriaClassification, ValidationType


def make_results(features_test=None, target_test=None, pred_test=None):
    """Build a Results instance with internal state set directly,
    bypassing predict()/set_test() (which require a real sklearn model)."""
    r = ClassificationResults(DummyModel())
    r._features_test = features_test
    r._target_test = target_test
    r._pred_test = pred_test
    return r


def test_append_results_concatenates_and_chains():
    r1 = make_results(features_test=np.array([[1, 1]]), target_test=np.array([0]), pred_test=np.array([0]))
    r2 = make_results(features_test=np.array([[2, 2]]), target_test=np.array([1]), pred_test=np.array([1]))
    r3 = make_results(target_test=np.array([2]))
    assert r1.append_results(r2).append_results(r3) is r1
    np.testing.assert_array_equal(r1._features_test, [[1, 1], [2, 2]])  # r3 had none: kept as is
    np.testing.assert_array_equal(r1._target_test, [0, 1, 2])
    np.testing.assert_array_equal(r1._pred_test, [0, 1])


def test_append_results_copies_into_an_empty_result():
    incoming = np.array([[5, 6]])
    r1, r2 = make_results(), make_results(features_test=incoming)
    r1.append_results(r2)
    r1._features_test[0, 0] = 999
    assert incoming[0, 0] == 5  # a copy, so the source is untouched


def test_append_results_of_two_empty_results_stays_empty():
    r1 = make_results()
    r1.append_results(make_results())
    assert r1._features_test is None and r1._target_test is None and r1._pred_test is None


def test_filter_by_returns_best(classifier):
    r1 = ClassificationResults(None)
    r1._accuracy = 0.7

    r2 = ClassificationResults(None)
    r2._accuracy = 0.9

    classifier._results = [r1, r2]

    result = classifier.filter_by(FilteringCriteriaClassification.accuracy)

    assert result == [r2]


# ---------------------------------------------------------------------------
# metrics, prediction, reports and plots
# ---------------------------------------------------------------------------
def fitted(n_classes: int) -> ClassificationResults:
    """LogisticRegression fitted on separable data, evaluated on held-out rows."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=120, n_features=4, n_informative=3, n_redundant=0,
                               n_classes=n_classes, n_clusters_per_class=1, random_state=0)
    r = ClassificationResults(LogisticRegression(max_iter=500).fit(X[:80], y[:80]))
    r.set_test(X[80:], y[80:])
    r.predict()
    r.validation = ValidationType.train_test_split
    return r


class PredictOnly:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestEvaluate:
    def test_known_metric_values(self):
        r = ClassificationResults(None)
        r.evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        assert r.accuracy == pytest.approx(0.75)
        assert r.precision == pytest.approx((1 + 2 / 3) / 2)  # weighted by support 2/2
        assert r.recall == pytest.approx((0.5 + 1) / 2)
        assert r.f1 == pytest.approx((2 / 3 + 0.8) / 2)

    def test_stores_predictions_and_drops_features(self):
        r = ClassificationResults(None)
        r._features_test = np.ones((2, 2))
        proba = np.array([[0.9, 0.1], [0.2, 0.8]])
        r.evaluate(np.array([0, 1]), np.array([0, 1]), proba)
        np.testing.assert_array_equal(r._pred_test, [0, 1])
        np.testing.assert_array_equal(r.target_test, [0, 1])
        assert r._predict_score is proba
        assert r.features_test is None

    def test_class_never_predicted_gives_zero_without_warning(self):
        import warnings
        r = ClassificationResults(None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r.evaluate(np.array([0, 1, 1]), np.array([0, 0, 0]))
        # class 0: precision 1/3, support 1; class 1: never predicted -> 0, support 2
        assert r.precision == pytest.approx((1 / 3 * 1 + 0 * 2) / 3)


class TestPredict:
    def test_predict_with_probabilities(self):
        r = fitted(3)
        assert r.accuracy > 0.8
        assert r._predict_score.shape == (40, 3)
        np.testing.assert_allclose(r._predict_score.sum(axis=1), 1)

    def test_predict_without_predict_proba(self):
        r = ClassificationResults(PredictOnly())
        r.set_test(np.ones((3, 2)), np.array([0, 0, 1]))
        r.predict()
        assert r._predict_score is None
        assert r.accuracy == pytest.approx(2 / 3)


class TestReports:
    def test_report_lines_contain_all_metrics(self):
        text = "\n".join(fitted(2).report_lines())
        for label in ("Accuracy", "Precision", "Recall", "F1 Score", "train_test_split", "LogisticRegression"):
            assert label in text

    def test_report_matrix_is_per_class_report(self):
        matrix = fitted(3).report_matrix()
        assert "precision" in matrix and "support" in matrix
        assert all(str(c) in matrix for c in (0, 1, 2))

    def test_str_combines_lines_and_matrix(self):
        r = fitted(2)
        assert str(r).startswith("Classification Report - LogisticRegression")
        assert r.report_matrix() in str(r)


@pytest.fixture
def show_calls(monkeypatch):
    calls = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: calls.append(1))
    yield calls
    import matplotlib.pyplot as plt
    plt.close("all")


class TestPlots:
    @pytest.mark.parametrize("normalize", [None, "true"])
    def test_confusion_matrix_shows(self, show_calls, normalize):
        fitted(3).plot_confusion_matrix(normalize=normalize)
        assert show_calls == [1]

    def test_confusion_matrix_show_false(self, show_calls):
        fitted(3).plot_confusion_matrix(show=False)
        assert show_calls == []

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_roc_curve_binary_and_multiclass(self, show_calls, n_classes):
        import matplotlib.pyplot as plt
        fitted(n_classes).plot_roc_curve()
        assert show_calls == [1]
        curves = [line for line in plt.gca().get_lines() if line.get_label().startswith(("AUC", "Class"))]
        assert len(curves) == (1 if n_classes == 2 else n_classes)

    def test_roc_curve_skipped_without_probabilities(self, show_calls):
        r = ClassificationResults(PredictOnly())
        r.set_test(np.ones((3, 2)), np.array([0, 0, 1]))
        r.predict()
        r.plot_roc_curve()
        assert show_calls == []


class TestSavePdfReport:
    def test_without_prediction_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No predictions available"):
            ClassificationResults(None).save_pdf_report(str(tmp_path))

    @pytest.mark.parametrize("with_proba", [True, False])
    def test_creates_pdf(self, tmp_path, with_proba):
        r = fitted(3)
        if not with_proba:
            r._predict_score = None  # report without the ROC section
        r.save_pdf_report(str(tmp_path))
        pdfs = list(tmp_path.glob("report_LogisticRegression_*.pdf"))
        assert len(pdfs) == 1 and pdfs[0].stat().st_size > 1000


def test_roc_curve_binary_with_text_labels(show_calls):
    """roc_curve needs pos_label when labels aren't 0/1; this used to raise."""
    r = ClassificationResults(None)
    r.evaluate(np.array(["no", "yes", "yes", "no"]), np.array(["no", "yes", "no", "no"]),
               np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]))
    r.plot_roc_curve()
    assert show_calls == [1]


def test_confusion_matrix_axes_show_class_names(show_calls):
    import matplotlib.pyplot as plt
    r = ClassificationResults(None)
    r.evaluate(np.array(["cat", "dog", "dog"]), np.array(["cat", "dog", "cat"]))
    r.plot_confusion_matrix(show=False)
    assert [t.get_text() for t in plt.gca().get_xticklabels()] == ["cat", "dog"]
