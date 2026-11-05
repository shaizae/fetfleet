import numpy as np
import pytest

from results.RegressionResults import RegressionResults
from tests.conftest import DummyModel
from utils.utils import ValidationType


def make_results(features_test=None, target_test=None, pred_test=None):
    r = RegressionResults(DummyModel())
    r._features_test = features_test
    r._target_test = target_test
    r._pred_test = pred_test
    return r


class TestConstructor:
    def test_default_values(self):
        r = RegressionResults(DummyModel())

        assert r.r2 == 0
        assert r.mae == 0
        assert r.mse == 0
        assert r.rmse == 0
        assert r._pred_test is None


class TestProperties:
    def test_metric_properties(self):
        r = RegressionResults(DummyModel())

        r._r2 = 0.95
        r._mae = 1.2
        r._mse = 3.4
        r._rmse = 1.84

        assert r.r2 == 0.95
        assert r.mae == 1.2
        assert r.mse == 3.4
        assert r.rmse == 1.84


class TestPredict:

    def test_predict_updates_metrics(self):
        x = np.arange(10).reshape(-1, 1)
        y = np.arange(10)

        class Model:
            def predict(self, features):
                return y

        r = RegressionResults(Model())
        r._features_test = x
        r._target_test = y

        r.predict()

        assert r.r2 == pytest.approx(1)
        assert r.mae == pytest.approx(0)
        assert r.mse == pytest.approx(0)
        assert r.rmse == pytest.approx(0)
        assert r._features_test is None

    def test_prediction_is_saved(self):
        x = np.arange(5).reshape(-1, 1)
        y = np.arange(5)

        class Model:
            def predict(self, features):
                return y

        r = RegressionResults(Model())
        r._features_test = x
        r._target_test = y

        r.predict()

        np.testing.assert_array_equal(r._pred_test, y)


class TestReportLines:

    def test_contains_all_metrics(self):
        r = RegressionResults(DummyModel())

        r._r2 = 0.91
        r._mae = 2
        r._mse = 4
        r._rmse = 2
        r.validation = ValidationType.train_test_split

        lines = list(r.report_lines())

        assert any("Regression Report" in line for line in lines)
        assert any("R² Score" in line for line in lines)
        assert any("MAE" in line for line in lines)
        assert any("MSE" in line for line in lines)
        assert any("RMSE" in line for line in lines)
        assert any("Validation type" in line for line in lines)


class TestReportMatrix:

    def test_contains_all_metrics(self):
        r = RegressionResults(DummyModel())

        r._r2 = 0.5
        r._mae = 1
        r._mse = 2
        r._rmse = np.sqrt(2)

        text = r.report_matrix()

        assert "Metric" in text
        assert "R²" in text
        assert "MAE" in text
        assert "MSE" in text
        assert "RMSE" in text


class TestStr:

    def test_contains_report(self):
        r = RegressionResults(DummyModel())

        r._r2 = 0.9
        r._mae = 1
        r._mse = 2
        r._rmse = np.sqrt(2)

        text = str(r)

        assert "Regression Report" in text
        assert "Metric" in text
        assert "RMSE" in text


class TestPlotPrediction:

    def test_show_true_calls_show(self, monkeypatch):
        called = False

        def fake_show():
            nonlocal called
            called = True

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        r = make_results(
            target_test=np.array([1, 2, 3]),
            pred_test=np.array([1, 2, 3]),
        )

        r.plot_prediction()

        assert called

    def test_show_false_does_not_call_show(self, monkeypatch):
        called = False

        def fake_show():
            nonlocal called
            called = True

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        r = make_results(
            target_test=np.array([1, 2, 3]),
            pred_test=np.array([1, 2, 3]),
        )

        r.plot_prediction(show=False)

        assert not called


class TestPlotResiduals:

    def test_show_true_calls_show(self, monkeypatch):
        called = False

        def fake_show():
            nonlocal called
            called = True

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        r = make_results(            target_test=np.array([1, 2, 3]),            pred_test=np.array([1, 2, 3]),        )

        r.plot_residuals()

        assert called


class TestSavePdfReport:

    def test_without_prediction_raises(self, tmp_path):
        r = RegressionResults(DummyModel())

        with pytest.raises(ValueError, match="No predictions available"):
            r.save_pdf_report(tmp_path)

    def test_creates_pdf(self, tmp_path):
        r = make_results(            target_test=np.array([1, 2, 3]),            pred_test=np.array([1, 2, 3]),        )

        r._r2 = 1
        r._mae = 0
        r._mse = 0
        r._rmse = 0

        r.save_pdf_report(tmp_path)

        pdfs = list(tmp_path.glob("*.pdf"))

        assert len(pdfs) == 1
        assert pdfs[0].exists()
        assert pdfs[0].stat().st_size > 0
