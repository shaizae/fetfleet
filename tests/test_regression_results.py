import numpy as np
import pytest

from ml_pipe.results.regression import RegressionResults
from tests.conftest import DummyModel
from ml_pipe.utils import ValidationType


def make_results(features_test=None, target_test=None, pred_test=None):
    r = RegressionResults(DummyModel())
    r._features_test = features_test
    r._target_test = target_test
    r._pred_test = pred_test
    return r


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
