import matplotlib.pyplot as plt
import numpy as np
import pytest

from fitfleet.results.regression import RegressionResults
from tests.conftest import DummyModel
from fitfleet.utils import ValidationType


@pytest.fixture(autouse=True)
def close_figures():
    """The plot tests draw real figures; close them so they do not pile up for the whole session."""
    yield
    plt.close("all")


def make_results(features_test=None, target_test=None, pred_test=None):
    r = RegressionResults(DummyModel())
    r._features_test = features_test
    r._target_test = target_test
    r._pred_test = pred_test
    return r


class TestPredict:

    def test_predict_updates_metrics(self):
        # errors 1, 0, -1, 2 -> every metric has a distinct, hand-computed value
        x = np.arange(4).reshape(-1, 1)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([2.0, 2.0, 2.0, 6.0])

        class Model:
            def predict(self, features):
                return pred

        r = RegressionResults(Model())
        r._features_test = x
        r._target_test = y

        r.predict()

        assert r.mae == pytest.approx(1.0)            # (1 + 0 + 1 + 2) / 4
        assert r.mse == pytest.approx(1.5)            # (1 + 0 + 1 + 4) / 4
        assert r.rmse == pytest.approx(np.sqrt(1.5))
        assert r.r2 == pytest.approx(-0.2)            # 1 - 6 / 5 (worse than predicting the mean)
        assert r._features_test is None
        np.testing.assert_array_equal(r._pred_test, pred)


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

    def test_contains_metric_table(self):
        r = RegressionResults(DummyModel())

        r._r2 = 0.9
        r._mae = 1
        r._mse = 2
        r._rmse = np.sqrt(2)

        text = str(r)

        assert "Metric" in text  # the report_matrix table; the report lines are covered by TestReportLines


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
