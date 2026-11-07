"""Behaviour shared by every results class (BaseResults), tested once for each subclass."""
import numpy as np
import pytest
from joblib import load
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipe.results.classification import ClassificationResults
from ml_pipe.results.regression import RegressionResults
from tests.conftest import DummyModel
from ml_pipe.utils import ValidationType

RESULT_TYPES = [ClassificationResults, RegressionResults]


def fitted_result(results_type):
    """A result whose model is fitted and whose predictions are evaluated."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    if results_type is ClassificationResults:
        y, model = (X[:, 0] > 0).astype(int), LogisticRegression()
    else:
        y, model = X @ [1.0, 2.0, 3.0], LinearRegression()
    model.fit(X[:30], y[:30])
    result = results_type(model)
    result.set_test(X[30:], y[30:])
    result.predict()
    result.validation = ValidationType.train_test_split
    return result


@pytest.mark.parametrize("results_type", RESULT_TYPES)
class TestValidation:
    def test_returns_empty_string_by_default(self, results_type):
        assert results_type(DummyModel()).validation == ""

    @pytest.mark.parametrize("value", list(ValidationType))
    def test_accepts_every_validation_type(self, results_type, value):
        r = results_type(DummyModel())
        r.validation = value
        assert r.validation == value

    @pytest.mark.parametrize("value", ["invalid", None, 123, object()])
    def test_rejects_invalid_validation_type(self, results_type, value):
        r = results_type(DummyModel())
        with pytest.raises(ValueError):
            r.validation = value


@pytest.mark.parametrize("results_type", RESULT_TYPES)
class TestName:
    def test_name_is_model_class(self, results_type):
        assert results_type(LinearRegression()).name == "LinearRegression"

    def test_name_of_pipeline_is_final_estimator(self, results_type):
        pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
        assert results_type(pipe).name == "LogisticRegression"


@pytest.mark.parametrize("results_type", RESULT_TYPES)
class TestSaving:
    def test_save_model_writes_loadable_model(self, results_type, tmp_path):
        result = fitted_result(results_type)
        result.save_model(str(tmp_path))
        files = list(tmp_path.glob("model_*.joblib"))
        assert len(files) == 1
        loaded = load(files[0])
        np.testing.assert_allclose(loaded.predict(np.eye(3)), result.model.predict(np.eye(3)))

    def test_save_model_refuses_unfitted(self, results_type, tmp_path):
        model = LogisticRegression() if results_type is ClassificationResults else LinearRegression()
        with pytest.raises(ValueError, match="not fitted"):
            results_type(model).save_model(str(tmp_path))
        assert not any(tmp_path.iterdir())

    def test_save_results_creates_folder_with_report_and_model(self, results_type, tmp_path):
        result = fitted_result(results_type)
        result.save_results(str(tmp_path / "nested" / "reports"))
        folders = list((tmp_path / "nested" / "reports").iterdir())
        assert len(folders) == 1 and folders[0].name.startswith(result.name)
        names = sorted(p.suffix for p in folders[0].iterdir())
        assert names == [".joblib", ".pdf"]
