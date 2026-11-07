import pytest
from sklearn.linear_model import LinearRegression, Ridge

from ml_pipe.base import _add_hyper_parameter
from ml_pipe.regression import Regression
from ml_pipe.results.regression import RegressionResults
from ml_pipe.utils import ValidationType, FilteringCriteriaRegression


def test_set_regression(regression_class):
    assert regression_class.features is not None
    assert regression_class.targets is not None

    assert len(regression_class._train_data) == 3
    assert regression_class._features_index == [list(range(5))]


def test_train_test_split_regression(regression_class):
    results = regression_class.train_test_split(ratio=0.2)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, RegressionResults )
        assert (result.validation == ValidationType.train_test_split)
        assert result.r2 is not None
        assert result.rmse >= 0


def test_k_folds_regression(regression_class):
    results = regression_class.k_folds(n_splits=5)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, RegressionResults, )
        assert (result.validation == ValidationType.k_folds)


def test_leave_one_out_regression(regression_class):
    regression_class._process_limit = 2
    results = regression_class.leave_one_out()
    assert len(results) == 3
    for result in results:
        assert (result.validation == ValidationType.leave_one_out)


def test_hyper_parameter_create_combinations(regression_class):
    regression_class._hyper_parameter = {"alpha": [0.1, 1, 10, ], "fit_intercept": [True, False, ],
                                         "fake_parameter": [1, 2, ], }
    train_data = regression_class._train_data[1]
    result = _add_hyper_parameter(regression_class._hyper_parameter, train_data)
    assert len(result) == 6


def test_hyper_parameter_invalid_parameter(regression_class):
    regression_class._hyper_parameter = {"fake_parameter": [1, 2, 3, ]}
    train_data = regression_class._train_data[0]
    result = _add_hyper_parameter(regression_class._hyper_parameter, train_data)
    assert len(result) == 1
    assert result[0].model is not train_data.model


def test_multiple_models_regression(regression_class):
    regression_class.set_hyper_parameters({"n_estimators": [5, 10, ], "max_depth": [2, 5, ], })
    results = regression_class.train_test_split()
    assert len(results) == 6


def test_filter_by_r2(regression_class):
    r1 = RegressionResults(LinearRegression())
    r2 = RegressionResults(Ridge())
    r1._r2 = 0.5
    r2._r2 = 0.8
    regression_class._results = [r1, r2, ]
    result = regression_class.filter_by(FilteringCriteriaRegression.r2)
    assert result == [r2]


def test_filter_by_mae(regression_class):
    r1 = RegressionResults(LinearRegression())
    r2 = RegressionResults(Ridge())
    r1._mae = 0.5
    r2._mae = 0.2
    regression_class._results = [r1, r2, ]
    result = regression_class.filter_by(FilteringCriteriaRegression.mae)
    assert result == [r2]


def test_invalid_kfold(regression_class):
    with pytest.raises(ValueError):
        regression_class.k_folds(n_splits=1)


def test_invalid_process_limit():
    with pytest.raises(ValueError):
        Regression.process_limit(-1)


def test_filter_by_rejects_classification_criteria(regression_class):
    from ml_pipe.utils import FilteringCriteriaClassification
    regression_class._results = [RegressionResults(LinearRegression())]
    with pytest.raises(ValueError, match="criteria must be one of"):
        regression_class.filter_by(FilteringCriteriaClassification.accuracy)


def test_regression_feature_selection_and_brute_force(regression_class):
    from sklearn.feature_selection import f_regression
    regression_class.feature_selection(f_regression, [2, 3])
    assert [len(i) for i in regression_class._features_index] == [2, 3]
    regression_class.brute_force_features()
    assert len(regression_class._features_index) == 2 ** 5 - 1
