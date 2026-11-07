from sklearn.linear_model import LinearRegression, Ridge

from ml_pipe.results.regression import RegressionResults
from ml_pipe.utils import ValidationType, FilteringCriteriaRegression


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


def test_regression_feature_selection_and_brute_force(regression_class):
    from sklearn.feature_selection import f_regression
    regression_class.feature_selection(f_regression, [2, 3])
    assert [len(i) for i in regression_class._features_index] == [2, 3]
    regression_class.brute_force_features()
    assert len(regression_class._features_index) == 2 ** 5 - 1
