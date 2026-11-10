import pandas as pd
from sklearn.datasets import load_iris as DB, load_diabetes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.feature_selection import chi2, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, Sound

REPORTS = r".\reports"


def load():
    data = DB()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # iris targets are 0..2; label them with iris's own class names
    y = pd.Series(data.target).map(dict(enumerate(data.target_names)))
    return X, y


def models():
    return [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC()]


def main():
    # classification (iris)
    train_test_split()
    k_folds()
    leave_one_out()
    # regression (diabetes)
    regression_train_test_split()
    regression_k_folds()
    regression_polynomial()


def train_test_split():
    # `with` closes the worker pool when done; runs inside the block reuse it.
    # sound= plays when the block ends (after training): none ping double chime success sparkle alarm bell
    with Classification(sound=Sound.ping) as test_class:
        test_class.set(*load(), models())
        test_class.features.min_max_scaler()  # refitted on the training rows of every split
        test_class.brute_force_features()
        test_class.train_test_split(0.8)  # 80% train, 20% test
        filtered_results = test_class.filter_by(FilteringCriteriaClassification.accuracy)  # refits the winners
        for i in tqdm(filtered_results):
            i.save_results(REPORTS)


def k_folds():
    with Classification(sound=Sound.double) as test_class:
        test_class.set(*load(), models())
        test_class.features.min_max_scaler()
        test_class.feature_selection(chi2, [1, 2, 3])
        test_class.k_folds()
        filtered_results = test_class.filter_by(FilteringCriteriaClassification.precision)
        for i in tqdm(filtered_results):
            i.save_results(REPORTS)


def leave_one_out():
    with Classification(sound=Sound.chime) as test_class:
        test_class.set(*load(), models())
        test_class.features.standard_scaler()
        test_class.features.polynomial_features(degree=3, include_bias=True)
        test_class.set_hyper_parameters(
            {"n_estimators": [50, 100, 120], "min_samples_split": [2, 3, 6], "C": [1, 2, 3]})
        test_class.leave_one_out()
        filtered_results = test_class.filter_by(FilteringCriteriaClassification.f1)
        for i in tqdm(filtered_results):
            i.save_results(REPORTS)


# ---------------------------------------------------------------- regression
def load_regression():
    # diabetes: 442 patients, 10 numeric features, target = disease progression after one year
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


def regression_models():
    return [LinearRegression(), Ridge(), RandomForestRegressor(random_state=0), SVR()]


def regression_train_test_split():
    with Regression(sound=Sound.success) as test_class:
        test_class.set(*load_regression(), regression_models())
        test_class.features.standard_scaler()  # refitted on the training rows of every split
        test_class.train_test_split(0.8)
        filtered_results = test_class.filter_by(FilteringCriteriaRegression.r2)  # highest R²
        for i in tqdm(filtered_results):
            print(i)
            i.save_results(REPORTS)


def regression_k_folds():
    with Regression(sound=Sound.bell) as test_class:
        test_class.set(*load_regression(), regression_models())
        test_class.features.standard_scaler()
        test_class.feature_selection(f_regression, [3, 5, 8])  # best 3 / 5 / 8 features
        test_class.set_hyper_parameters({"alpha": [0.1, 1, 10], "C": [1, 10, 100], "n_estimators": [100, 300]})
        test_class.k_folds(5)
        filtered_results = test_class.filter_by(FilteringCriteriaRegression.rmse)  # lowest RMSE
        for i in tqdm(filtered_results):
            print(i)
            i.save_results(REPORTS)


def regression_polynomial():
    with Regression(sound=Sound.alarm) as test_class:
        test_class.set(*load_regression(), [LinearRegression(), Ridge()])
        test_class.features.standard_scaler()
        test_class.features.polynomial_features(degree=2)  # 10 features -> 65 columns
        test_class.set_hyper_parameters({"alpha": [1, 10, 100]})
        test_class.k_folds(5)
        filtered_results = test_class.filter_by(FilteringCriteriaRegression.mae)  # lowest MAE
        for i in tqdm(filtered_results):
            print(i)
            i.save_results(REPORTS)


if __name__ == '__main__':
    main()
