from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris as DB, load_diabetes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.feature_selection import chi2, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from ml_pipe.feature_research import FeatureResearch
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
    # feature research (no training run needed)
    feature_research_classification()
    feature_research_regression()
    # classification (iris)
    train_test_split()
    k_folds()
    leave_one_out()
    # regression (diabetes)
    regression_train_test_split()
    regression_k_folds()
    regression_polynomial()
    # experiment reports with every plot (PDF + PNGs)
    classification_experiment_report()
    regression_experiment_report()


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


# ---------------------------------------------------------------- feature research
def feature_research_classification():
    # stand-alone: look at the features before choosing models / feature subsets
    research = FeatureResearch(*load(), cv=5, random_state=0)  # task guessed from y -> classification
    print(research)
    print(research.targets.classes)  # target is label-encoded like in a run: code i = classes[i]
    print(research.missing_report())
    print(research.univariate_scores())  # ANOVA F-test per feature
    print(research.mutual_information())  # also catches non-linear relations
    print(research.high_correlation_pairs(0.9))  # `drop` = the less informative feature of each pair
    print(research.permutation_importance(RandomForestClassifier(random_state=0)))
    print(research.leakage_check(0.98))  # features that alone score suspiciously high
    report = research.report()  # one row per feature, every score + mean rank (1 = best)
    print(report)
    Path(REPORTS).mkdir(parents=True, exist_ok=True)
    report.to_csv(Path(REPORTS) / "feature_research_iris.csv")
    # everything in one PDF: recommendations, ranking, importance chart, data quality, correlation heatmap
    research.save_pdf_report(REPORTS, model=RandomForestClassifier(random_state=0))


def feature_research_regression():
    # research first, then train only on the features worth keeping
    X, y = load_regression()
    research = FeatureResearch(X, y, random_state=0)  # float target -> regression
    research.features.standard_scaler()  # every method after this sees the scaled data
    print(research.distribution_summary())  # skew, kurtosis, % outliers
    print(research.vif())  # multicollinearity (matters for linear models)
    print(research.random_probe(RandomForestRegressor(random_state=0)))  # beats_noise False -> likely useless
    to_drop = research.recommend(correlation_threshold=0.9, probe=True,
                                 model=RandomForestRegressor(random_state=0))
    print(to_drop)  # index = feature, reasons = why

    with Regression(sound=Sound.sparkle) as test_class:
        test_class.set(X.drop(columns=to_drop.index), y, regression_models())
        test_class.features.standard_scaler()
        test_class.k_folds(5)
        filtered_results = test_class.filter_by(FilteringCriteriaRegression.r2)
        for i in tqdm(filtered_results):
            print(i)
            i.save_results(REPORTS)


# ---------------------------------------------------------------- experiment reports and plots
def show_plot_status(viz):
    """What was drawn, skipped (not applicable / not measured) or failed - a failure never stops the run."""
    print(viz.summary())  # e.g. "23 generated, 3 skipped, 0 failed"
    for r in viz.skipped:
        print(f"  skipped {r.title}: {r.reason}")
    for r in viz.failed:
        print(f"  failed  {r.title}: {r.error_type}: {r.reason}")


def classification_experiment_report():
    with Classification(sound=Sound.double) as test_class:
        test_class.random_state = 0
        test_class.set(*load(), [RandomForestClassifier(random_state=0), DecisionTreeClassifier(random_state=0),
                                 SVC(probability=True, random_state=0)])  # probability=True -> ROC / PR curves
        test_class.features.standard_scaler()
        test_class.set_hyper_parameters({"n_estimators": [50, 200], "max_depth": [2, 4, None], "C": [0.1, 1, 10]})
        test_class.k_folds(5)  # k-fold -> per-fold plot; fit times -> runtime and cost plots
        best = test_class.filter_by(FilteringCriteriaClassification.accuracy)  # refitted -> feature importance

        report = test_class.experiment_report()
        print(report)
        # one PDF: summary, configuration, model comparison, confusion matrices, per-class scores,
        # ROC / precision-recall, per-fold scores, hyper-parameter sensitivity, runtime, plot index
        print(report.save_pdf_report(REPORTS))
        show_plot_status(report.last_visualizations)

        # the same plots without a PDF, as PNG files
        viz = report.visualize()
        viz.save_images(Path(REPORTS) / "plots_iris")

        # per-model report of the winner: its confusion matrix, ROC, PR, importance, ...
        print(best[0].save_pdf_report(REPORTS))
        show_plot_status(best[0].last_visualizations)


def regression_experiment_report():
    with Regression(sound=Sound.chime) as test_class:
        test_class.random_state = 0
        test_class.set(*load_regression(), [Ridge(), RandomForestRegressor(random_state=0)])
        test_class.features.standard_scaler()
        test_class.feature_selection(f_regression, [3, 6, 10])  # -> "score vs number of features" plot
        test_class.set_hyper_parameters({"alpha": [0.1, 1, 10], "max_depth": [3, 6, None]})
        test_class.k_folds(5)
        best = test_class.filter_by(FilteringCriteriaRegression.r2)

        report = test_class.experiment_report(FilteringCriteriaRegression.rmse)  # rank by lowest RMSE
        # actual vs predicted, residuals, Q-Q, error by range, metric comparison, ...
        print(report.save_pdf_report(REPORTS))
        show_plot_status(report.last_visualizations)
        print(best[0].save_pdf_report(REPORTS))


if __name__ == '__main__':
    main()
