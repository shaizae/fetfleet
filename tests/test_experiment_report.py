import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from ml_pipe import Classification, FilteringCriteriaRegression, Regression
from ml_pipe.results.experiment import ExperimentReport


@pytest.fixture(scope="module")
def classification_report():
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    with Classification() as run:
        run.random_state = 0
        run.set(X, pd.Series(d.target), [DecisionTreeClassifier(random_state=0), LogisticRegression(max_iter=500)])
        run.set_hyper_parameters({"max_depth": [1, 3], "C": [0.01, 1.0]})
        run.k_folds(3)
        return run.experiment_report()


@pytest.fixture(scope="module")
def regression_run_report():
    X, y = make_regression(60, 3, noise=5, random_state=0)
    with Regression() as run:
        run.random_state = 0
        run.set(pd.DataFrame(X), y, [Ridge(), LinearRegression()])
        run.train_test_split(0.8)
        return run.experiment_report(FilteringCriteriaRegression.mae)


def test_classification_report_covers_every_configuration(classification_report):
    rep = classification_report
    assert rep.task == "classification"
    assert rep.n_configurations == 4
    assert list(rep.summary["configurations"]) == [2, 2]
    accuracy = rep.leaderboard["accuracy"]
    assert accuracy.is_monotonic_decreasing  # best first
    assert rep.best["accuracy"] == accuracy.max()
    assert set(rep.summary.index) == {"DecisionTreeClassifier", "LogisticRegression"}
    assert rep.summary["best_rank"].iloc[0] == 1


def test_classification_summary_stats_match_leaderboard(classification_report):
    rep = classification_report
    for model, row in rep.summary.iterrows():
        values = rep.leaderboard.loc[rep.leaderboard["model"] == model, "accuracy"]
        assert row["best_accuracy"] == values.max()
        assert row["worst_accuracy"] == values.min()
        assert row["mean_accuracy"] == pytest.approx(values.mean())
        assert "mean_f1" in row.index


def test_regression_report_ranks_lower_is_better(regression_run_report):
    rep = regression_run_report
    assert rep.task == "regression"
    assert rep.leaderboard["mae"].is_monotonic_increasing
    assert "lower is better" in str(rep)
    assert "mean_r2" in rep.summary.columns


def test_report_text_and_files(classification_report, tmp_path):
    text = str(classification_report)
    assert "Experiment Report - classification" in text and "Per model" in text
    board, summary = classification_report.save_csv(tmp_path)
    assert len(pd.read_csv(board)) == 4 and len(pd.read_csv(summary)) == 2
    pdf = classification_report.save_pdf_report(tmp_path)
    assert pdf.exists() and pdf.stat().st_size > 0


def test_report_before_any_run_raises():
    run = Classification()
    with pytest.raises(ValueError, match="no results"):
        run.experiment_report()


def test_report_of_empty_results_raises():
    with pytest.raises(ValueError, match="no results"):
        ExperimentReport("classification", [], None, None)


def test_roc_curve_per_model(classification_report):
    curves = classification_report.roc_curves()
    assert set(curves) == {"DecisionTreeClassifier", "LogisticRegression"}
    for fpr, tpr, roc_auc in curves.values():
        assert fpr[0] == 0 and tpr[-1] == 1 and 0.5 < roc_auc <= 1
    assert classification_report.plot_roc_curves(show=False)


def test_binary_roc_and_model_without_proba(tmp_path):
    from sklearn.svm import LinearSVC
    d = load_iris()
    mask = d.target < 2
    X = pd.DataFrame(d.data[mask], columns=d.feature_names)
    y = pd.Series(["no", "yes"])[d.target[mask]].reset_index(drop=True)  # string labels
    with Classification() as run:
        run.set(X, y, [LogisticRegression(max_iter=500), LinearSVC()])
        run.k_folds(3)
        rep = run.experiment_report()
    assert set(rep.roc_curves()) == {"LogisticRegression"}  # LinearSVC has no predict_proba
    assert rep.save_pdf_report(tmp_path).stat().st_size > 0


def test_regression_has_no_roc(regression_run_report):
    assert regression_run_report.roc_curves() == {}
    assert not regression_run_report.plot_roc_curves(show=False)


def test_report_lists_features_and_hyper_parameters(classification_report):
    rep = classification_report
    assert rep.feature_names == list(load_iris().feature_names)
    assert rep.hyper_parameters == {"max_depth": [1, 3], "C": [0.01, 1.0]}
    text = str(rep)
    assert "Experiment setup" in text and "petal width (cm)" in text
    assert "max_depth" in text and "[0.01, 1.0]" in text
    assert "Feature subsets tried: 1" in text
