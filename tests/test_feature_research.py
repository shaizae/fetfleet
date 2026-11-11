import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ml_pipe import FeatureResearch

RANDOM_STATE = 42
N_ESTIMATORS = 20


def iris() -> tuple[pd.DataFrame, pd.Series]:
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)


def crafted_classification() -> tuple[pd.DataFrame, pd.Series]:
    """signal / its near copy / pure noise / constant / id / mostly missing / a text column."""
    rng = np.random.default_rng(0)
    n = 200
    y = pd.Series(rng.integers(0, 2, n))
    signal = y + rng.normal(0, 0.3, n)
    missing = rng.normal(size=n)
    missing[: int(n * 0.7)] = np.nan
    X = pd.DataFrame({
        "signal": signal,
        "signal_copy": signal * 2 + rng.normal(0, 0.01, n),
        "noise": rng.normal(size=n),
        "constant": np.ones(n),
        "row_id": np.arange(n),
        "mostly_missing": missing,
        "city": rng.choice(["a", "b", "c"], n),
    })
    return X, y


def small_model():
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- construction
def test_task_guess():
    X, y = iris()
    assert FeatureResearch(X, y).task == "classification"
    assert FeatureResearch(X, y.map({0: "a", 1: "b", 2: "c"})).task == "classification"
    Xr, yr = make_regression(100, 3, random_state=RANDOM_STATE)
    assert FeatureResearch(Xr, yr).task == "regression"


def test_explicit_task_wins():
    X, y = iris()
    assert FeatureResearch(X, y, task="regression").task == "regression"


def test_numpy_input_gets_column_names():
    Xr, yr = make_regression(50, 3, random_state=RANDOM_STATE)
    research = FeatureResearch(Xr, yr)
    assert research.numeric_features == ["x0", "x1", "x2"]


@pytest.mark.parametrize("kwargs, message", [
    ({"task": "clustering"}, "task"),
    ({"cv": 1}, "cv"),
])
def test_bad_arguments(kwargs, message):
    X, y = iris()
    with pytest.raises(ValueError, match=message):
        FeatureResearch(X, y, **kwargs)


def test_length_mismatch_and_missing_target():
    X, y = iris()
    with pytest.raises(ValueError, match="rows"):
        FeatureResearch(X, y[:-1])
    y = y.astype(float)
    y[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        FeatureResearch(X, y)


def test_non_numeric_columns_are_skipped():
    X, y = crafted_classification()
    research = FeatureResearch(X, y)
    assert research.skipped == ["city"]
    assert "city" not in research.numeric_features
    assert not research._numeric.isna().any().any()


def test_does_not_modify_input():
    X, y = crafted_classification()
    before = X.copy()
    FeatureResearch(X, y).report(model=small_model())
    pd.testing.assert_frame_equal(X, before)


# ---------------------------------------------------------------- data quality
def test_missing_report():
    X, y = crafted_classification()
    table = FeatureResearch(X, y).missing_report()
    assert table.index[0] == "mostly_missing"
    assert table.loc["mostly_missing", "missing_pct"] == pytest.approx(70)
    assert table.loc["signal", "missing"] == 0


def test_constant_features():
    X, y = crafted_classification()
    research = FeatureResearch(X, y)
    assert list(research.constant_features().index) == ["constant"]
    with pytest.raises(ValueError):
        research.constant_features(0)


def test_cardinality_flags_ids_not_floats():
    X, y = crafted_classification()
    table = FeatureResearch(X, y).cardinality()
    assert table.loc["row_id", "id_like"]
    assert not table.loc["noise", "id_like"]  # continuous floats are unique too, but not ids


def test_distribution_summary():
    X, y = crafted_classification()
    table = FeatureResearch(X, y).distribution_summary()
    assert {"skew", "kurtosis", "outlier_pct"} <= set(table.columns)
    assert "city" not in table.index


# ---------------------------------------------------------------- vs target
def test_univariate_scores_rank_signal_first():
    X, y = crafted_classification()
    table = FeatureResearch(X, y).univariate_scores()
    assert table.index[0] in ("signal", "signal_copy")
    assert table.loc["signal", "p_value"] < 0.001


def test_univariate_regression_has_correlations():
    Xr, yr = make_regression(100, 3, n_informative=1, random_state=RANDOM_STATE)
    table = FeatureResearch(Xr, yr).univariate_scores()
    assert {"pearson", "spearman"} <= set(table.columns)


def test_mutual_information():
    X, y = crafted_classification()
    mi = FeatureResearch(X, y).mutual_information()
    assert mi["signal"] > mi["noise"]
    assert (mi >= 0).all()


def test_single_feature_score():
    X, y = iris()
    table = FeatureResearch(X, y).single_feature_score()
    assert table.index[0].startswith("petal")
    assert table["score"].between(0, 1).all()


def test_leakage_check():
    X, y = crafted_classification()
    X["leak"] = y
    leaks = FeatureResearch(X, y).leakage_check()
    assert list(leaks.index) == ["leak"]


# ---------------------------------------------------------------- redundancy
def test_correlation_matrix_symmetric():
    X, y = iris()
    corr = FeatureResearch(X, y).correlation_matrix("pearson")
    assert corr.shape == (4, 4)
    np.testing.assert_allclose(corr.values, corr.values.T)


def test_high_correlation_pairs_drops_the_weaker_one():
    X, y = crafted_classification()
    pairs = FeatureResearch(X, y).high_correlation_pairs(0.95)
    assert len(pairs) == 1
    assert {pairs.loc[0, "feature_a"], pairs.loc[0, "feature_b"]} == {"signal", "signal_copy"}
    assert pairs.loc[0, "drop"] in ("signal", "signal_copy")


def test_high_correlation_pairs_empty():
    Xr, yr = make_regression(100, 3, random_state=RANDOM_STATE)
    pairs = FeatureResearch(Xr, yr).high_correlation_pairs(0.99)
    assert pairs.empty
    assert list(pairs.columns) == ["feature_a", "feature_b", "correlation", "drop"]


def test_vif():
    X, y = crafted_classification()
    vif = FeatureResearch(X.drop(columns="constant"), y).vif()
    assert vif.index[0] in ("signal", "signal_copy")
    assert vif["noise"] < 2


# ---------------------------------------------------------------- importance
def test_model_importance_tree_and_linear():
    X, y = iris()
    research = FeatureResearch(X, y)
    assert research.model_importance(small_model()).sum() == pytest.approx(1)
    linear = research.model_importance(LogisticRegression(max_iter=1000))
    assert len(linear) == 4 and (linear >= 0).all()


def test_model_importance_unsupported_model():
    X, y = iris()
    with pytest.raises(ValueError, match="permutation_importance"):
        FeatureResearch(X, y).model_importance(SVC(kernel="rbf"))


def test_permutation_importance():
    X, y = crafted_classification()
    table = FeatureResearch(X.drop(columns="signal_copy"), y).permutation_importance(small_model(), n_repeats=3)
    assert table.index[0] == "signal"
    assert {"importance", "std_across_folds"} == set(table.columns)


def test_random_probe():
    X, y = crafted_classification()
    table = FeatureResearch(X.drop(columns="signal_copy"), y).random_probe(small_model(), n_probes=3, n_repeats=3)
    assert table.loc["signal", "beats_noise"]
    assert not table.loc["constant", "beats_noise"]
    assert not any(name.startswith("__probe") for name in table.index)


# ---------------------------------------------------------------- reporting
def test_report():
    X, y = iris()
    table = FeatureResearch(X, y).report(model=small_model())
    assert list(table.index[:2]) != []
    assert table["mean_rank"].is_monotonic_increasing
    assert table.index[0].startswith("petal")
    assert "permutation_importance" in table.columns


def test_report_without_importance():
    X, y = iris()
    table = FeatureResearch(X, y).report(importance=False)
    assert "permutation_importance" not in table.columns


def test_recommend():
    X, y = crafted_classification()
    rec = FeatureResearch(X, y).recommend()
    assert "constant" in rec.loc["constant", "reasons"]
    assert "id_like" in rec.loc["row_id", "reasons"]
    assert "missing" in rec.loc["mostly_missing", "reasons"]
    assert ("signal" in rec.index) != ("signal_copy" in rec.index)  # exactly one of the pair
    assert "noise" not in rec.index  # no probe -> noise is not judged
    assert "city" not in rec.index
    X.drop(columns=rec.index)  # index works directly with drop


def test_recommend_with_probe():
    X, y = crafted_classification()
    rec = FeatureResearch(X, y).recommend(probe=True, model=small_model())
    assert "noise" in rec.loc["noise", "reasons"]


# ---------------------------------------------------------------- storage (Features / Target)
def test_data_is_kept_in_features_and_target():
    from ml_pipe.features import Features
    from ml_pipe.target import Target
    X, y = iris()
    research = FeatureResearch(X, y)
    assert isinstance(research.features, Features)
    assert isinstance(research.targets, Target)
    np.testing.assert_array_equal(research.features.data, X.values)
    assert list(research.features.columns) == list(X.columns)


def test_classification_target_is_label_encoded():
    X, y = iris()
    labels = y.map({0: "setosa", 1: "versicolor", 2: "virginica"})
    research = FeatureResearch(X, labels)
    assert set(research.targets.data) == {0, 1, 2}
    assert list(research.targets.classes) == ["setosa", "versicolor", "virginica"]
    assert (research.targets.show_original_data == labels.to_numpy()).all()


def test_regression_target_is_not_encoded():
    Xr, yr = make_regression(50, 3, random_state=RANDOM_STATE)
    research = FeatureResearch(Xr, yr)
    assert research.targets.classes is None
    np.testing.assert_allclose(research.targets.data, yr)


def test_column_types_survive_features_storage():
    X, y = crafted_classification()
    research = FeatureResearch(X, y)
    assert research.features.data.dtype == object  # mixed table -> one object array
    pd.testing.assert_series_equal(research.X.dtypes, X.dtypes)


def test_methods_see_steps_applied_to_features():
    X, y = iris()
    research = FeatureResearch(X, y)
    research.features.standard_scaler()
    summary = research.distribution_summary()
    np.testing.assert_allclose(summary["mean"], 0, atol=1e-9)


def test_polynomial_features_renames_columns():
    X, y = iris()
    research = FeatureResearch(X, y)
    research.features.polynomial_features(2)
    assert research.numeric_features[:2] == ["x0", "x1"]
    assert len(research.mutual_information()) == 14


# ---------------------------------------------------------------- noise check uses both importance and MI
def test_random_probe_columns():
    X, y = crafted_classification()
    table = FeatureResearch(X, y).random_probe(small_model(), n_probes=3, n_repeats=2)
    assert {"mutual_information", "mi_noise_level", "mi_beats_noise", "is_noise"} <= set(table.columns)
    assert (table["is_noise"] == (~table["beats_noise"] & ~table["mi_beats_noise"])).all()


def test_correlated_twins_are_not_called_noise():
    # two copies of the signal share the importance, so shuffling one costs nothing;
    # mutual information still sees the signal, so neither may be flagged as noise
    X, y = crafted_classification()
    table = FeatureResearch(X[["signal", "signal_copy", "noise"]], y).random_probe(small_model(), n_repeats=2)
    assert not table.loc["signal", "is_noise"]
    assert not table.loc["signal_copy", "is_noise"]


# ---------------------------------------------------------------- charts and PDF
def test_plot_importance_draws_noise_line_from_probe():
    X, y = iris()
    research = FeatureResearch(X, y)
    probed = research.random_probe(small_model(), n_probes=2, n_repeats=2)
    fig = research.plot_importance(probed)
    ax = fig.axes[0]
    assert len(ax.patches) == 4  # one bar per feature
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_correlation():
    X, y = iris()
    fig = FeatureResearch(X, y).plot_correlation()
    assert len(fig.axes[0].texts) == 16  # 4 x 4 values written in the cells
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_overview_lines_show_original_labels():
    X, y = iris()
    lines = FeatureResearch(X, y.map({0: "setosa", 1: "versicolor", 2: "virginica"})).overview_lines()
    assert any("setosa: 50 (33%)" in line for line in lines)


@pytest.mark.parametrize("probe", [True, False])
def test_save_pdf_report(tmp_path, probe):
    X, y = crafted_classification()
    path = FeatureResearch(X, y).save_pdf_report(tmp_path / "out", model=small_model(), probe=probe)
    assert path.parent == tmp_path / "out"
    assert path.name.startswith("feature_research_") and path.suffix == ".pdf"
    assert path.read_bytes()[:4] == b"%PDF"
    assert path.stat().st_size > 10_000  # charts are embedded


def test_save_pdf_report_single_feature(tmp_path):
    Xr, yr = make_regression(60, 1, random_state=RANDOM_STATE)
    path = FeatureResearch(Xr, yr).save_pdf_report(tmp_path, model=None, probe=False)
    assert path.exists()
