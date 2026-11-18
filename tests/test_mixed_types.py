"""Tables with categorical / boolean columns and missing values (fitfleet.schema, fitfleet.encoding)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from fitfleet.classification import Classification
from fitfleet.cli import main
from fitfleet.encoding import build_encoder, is_tree_model
from fitfleet.model import load
from fitfleet.regression import Regression
from fitfleet.schema import ColumnKind, ColumnSchema
from fitfleet.utils import FilteringCriteriaClassification


def mixed(n=240, seed=0):
    """city decides the label together with card; plan and income are noise with gaps."""
    r = np.random.default_rng(seed)
    city = r.choice(["Paris", "Rome", "Oslo"], n)
    plan = pd.Series(r.choice(["basic", "pro", None], n, p=[.5, .4, .1]), dtype=object)
    income = r.normal(50, 10, n)
    income[r.random(n) < .1] = np.nan
    card = r.random(n) < .5
    y = pd.Series(np.where((city == "Paris") ^ card, "yes", "no"))
    return pd.DataFrame({"city": city, "plan": plan, "income": income, "card": card}), y


# ---------------------------------------------------------------- schema
def test_schema_detects_kinds_categories_and_missing():
    X, _ = mixed()
    s = ColumnSchema.infer(X)
    assert s.kinds == [ColumnKind.categorical, ColumnKind.categorical, ColumnKind.numeric, ColumnKind.boolean]
    assert s.categories == {"city": ["Oslo", "Paris", "Rome"], "plan": ["basic", "pro"]}
    assert s.missing == ["plan", "income"]
    assert not s.simple
    assert s.categorical_index == [0, 1] and s.numeric_index == [2, 3]


def test_schema_encodes_codes_missing_and_unknown():
    X, _ = mixed()
    s = ColumnSchema.infer(X)
    new = pd.DataFrame({"city": ["Rome", "Berlin", None], "plan": ["pro", None, "basic"],
                        "income": [1.5, np.nan, 3], "card": [True, False, None]})
    out = s.encode(new)
    np.testing.assert_array_equal(out[:, 0], [2, 3, np.nan])  # Berlin was never seen: code n = 3
    np.testing.assert_array_equal(out[:, 1], [1, np.nan, 0])
    np.testing.assert_array_equal(out[:, 2], [1.5, np.nan, 3])
    np.testing.assert_array_equal(out[:, 3], [1, 0, np.nan])
    assert s.label("city", 3) == "unknown" and s.label("city", np.nan) == "missing" and s.label("city", 1) == "Paris"


def test_schema_overrides_and_errors():
    X = pd.DataFrame({"zip": [10, 20, 10], "score": ["1", "2", "3"], "when": ["2020-01-01", "x", None]})
    s = ColumnSchema.infer(X.drop(columns="when"), categorical=["zip"], numeric=["score"])
    assert s.kinds == [ColumnKind.categorical, ColumnKind.numeric]
    np.testing.assert_array_equal(s.encode(X)[:, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="not dates"):
        ColumnSchema.infer(X, datetime=["when"])
    with pytest.raises(ValueError, match="not in the data"):
        ColumnSchema.infer(X, categorical=["nope"])
    with pytest.raises(ValueError, match="both"):
        ColumnSchema.infer(X, categorical=["zip"], numeric=["zip"])


def test_numeric_table_without_gaps_is_simple():
    s = ColumnSchema.infer(pd.DataFrame({"a": [1, 2], "b": [0.5, 1.0]}))
    assert s.simple


# ---------------------------------------------------------------- encoder per model
def _encoder(model, encoding="auto", preprocessing=None):
    X, _ = mixed()
    return build_encoder(ColumnSchema.infer(X), None, model, preprocessing, encoding)


def test_linear_and_distance_models_get_one_hot_and_imputation():
    for model in (LogisticRegression(), KNeighborsClassifier(), SVC()):
        enc, _ = _encoder(model, preprocessing=[StandardScaler()])
        cat, num = enc.transformers[0][1], enc.transformers[1][1]
        assert isinstance(cat, OneHotEncoder)
        assert isinstance(num.steps[0][1], SimpleImputer) and isinstance(num.steps[1][1], StandardScaler)


def test_tree_models_keep_codes_and_nan():
    enc, _ = _encoder(RandomForestClassifier())
    assert enc.transformers[0][1] == "passthrough"  # RandomForest accepts NaN (scikit-learn >= 1.4)
    assert enc.transformers[1][1] == "passthrough"
    assert is_tree_model(RandomForestClassifier()) and not is_tree_model(LogisticRegression())


def test_hist_gradient_boosting_gets_native_categories():
    model = HistGradientBoostingClassifier()
    _, prepared = _encoder(model)
    np.testing.assert_array_equal(prepared.categorical_features, [0, 1])
    assert model.categorical_features == "from_dtype"  # the user's model is not changed
    _, own = _encoder(HistGradientBoostingClassifier(categorical_features=[0]))
    assert list(own.categorical_features) == [0]  # a user setting is kept


def test_encoding_override():
    enc, _ = _encoder(RandomForestClassifier(), encoding="onehot")
    assert isinstance(enc.transformers[0][1], OneHotEncoder)
    enc, _ = _encoder(LogisticRegression(), encoding="ordinal")
    assert isinstance(enc.transformers[0][1], SimpleImputer)  # codes, missing ones filled for a NaN-less model
    with pytest.raises(ValueError, match="encoding"):
        _encoder(LogisticRegression(), encoding="nope")


# ---------------------------------------------------------------- training
@pytest.mark.parametrize("model", [LogisticRegression(max_iter=500), RandomForestClassifier(30, random_state=0),
                                   HistGradientBoostingClassifier(max_iter=30), KNeighborsClassifier()])
def test_every_model_family_trains_on_mixed_data(model):
    X, y = mixed()
    with Classification() as cls:
        cls.set(X, y, [model])
        cls.features.standard_scaler()
        result, = cls.k_folds(3)
    assert result.accuracy > 0.6
    assert result.features_index == [0, 1, 2, 3]


def test_feature_subsets_are_original_columns():
    X, y = mixed()
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [RandomForestClassifier(30, random_state=0)])
        cls.brute_force_features()
        assert len(cls._features_index) == 2 ** 4 - 1  # 4 columns, not the one-hot width
        results = cls.k_folds(3)
        best = max(results, key=lambda r: (r.accuracy, -len(r.features_index)))
        assert set(best.features_index) == {0, 3}  # city and card
        board = cls.experiment_report().leaderboard
    assert board.iloc[0]["features"] in ("city, card", "card, city")


def test_greedy_feature_search_on_mixed_data():
    X, y = mixed()
    with Classification() as cls:
        cls.set(X, y, [RandomForestClassifier(30, random_state=0)])
        cls.forward_selection(max_features=2)
        results = cls.k_folds(3)
    assert max(len(r.features_index) for r in results) <= 2


def test_feature_selection_uses_mutual_information_for_categories():
    X, y = mixed(400)
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [RandomForestClassifier(30, random_state=0)])
        cls.feature_selection(mutual_info_classif, [1, 2])
        assert len(cls._features_index) == 2
        with pytest.raises(ValueError, match="mutual_info"):
            cls.feature_selection(f_classif, [1])


def test_numeric_table_with_gaps_trains():
    r = np.random.default_rng(1)
    X = pd.DataFrame(r.normal(size=(150, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + r.normal(scale=.1, size=150)
    X.loc[r.random(150) < .15, "a"] = np.nan
    with Regression() as reg:
        reg.set(X, y, [LinearRegression()])
        reg.features.polynomial_features(2)  # can't take NaN: an imputer goes first
        result, = reg.k_folds(3)
    assert result.r2 > 0.5


def test_encoders_are_fitted_on_training_rows_only():
    X, y = mixed()
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [LogisticRegression(max_iter=500)])
        cls.features.standard_scaler()
        result, = cls.train_test_split(0.7)
        model, = cls.refit([result])
        numeric = model.model.named_steps["encode"].named_transformers_["numeric"]
        imputer, scaler = numeric.steps[0][1], numeric.steps[1][1]
        train_rows = cls._train_index
    train_income = X["income"].to_numpy()[train_rows]
    assert imputer.statistics_[0] == pytest.approx(np.nanmedian(train_income))
    assert scaler.n_samples_seen_ == len(train_rows) < len(X)
    filled = np.where(np.isnan(train_income), imputer.statistics_[0], train_income)
    assert scaler.mean_[0] == pytest.approx(filled.mean())


# ---------------------------------------------------------------- saved models
def test_saved_model_predicts_raw_rows_with_text_unseen_and_missing(tmp_path):
    X, y = mixed()
    with Classification() as cls:
        cls.set(X, y, [LogisticRegression(max_iter=500), RandomForestClassifier(30, random_state=0)])
        cls.k_folds(3)
        best = cls.filter_by(FilteringCriteriaClassification.accuracy)[0]
        path = best.save_model(tmp_path)
    model = load(path)
    new = pd.DataFrame({"card": [True, False, True], "city": ["Paris", "Berlin", None],
                        "plan": [None, "pro", "gold"], "income": [np.nan, 40.0, 55.0], "extra": [1, 2, 3]})
    assert list(model.predict(new)) == list(model.predict(new[["city", "plan", "income", "card"]].to_numpy()))
    assert set(model.predict(new)) <= {"yes", "no"}
    assert model.predict(new)[0] == "no"  # Paris with a card
    assert model.predict_proba(new).shape == (3, 2)
    if "encode" in model.estimator.named_steps:
        names = model.estimator.named_steps["encode"].get_feature_names_out(["city", "plan", "income", "card"])
        assert "city=Paris" in names or "city" in names


# ---------------------------------------------------------------- config / CLI
def test_cli_runs_and_predicts_a_mixed_csv(tmp_path):
    X, y = mixed()
    X.assign(customer_id=range(len(X)), label=y).to_csv(tmp_path / "data.csv", index=False)
    raw = {
        "data": {"path": "data.csv", "target": "label", "task": "classification"},
        "columns": {"drop": ["customer_id"]},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
        "preprocessing": ["standard_scaler"],
        "models": [{"name": "LogisticRegression", "params": {"max_iter": 500}},
                   {"name": "RandomForestClassifier", "params": {"n_estimators": 30, "random_state": 0}}],
        "output": {"dir": "runs", "keep": 1, "report": False},
    }
    (tmp_path / "exp.json").write_text(json.dumps(raw), encoding="utf-8")
    assert main(["run", str(tmp_path / "exp.json")]) == 0
    run = next((tmp_path / "runs").iterdir())
    board = pd.read_csv(run / "leaderboard.csv")
    assert "customer_id" not in board.loc[0, "features"]
    model_file = next(run.rglob("model.joblib"))
    X.head(5).to_csv(tmp_path / "new.csv", index=False)
    assert main(["predict", str(model_file), str(tmp_path / "new.csv"), "-o", str(tmp_path / "pred.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "pred.csv")) == 5


def test_columns_section_is_validated(tmp_path):
    from fitfleet.config import ConfigError, Experiment
    base = {"data": {"path": "d.csv", "target": "y", "task": "classification"},
            "models": [{"name": "LogisticRegression"}]}
    with pytest.raises(ConfigError, match=r"\[columns\] unknown"):
        Experiment.from_dict({**base, "columns": {"cat": ["a"]}})
    with pytest.raises(ConfigError, match="encoding"):
        Experiment.from_dict({**base, "columns": {"encoding": "nope"}})
    with pytest.raises(ConfigError, match="two kinds"):
        Experiment.from_dict({**base, "columns": {"categorical": ["a"], "numeric": ["a"]}})
    with pytest.raises(ConfigError, match="ordinal"):
        Experiment.from_dict({**base, "columns": {"ordinal": {"size": "S"}}})
    with pytest.raises(ConfigError, match="max_onehot"):
        Experiment.from_dict({**base, "columns": {"max_onehot": 0}})
