"""Wide categorical columns, ordinal and date columns, id-like columns, FeatureResearch on mixed data."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from fitfleet.classification import Classification
from fitfleet.cli import main
from fitfleet.encoding import DateParts, build_encoder
from fitfleet.feature_research import FeatureResearch
from fitfleet.model import load
from fitfleet.regression import Regression
from fitfleet.schema import ColumnKind, ColumnSchema, id_like_columns
from fitfleet.utils import FilteringCriteriaRegression


def shops(n=400, seed=0):
    """Regression: sales depend on the shop (60 shops - a wide categorical), the size (ordinal),
    the month of the date and a numeric column with gaps."""
    r = np.random.default_rng(seed)
    shop = r.integers(0, 60, n)
    shop_effect = r.permutation(np.linspace(-3, 3, 60))[shop]  # not in name order: codes alone can't line it up
    size = r.choice(["S", "M", "L"], n)
    date = pd.Timestamp("2024-01-01") + pd.to_timedelta(r.integers(0, 365, n), unit="D")
    temp = r.normal(15, 5, n)
    sales = (shop_effect + pd.Series(size).map({"S": 0.0, "M": 1.0, "L": 2.0}).to_numpy()
             + np.where(date.month.isin([11, 12]), 2.0, 0.0) + 0.05 * temp + r.normal(0, 0.3, n))
    temp[r.random(n) < 0.1] = np.nan
    X = pd.DataFrame({"shop": [f"shop_{i:02d}" for i in shop], "size": size, "date": date, "temp": temp})
    return X, pd.Series(sales)


# ---------------------------------------------------------------- schema
def test_ordinal_and_datetime_kinds():
    X, _ = shops()
    s = ColumnSchema.infer(X, ordinal={"size": ["S", "M", "L"]})
    assert s.kinds == [ColumnKind.categorical, ColumnKind.ordinal, ColumnKind.datetime, ColumnKind.numeric]
    out = s.encode(pd.DataFrame({"shop": ["shop_00"], "size": ["L"], "date": [pd.Timestamp("1970-01-02")],
                                 "temp": [1.0]}))
    np.testing.assert_array_equal(out[0], [0, 2, 86400, 1.0])
    unseen = s.encode(pd.DataFrame({"shop": ["x"], "size": ["XL"], "date": [None], "temp": [None]}))
    assert unseen[0, 0] == 60 and np.isnan(unseen[0, 1:]).all()  # unseen ordinal value -> missing
    assert s.numeric_index == [1, 3] and s.datetime_index == [2]


def test_ordinal_errors_and_ordered_pandas_categories():
    X = pd.DataFrame({"size": ["S", "M", "XL"]})
    with pytest.raises(ValueError, match="missing from its order"):
        ColumnSchema.infer(X, ordinal={"size": ["S", "M", "L"]})
    with pytest.raises(ValueError, match="at least 2"):
        ColumnSchema.infer(X, ordinal={"size": ["S"]})
    ordered = pd.DataFrame({"size": pd.Categorical(["S", "L"], categories=["S", "M", "L"], ordered=True)})
    s = ColumnSchema.infer(ordered)
    assert s.kinds == [ColumnKind.ordinal] and s.categories["size"] == ["S", "M", "L"]


def test_text_dates_are_parsed_and_time_zones_converted():
    X = pd.DataFrame({"when": ["2024-03-01", "2024-03-02 06:00", None]})
    s = ColumnSchema.infer(X, datetime=["when"])
    out = s.encode(X)[:, 0]
    assert out[1] - out[0] == 30 * 3600 and np.isnan(out[2])
    tz = pd.DataFrame({"when": pd.to_datetime(["2024-01-01 02:00"]).tz_localize("Europe/Paris")})
    assert ColumnSchema.infer(tz).encode(tz)[0, 0] == ColumnSchema.infer(
        pd.DataFrame({"when": pd.to_datetime(["2024-01-01 01:00"])})).encode(
        pd.DataFrame({"when": pd.to_datetime(["2024-01-01 01:00"])}))[0, 0]


def test_id_like_columns():
    X = pd.DataFrame({"id": [f"c{i}" for i in range(50)], "city": ["a", "b"] * 25, "n": range(50)})
    assert id_like_columns(X) == ["id"]  # numbers are never flagged, only text
    assert id_like_columns(X.head(10)) == []  # too few rows to tell


# ---------------------------------------------------------------- encoder
def test_date_parts():
    seconds = np.array([[0.0], [86400 * 31 + 3600 * 5], [np.nan]])
    parts = DateParts().fit(seconds)
    out = parts.transform(seconds)
    assert list(parts.get_feature_names_out(["d"])) == ["d_year", "d_month", "d_day", "d_weekday",
                                                        "d_day_of_year", "d_hour", "d_days"]
    np.testing.assert_array_equal(out[0], [1970, 1, 1, 3, 1, 0, 0])  # 1970-01-01 was a Thursday
    np.testing.assert_array_equal(out[1], [1970, 2, 1, 6, 32, 5, 31 + 5 / 24])
    assert np.isnan(out[2]).all()
    assert "d_hour" not in DateParts().fit(np.array([[0.0], [86400.0]])).get_feature_names_out(["d"])


def test_wide_categories_are_target_encoded_for_linear_models_only():
    X, _ = shops()
    s = ColumnSchema.infer(X, ordinal={"size": ["S", "M", "L"]})
    enc, _ = build_encoder(s, None, Ridge(), [], regression=True)
    kinds = {name: t for name, t, _ in enc.transformers}
    assert isinstance(kinds["categorical_target"], TargetEncoder)
    assert kinds["categorical_target"].target_type == "continuous"
    enc, _ = build_encoder(s, None, Ridge(), [], max_onehot=100)
    assert isinstance({n: t for n, t, _ in enc.transformers}["categorical_onehot"], OneHotEncoder)
    enc, _ = build_encoder(s, None, RandomForestRegressor(), [])
    assert {n for n, _, _ in enc.transformers} == {"categorical", "numeric", "datetime"}
    enc, _ = build_encoder(s, None, RandomForestRegressor(), [], "target", regression=True)
    assert "categorical_target" in {n for n, _, _ in enc.transformers}


# ---------------------------------------------------------------- training
# Ridge gets the 60 shops target encoded; the forest gets codes, which on 400 rows and shuffled
# shop effects carry the signal only partly - both must still learn something real.
@pytest.mark.parametrize("model, min_r2", [(Ridge(), 0.7), (RandomForestRegressor(50, random_state=0), 0.3)])
def test_regression_with_wide_categories_ordinal_and_dates(model, min_r2):
    X, y = shops()
    with Regression() as reg:
        reg.random_state = 0
        with pytest.warns(UserWarning, match="date columns"):
            reg.set(X, y, [model], ordinal={"size": ["S", "M", "L"]})
        reg.features.standard_scaler()
        result, = reg.k_folds(4)
    assert result.r2 > min_r2, result.r2


def test_target_encoding_beats_codes_for_a_linear_model():
    X, y = shops()
    scores = {}
    for encoding in ("ordinal", "target"):
        with Regression() as reg:
            reg.random_state = 0
            reg.categorical_encoding = encoding
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.set(X[["shop", "temp"]], y, [LinearRegression()])
            result, = reg.k_folds(4)
            scores[encoding] = result.r2
    assert scores["target"] > scores["ordinal"] + 0.2, scores


def test_id_columns_are_dropped_with_a_warning_unless_declared():
    X, y = shops()
    X = X.assign(order_id=[f"o{i}" for i in range(len(X))]).drop(columns="date")
    with Regression() as reg:
        with pytest.warns(UserWarning, match="id-like"):
            reg.set(X, y, [Ridge()])
        assert "order_id" not in list(reg.features.columns)
        reg.set(X, y, [Ridge()], categorical=["order_id"])
        assert "order_id" in list(reg.features.columns)
        reg.set(X, y, [Ridge()], drop_ids=False)
        assert "order_id" in list(reg.features.columns)


def test_saved_model_reads_dates_and_ordinals(tmp_path):
    X, y = shops()
    with Regression() as reg:
        reg.random_state = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.set(X, y, [RandomForestRegressor(50, random_state=0)], ordinal={"size": ["S", "M", "L"]})
        reg.k_folds(3)
        path = reg.filter_by(FilteringCriteriaRegression.r2)[0].save_model(tmp_path)
    model = load(path)
    new = pd.DataFrame({"shop": ["shop_05", "shop_99"], "size": ["L", "XL"],
                        "date": ["2024-12-01", None], "temp": [10.0, None]})
    new["date"] = pd.to_datetime(new["date"])
    pred = model.predict(new)
    assert pred.shape == (2,) and np.isfinite(pred).all()


# ---------------------------------------------------------------- FeatureResearch
def test_feature_research_scores_categorical_and_date_columns():
    X, y = shops()
    research = FeatureResearch(X, y, task="regression", random_state=0)
    assert research.code_features == ["shop", "size"]
    mi = research.mutual_information()
    assert set(mi.index) == {"shop", "size", "date", "temp"}
    assert mi["shop"] > mi["temp"]
    importance = research.model_importance()
    assert set(importance.index) == {"shop", "size", "date", "temp"}
    assert research.skipped == ["shop", "size", "date"]  # still left out of the purely statistical methods


# ---------------------------------------------------------------- config
def test_config_with_ordinal_and_date_columns(tmp_path):
    X, y = shops()
    X.assign(sales=y, date=X["date"].dt.strftime("%Y-%m-%d")).to_csv(tmp_path / "shops.csv", index=False)
    raw = {
        "data": {"path": "shops.csv", "target": "sales", "task": "regression"},
        "columns": {"ordinal": {"size": ["S", "M", "L"]}, "datetime": ["date"], "max_onehot": 20},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
        "models": [{"name": "Ridge"}, {"name": "RandomForestRegressor", "params": {"n_estimators": 30}}],
        "output": {"dir": "runs", "keep": 1, "report": False},
    }
    (tmp_path / "exp.json").write_text(json.dumps(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["run", str(tmp_path / "exp.json")]) == 0
    board = pd.read_csv(next((tmp_path / "runs").iterdir()) / "leaderboard.csv")
    assert board.loc[0, "r2"] > 0.7
