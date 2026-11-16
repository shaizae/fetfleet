"""Free-text columns, polars input, LightGBM / XGBoost on mixed data, encoding as a search dimension."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from fitfleet.classification import Classification
from fitfleet.cli import main
from fitfleet.encoding import build_encoder
from fitfleet.feature_research import FeatureResearch
from fitfleet.model import load
from fitfleet.schema import ColumnKind, ColumnSchema, looks_like_text
from fitfleet.utils import FilteringCriteriaClassification

GOOD = ["great product works perfectly", "love it excellent quality", "fantastic value very happy",
        "works great highly recommend", "excellent build quality love it"]
BAD = ["broke after one day", "terrible quality do not buy", "stopped working very disappointed",
       "waste of money broke fast", "awful product returned it"]


def reviews(n=300, seed=0):
    """Classification from a free-text review (the signal), a category and a noisy number."""
    r = np.random.default_rng(seed)
    good = r.random(n) < 0.5
    text = [f"{r.choice(GOOD if g else BAD)} {r.choice(['honestly', 'overall', 'so far', 'really'])} #{i}"
            for i, g in enumerate(good)]
    flip = r.random(n) < 0.05
    label = np.where(good ^ flip, "positive", "negative")
    X = pd.DataFrame({"review": text, "store": r.choice(["web", "shop"], n), "price": r.normal(20, 5, n)})
    X.loc[r.random(n) < 0.05, "review"] = None
    return X, pd.Series(label)


# ---------------------------------------------------------------- text
def test_free_text_is_detected_and_embedded():
    X, _ = reviews()
    assert looks_like_text(X["review"]) and not looks_like_text(X["store"])
    s = ColumnSchema.infer(X)
    assert s.kinds == [ColumnKind.text, ColumnKind.categorical, ColumnKind.numeric]
    assert s.missing == []  # a missing review is an empty text, not NaN
    width = s.width(0)
    assert 2 <= width <= 30 and s.n_matrix == width + 2
    assert s.matrix_index([1, 2]) == [width, width + 1]
    assert s.matrix_names([0])[:2] == ["review_text0", "review_text1"]
    out = s.encode(X)
    assert out.shape == (len(X), s.n_matrix) and np.isfinite(out[:, :width]).all()


def test_text_can_be_declared_and_short_text_is_categorical():
    X = pd.DataFrame({"name": ["red shoe", "blue shoe", "red hat"] * 10})
    assert ColumnSchema.infer(X).kinds == [ColumnKind.categorical]  # few words, few values
    assert ColumnSchema.infer(X, text=["name"]).kinds == [ColumnKind.text]


def test_encoder_positions_follow_the_text_width():
    X, _ = reviews()
    s = ColumnSchema.infer(X)
    w = s.width(0)
    enc, _ = build_encoder(s, [0, 1], LogisticRegression(), [])
    parts = {name: cols for name, _, cols in enc.transformers}
    assert parts["text"] == list(range(w)) and parts["categorical_onehot"] == [w]
    enc, _ = build_encoder(s, [1, 2], LogisticRegression(), [])
    assert {name: cols for name, _, cols in enc.transformers} == {"categorical_onehot": [0], "numeric": [1]}


@pytest.mark.parametrize("model", [LogisticRegression(max_iter=1000), RandomForestClassifier(100, random_state=0)])
def test_models_learn_from_text(model):
    X, y = reviews()
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [model])
        cls.features.standard_scaler()
        result, = cls.k_folds(3)
    assert result.accuracy > 0.85, result.accuracy


def test_text_is_one_column_for_feature_search_and_saved_models(tmp_path):
    X, y = reviews()
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [LogisticRegression(max_iter=1000)])
        cls.brute_force_features()
        assert len(cls._features_index) == 2 ** 3 - 1
        results = cls.k_folds(3)
        best = max(results, key=lambda r: (r.accuracy, -len(r.features_index)))
        assert 0 in best.features_index  # the review
        assert cls.experiment_report().leaderboard.iloc[0]["features"].startswith("review")
        path = cls.filter_by(FilteringCriteriaClassification.accuracy)[0].save_model(tmp_path)
    model = load(path)
    new = pd.DataFrame({"review": ["excellent quality love it", "broke after a day, awful", None],
                        "store": ["web", "outlet", None], "price": [20.0, None, 15.0]})
    assert list(model.predict(new)[:2]) == ["positive", "negative"]


def test_feature_selection_scores_text_as_one_column():
    X, y = reviews(400)
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X[["review", "price"]], y, [LogisticRegression(max_iter=1000)])
        cls.feature_selection(f_classif, [1, 2])
        assert [list(i) for i in cls._features_index] == [[0], [0, 1]]
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [LogisticRegression(max_iter=1000)])
        cls.feature_selection(mutual_info_classif, [1])
        assert [list(i) for i in cls._features_index] == [[0]]


def test_text_columns_are_not_dropped_as_ids():
    X, y = reviews()
    with Classification() as cls:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "id-like" warning
            cls.set(X, y, [LogisticRegression()])
        assert "review" in list(cls.features.columns)


# ---------------------------------------------------------------- encoding as a search dimension
def test_encoding_is_searched_like_a_hyper_parameter():
    X, y = reviews()
    X = X.drop(columns="review").assign(city=np.random.default_rng(1).choice([f"c{i}" for i in range(50)], len(X)))
    with Classification() as cls:
        cls.random_state = 0
        cls.set(X, y, [LogisticRegression(max_iter=500), RandomForestClassifier(30, random_state=0)])
        cls.set_hyper_parameters({"categorical_encoding": ["onehot", "target"], "C": [1.0]})
        results = cls.k_folds(3)
        assert len(results) == 4  # 2 models x 2 encodings
        assert sorted(r.params["categorical_encoding"] for r in results) == ["onehot", "onehot", "target", "target"]
        board = cls.experiment_report(all_params=False).leaderboard
        assert board["params"].str.contains("categorical_encoding").all()
        target_lr = next(r for r in results if r.name == "LogisticRegression"
                         and r.params["categorical_encoding"] == "target")
        fitted, = cls.refit([target_lr])
        assert "categorical_target" in fitted.model.named_steps["encode"].named_transformers_
    with pytest.raises(ValueError, match="categorical_encoding values"):
        cls.set_hyper_parameters({"categorical_encoding": ["bogus"]})


def test_encoding_search_needs_a_mixed_table():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] > 0).astype(int))
    with Classification() as cls:
        cls.set(X, y, [LogisticRegression()])
        cls.set_hyper_parameters({"categorical_encoding": ["onehot"]})
        with pytest.raises(ValueError, match="purely numeric"):
            cls.k_folds(3)


def test_config_searches_encodings_and_reads_text(tmp_path):
    X, y = reviews()
    X.assign(label=y).to_csv(tmp_path / "reviews.csv", index=False)
    raw = {
        "data": {"path": "reviews.csv", "target": "label", "task": "classification"},
        "columns": {"text": ["review"]},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
        "models": [{"name": "LogisticRegression", "params": {"max_iter": 1000}}],
        "hyper_parameters": {"categorical_encoding": ["onehot", "ordinal"]},
        "output": {"dir": "runs", "keep": 1, "report": False},
    }
    (tmp_path / "exp.json").write_text(json.dumps(raw), encoding="utf-8")
    assert main(["run", str(tmp_path / "exp.json")]) == 0
    board = pd.read_csv(next((tmp_path / "runs").iterdir()) / "leaderboard.csv")
    assert len(board) == 2 and board.loc[0, "accuracy"] > 0.85
    raw["hyper_parameters"] = {"categorical_encoding": ["nope"]}
    (tmp_path / "bad.json").write_text(json.dumps(raw), encoding="utf-8")
    assert main(["run", str(tmp_path / "bad.json")]) == 2


# ---------------------------------------------------------------- polars, LightGBM, XGBoost
def test_polars_input():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    X, y = reviews()
    with Classification() as cls:
        cls.random_state = 0
        cls.set(pl.from_pandas(X), pl.from_pandas(y), [RandomForestClassifier(30, random_state=0)])
        result, = cls.k_folds(3)
        model = cls.filter_by(FilteringCriteriaClassification.accuracy)[0].to_model()
    assert result.accuracy > 0.8
    assert len(model.predict(pl.from_pandas(X.head(3)))) == 3
    research = FeatureResearch(pl.from_pandas(X.drop(columns="review")), pl.from_pandas(y))
    assert "store" in research.mutual_information().index


@pytest.mark.parametrize("module, name", [("lightgbm", "LGBMClassifier"), ("xgboost", "XGBClassifier")])
def test_gradient_boosting_libraries_on_mixed_data(module, name):
    lib = pytest.importorskip(module)
    X, y = reviews()
    X = X.drop(columns="review").assign(tier=np.where(y == "positive", "gold", "basic"))
    X.loc[::7, "tier"] = None
    model = getattr(lib, name)(n_estimators=30, **({"verbose": -1} if module == "lightgbm" else {}))
    enc, _ = build_encoder(ColumnSchema.infer(X), None, model, [])
    assert enc.transformers[0][1] == "passthrough"  # codes and NaN as they are: both take NaN
    with Classification() as cls:
        cls.set(X, y, [model])
        result, = cls.k_folds(3)
    assert result.accuracy > 0.8
