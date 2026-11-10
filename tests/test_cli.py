"""ml-pipe init / run / predict and config validation."""
import json

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

import ml_pipe
from ml_pipe.cli import main
from ml_pipe.config import ConfigError, Experiment, find_estimator


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = data.data.assign(species=data.target.map(dict(enumerate(data.target_names))))
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    return path


def config(**overrides) -> dict:
    raw = {
        "data": {"path": "iris.csv", "target": "species", "task": "classification"},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
        "preprocessing": ["standard_scaler"],
        "models": [{"name": "LogisticRegression", "params": {"max_iter": 500}}, {"name": "DecisionTreeClassifier"}],
        "hyper_parameters": {"C": [0.1, 1]},
        "output": {"dir": "runs", "keep": 2, "report": False},
    }
    for section, values in overrides.items():
        raw[section] = values if not isinstance(values, dict) else {**raw.get(section, {}), **values}
    return raw


def write(tmp_path, raw, name="exp.json"):
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_init_writes_a_valid_config(tmp_path, iris_csv):
    path = tmp_path / "experiment.toml"
    assert main(["init", str(path)]) == 0
    assert main(["init", str(path)]) == 1  # no silent overwrite
    text = path.read_text(encoding="utf-8").replace("data.csv", "iris.csv").replace('"label"', '"species"')
    path.write_text(text, encoding="utf-8")
    assert main(["run", str(path), "--check"]) == 0


def test_run_writes_leaderboard_and_best_models(tmp_path, iris_csv, capsys):
    assert main(["run", str(write(tmp_path, config()))]) == 0
    (folder,) = (tmp_path / "runs").iterdir()
    board = pd.read_csv(folder / "leaderboard.csv")
    assert len(board) == 3  # LogisticRegression x 2 values of C + DecisionTree
    assert list(board["rank"]) == [1, 2, 3]
    assert board["accuracy"].is_monotonic_decreasing
    saved = sorted(folder.glob("*/model.joblib"))
    assert len(saved) == 2 and (folder / "config.json").exists()
    model = ml_pipe.load(saved[0])
    assert model.metrics["accuracy"] == pytest.approx(board["accuracy"][0])
    assert "configurations" in capsys.readouterr().out


def test_predict_command(tmp_path, iris_csv, capsys):
    main(["run", str(write(tmp_path, config(output={"keep": 1})))])
    (model,) = (tmp_path / "runs").glob("*/*/model.joblib")
    out = tmp_path / "pred.csv"
    assert main(["predict", str(model), str(iris_csv), "-o", str(out), "--proba"]) == 0
    pred = pd.read_csv(out)
    assert list(pred.columns) == ["prediction", "proba_setosa", "proba_versicolor", "proba_virginica"]
    assert (pred["prediction"] == pd.read_csv(iris_csv)["species"]).mean() > 0.9


def test_regression_train_test_split_with_selection(tmp_path):
    data = load_diabetes(as_frame=True)
    data.frame.to_csv(tmp_path / "diabetes.csv", index=False)
    raw = config(data={"path": "diabetes.csv", "target": "target", "task": "regression"},
                 run={"validation": "train_test_split", "train_ratio": 0.75, "select_by": "rmse"},
                 models=[{"name": "Ridge"}], hyper_parameters={"alpha": [0.1, 1.0]},
                 feature_selection={"method": "f_regression", "k": [3, 6]})
    folder = Experiment.from_dict(raw, base=tmp_path).run(log=lambda *_: None)
    board = pd.read_csv(folder / "leaderboard.csv")
    assert len(board) == 4 and board["rmse"].is_monotonic_increasing
    assert set(board["n_features"]) == {3, 6}


@pytest.mark.parametrize("override, message", [
    ({"data": {"task": "clustering"}}, "task must be"),
    ({"run": {"validation": "bootstrap"}}, "validation must be"),
    ({"run": {"select_by": "r2"}}, "select_by"),
    ({"models": [{"name": "RandomForestClasifier"}]}, "Did you mean RandomForestClassifier"),
    ({"models": [{"name": "Ridge", "alpha": 1}]}, "under params"),
    ({"models": [{"name": "Ridge", "params": {"nope": 1}}]}, "nope"),
    ({"preprocessing": ["robust_scaler"]}, "preprocessing steps"),
    ({"feature_selection": {"method": "chi3", "k": [2]}}, "chi3"),
    ({"feature_selection": {"method": "chi2"}}, "needs k"),
    ({"hyper_parameters": {"C": 1}}, "non-empty list"),
    ({"output": {"keep": 0}}, "keep"),
    ({"extra": {}}, "unknown section"),
])
def test_bad_configs_are_rejected_before_training(tmp_path, override, message):
    with pytest.raises(ConfigError, match=message):
        Experiment.from_dict(config(**override), base=tmp_path)


def test_data_problems_are_reported(tmp_path, iris_csv):
    with pytest.raises(ConfigError, match="target column"):
        Experiment.from_dict(config(data={"target": "kind"}), base=tmp_path).load_data()
    pd.read_csv(iris_csv).assign(color="red").to_csv(tmp_path / "text.csv", index=False)
    with pytest.raises(ConfigError, match="non-numeric"):
        Experiment.from_dict(config(data={"path": "text.csv"}), base=tmp_path).load_data()
    assert main(["run", str(write(tmp_path, config(data={"path": "missing.csv"})))]) == 2


def test_find_estimator_by_import_path():
    assert find_estimator("sklearn.linear_model.Ridge").__name__ == "Ridge"
    with pytest.raises(ConfigError, match="cannot import"):
        find_estimator("nolib.Model")


def test_processes_setting_does_not_leak(tmp_path, iris_csv):
    from ml_pipe.base import BaseML
    before = BaseML._process_limit
    Experiment.from_dict(config(run={"processes": 1}), base=tmp_path).run(log=lambda *_: None)
    assert BaseML._process_limit == before


@pytest.mark.parametrize("ping", [True, False])
def test_ping_at_end_of_experiment(tmp_path, iris_csv, monkeypatch, ping):
    calls = []
    monkeypatch.setattr("ml_pipe.config.play_ping", lambda: calls.append(1))
    exp = Experiment.from_dict(config(output={"ping": ping}), base=tmp_path)
    exp.run(log=lambda *_: None)
    assert calls == ([1] if ping else [])


def test_ping_must_be_boolean(tmp_path):
    with pytest.raises(ConfigError, match="ping"):
        Experiment.from_dict(config(output={"ping": "yes"}), base=tmp_path)
