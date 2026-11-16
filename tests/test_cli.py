"""fitfleet init / run / predict and config validation."""
import json

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

import fitfleet
from fitfleet.cli import main
from fitfleet.config import ConfigError, Experiment, find_estimator, new_run_folder
from fitfleet.utils import Sound


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
    from fitfleet.base import BaseML
    before = BaseML._process_limit
    assert main(["run", str(write(tmp_path, config(run={"processes": 1})))]) == 0
    assert BaseML._process_limit == before  # [run] processes applies to this run only
    (folder,) = (tmp_path / "runs").iterdir()
    board = pd.read_csv(folder / "leaderboard.csv")
    assert len(board) == 3  # LogisticRegression x 2 values of C + DecisionTree
    assert list(board["rank"]) == [1, 2, 3]
    assert board["accuracy"].is_monotonic_decreasing
    saved = sorted(folder.glob("*/model.joblib"))
    assert len(saved) == 2 and (folder / "config.json").exists()
    model = fitfleet.load(saved[0])
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
    ({"models": []}, "add at least one"),
    ({"models": [{"params": {}}]}, "needs a name"),
    ({"search": {"n_iter": 0}}, "n_iter must be a whole number"),
    ({"search": {"n_iter": "3"}}, "n_iter must be a whole number"),
    ({"search": {"min_rows": 1}}, "min_rows must be a whole number >= 2"),
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


@pytest.mark.parametrize("key", ["path", "target", "task"])
def test_data_section_needs_every_key(tmp_path, key):
    raw = config()
    del raw["data"][key]
    with pytest.raises(ConfigError, match=rf"\[data\] needs '{key}'"):
        Experiment.from_dict(raw, base=tmp_path)


@pytest.mark.parametrize("name, text", [("exp.json", '{"data": '), ("exp.toml", "[data\npath = 1")])
def test_malformed_config_file_is_a_config_error(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        Experiment.from_file(path)
    assert main(["run", str(path)]) == 2


def test_more_data_problems_are_reported(tmp_path, iris_csv):
    df = pd.read_csv(iris_csv)
    with pytest.raises(ConfigError, match="feature columns not in the data"):
        Experiment.from_dict(config(data={"features": ["petal width (cm)", "nope"]}), base=tmp_path).load_data()

    df.assign(**{"sepal length (cm)": df["sepal length (cm)"].mask(df.index == 3)}).to_csv(
        tmp_path / "holes.csv", index=False)
    with pytest.raises(ConfigError, match="missing values in"):
        Experiment.from_dict(config(data={"path": "holes.csv"}), base=tmp_path).load_data()

    (tmp_path / "iris.txt").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported data file type"):
        Experiment.from_dict(config(data={"path": "iris.txt"}), base=tmp_path).load_data()


def test_rows_without_a_target_are_dropped(tmp_path, iris_csv):
    df = pd.read_csv(iris_csv)
    df.loc[[0, 5, 9], "species"] = None
    df.to_csv(tmp_path / "unlabelled.csv", index=False)
    X, y = Experiment.from_dict(config(data={"path": "unlabelled.csv"}), base=tmp_path).load_data()
    assert len(X) == len(y) == len(df) - 3 and y.notna().all()
    assert not {0, 5, 9} & set(X.index)


def test_find_estimator_by_import_path():
    assert find_estimator("sklearn.linear_model.Ridge").__name__ == "Ridge"
    with pytest.raises(ConfigError, match="cannot import"):
        find_estimator("nolib.Model")


@pytest.mark.parametrize("output, expected", [
    ({}, Sound.none),
    ({"sound": "chime"}, Sound.chime),
    ({"sound": "none"}, Sound.none),
    ({"ping": True}, Sound.ping),     # older config key still works
    ({"ping": False}, Sound.none),
])
def test_sound_setting_is_parsed(tmp_path, output, expected):
    assert Experiment.from_dict(config(output=output), base=tmp_path).sound is expected


def test_sound_plays_at_end_of_experiment(tmp_path, iris_csv, monkeypatch):
    calls = []
    monkeypatch.setattr("fitfleet.base.BaseML.play_sound", staticmethod(calls.append))
    Experiment.from_dict(config(output={"sound": "chime"}), base=tmp_path).run(log=lambda *_: None)
    assert calls == [Sound.chime]


@pytest.mark.parametrize("output, message", [
    ({"sound": "trumpet"}, "sound must be one of"),
    ({"ping": "yes"}, "ping must be true or false"),
    ({"ping": True, "sound": "chime"}, "use either sound or ping, not both"),
])
def test_bad_sound_setting(tmp_path, output, message):
    with pytest.raises(ConfigError, match=message):
        Experiment.from_dict(config(output=output), base=tmp_path)


def test_every_sound_plays_without_error(monkeypatch):
    import io
    import wave
    from fitfleet.base import BaseML
    played = []
    monkeypatch.setattr("fitfleet.base._play_wav", lambda data: played.append(data) or True)
    for sound in Sound:
        BaseML.play_sound(sound)
        data = sound.wav_bytes()
        if sound in (Sound.none, Sound.bell):
            assert data == b""
            continue
        with wave.open(io.BytesIO(data)) as f:  # a real, non-empty WAV
            assert f.getnframes() > 0 and f.getframerate() == 44_100
    assert len(played) == len(Sound) - 2


def _tiny_run(**kwargs):
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True, as_frame=True)
    run = fitfleet.Classification(**kwargs)
    run.set(X, y, [LogisticRegression(max_iter=500)])
    return run


@pytest.fixture
def sound_calls(monkeypatch):
    from fitfleet.base import BaseML
    calls = []
    monkeypatch.setattr(BaseML, "play_sound", staticmethod(calls.append))
    return calls


def test_sound_plays_once_when_with_block_ends(sound_calls):
    with _tiny_run(sound="chime") as run:
        run.train_test_split(0.8)
        run.k_folds(3)
        assert sound_calls == []            # not while still in scope
    assert sound_calls == [Sound.chime]     # once, even after two runs


def test_sound_plays_when_garbage_collected(sound_calls):
    import gc
    run = _tiny_run(sound=Sound.ping)
    run.k_folds(3)
    run.close()
    del run
    gc.collect()
    assert sound_calls == [Sound.ping]


def test_no_sound_without_a_finished_run(sound_calls):
    with _tiny_run(sound="alarm"):
        pass
    assert sound_calls == []


def test_runs_in_the_same_second_get_their_own_folder(tmp_path, monkeypatch):
    """Two runs started within one second used to crash with FileExistsError."""
    import datetime as real_datetime
    import fitfleet.config as config_module

    class FrozenDatetime(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(config_module.datetime, "datetime", FrozenDatetime)
    folders = [new_run_folder(tmp_path / "runs") for _ in range(3)]
    assert [f.name for f in folders] == ["2026-01-02_03-04-05", "2026-01-02_03-04-05_2", "2026-01-02_03-04-05_3"]
    assert all(f.is_dir() for f in folders)
