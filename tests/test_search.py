"""Random search, successive halving and greedy (forward / backward) feature selection."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_pipe import Classification, FilteringCriteriaClassification, FilteringCriteriaRegression, Regression
from ml_pipe.config import ConfigError, Experiment
from ml_pipe.search import Candidate, CandidateSpace, halving_rounds
from ml_pipe.utils import create_shared_memory, SharedMemory
from ml_pipe.train_data import KFoldsTrainData
from ml_pipe.results.classification import ClassificationResults
from ml_pipe.workers import _fold_splits


# ---------------------------------------------------------------------- pure helpers
def test_halving_rounds_end_with_all_rows():
    rows = halving_rounds(n_candidates=27, n_rows=900, min_rows=30, factor=3)
    assert rows == [100, 300, 900]  # 27 -> 9 -> 3 candidates: 3 rounds, the first as large as allowed
    assert halving_rounds(3, 900, 30, 3) == [900]  # already down to `factor`: one round
    assert halving_rounds(4, 900, 30, 3) == [300, 900]
    assert halving_rounds(1, 900, 30, 3) == [900]  # one candidate: nothing to eliminate
    assert halving_rounds(100, 50, 30, 3) == [50]  # too few rows for a smaller round


def test_halving_rounds_limited_by_rows():
    rows = halving_rounds(n_candidates=1000, n_rows=270, min_rows=30, factor=3)
    assert rows == [30, 90, 270]
    assert all(b > a for a, b in zip(rows, rows[1:]))


def test_candidate_space_matches_grid_and_samples_without_repeats():
    models = [LogisticRegression(), DecisionTreeClassifier()]
    space = CandidateSpace(models, [[0], [1], [0, 1]], {"C": [0.1, 1, 10], "max_depth": [2, 4]})
    assert len(space) == 3 * 3 + 3 * 2
    everything = space.all()
    assert everything[0] == Candidate(0, (0,), {"C": 0.1})
    assert everything[-1] == Candidate(1, (0, 1), {"max_depth": 4})
    sample = space.sample(7, np.random.default_rng(0))
    assert len(sample) == 7 and len({repr(c) for c in sample}) == 7
    assert all(c in everything for c in sample)
    assert space.sample(100, np.random.default_rng(0)) == everything
    assert space.tracks() == [(0, {"C": 0.1}), (0, {"C": 1}), (0, {"C": 10}), (1, {"max_depth": 2}),
                              (1, {"max_depth": 4})]


def test_fold_splits_skip_rows_left_out_of_a_round(iris_data_shared_memory):
    X, y = iris_data_shared_memory
    ids = np.full(150, -1, dtype=np.int64)
    used = np.arange(0, 150, 2)
    ids[used] = np.arange(len(used)) % 3
    folds = create_shared_memory(ids, "search_folds")
    td = KFoldsTrainData(model=None, results_type=ClassificationResults, features=X, target=y,
                         number_of_folds=3, folds=folds)
    seen = []
    for train, test in _fold_splits(td):
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(used)
        seen += list(test)
    assert sorted(seen) == list(used)
    SharedMemory.cleanup()


# ---------------------------------------------------------------------- runs
@pytest.fixture
def iris():
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)


def iris_run(iris, models=None):
    X, y = iris
    run = Classification()
    run.random_state = 0
    run.set(X, y, models or [LogisticRegression(max_iter=500), KNeighborsClassifier()])
    return run


def test_random_search_trains_n_iter_configurations(iris):
    with iris_run(iris) as run:
        run.set_hyper_parameters({"C": [0.01, 0.1, 1, 10], "n_neighbors": [1, 3, 5, 7, 9]})
        run.search("random", n_iter=4)
        results = run.k_folds(3)
        assert len(results) == 4
        params = [(r.name, json.dumps(r.model.get_params(), default=str)) for r in results]
        assert len(set(params)) == 4
        again = run.k_folds(3)  # same random_state: same sample
        assert [r.accuracy for r in again] == [r.accuracy for r in results]


def test_halving_keeps_the_best_and_ends_on_all_rows(iris):
    with iris_run(iris) as run:
        run.set_hyper_parameters({"C": [0.001, 0.01, 0.1, 1, 10], "n_neighbors": [1, 3, 5, 7]})
        run.search("halving", factor=3, min_rows=30)
        results = run.k_folds(3)
        log = run.search_log
        assert [step["configurations"] for step in log] == [9, 3]
        assert log[-1]["rows"] == 150 and log[0]["rows"] < 150
        assert len(results) == 3
        # the last round is a normal k-fold on every row: same score as training that model alone
        best = max(results, key=lambda r: r.accuracy)
        assert best.accuracy > 0.9
        assert len(run.filter_by(FilteringCriteriaClassification.accuracy)) >= 1


def test_halving_drops_a_useless_model(iris):
    X, y = iris
    noise = X.assign(**{c: np.random.default_rng(0).normal(size=len(X)) for c in X.columns})
    with Classification() as run:
        run.random_state = 0
        run.set(noise.join(X, rsuffix="_real"), y, [LogisticRegression(max_iter=500)])
        run._features_index = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1], [2, 3], [4, 5], [6, 7], [0, 4], [1, 5], [2, 6]]
        run._all_features = False
        run.search("halving", factor=3, min_rows=45)
        results = run.k_folds(3)
        assert all(any(i >= 4 for i in r.features_index) for r in results)  # survivors use real columns


def test_halving_with_train_test_split_keeps_the_test_rows(iris):
    with iris_run(iris) as run:
        run.set_hyper_parameters({"C": [0.01, 0.1, 1, 10], "n_neighbors": [1, 3, 5, 7, 9]})
        run.search("halving", n_iter=6, factor=2, min_rows=30)
        results = run.train_test_split(0.8)
        assert run.search_log[0]["configurations"] == 6
        assert run.search_log[-1]["rows"] == 120
        assert all(len(r.target_test if r.target_test is not None else r._pred_test) == 30 for r in results)
        refitted = run.refit(results[:1])[0]
        assert refitted.model.predict(iris[0].iloc[:5].to_numpy()).shape == (5,)


def test_halving_refuses_leave_one_out(iris):
    with iris_run(iris) as run:
        run.search("halving")
        with pytest.raises(ValueError, match="leave-one-out"):
            run.leave_one_out()


def test_search_validates_arguments(iris):
    run = iris_run(iris)
    with pytest.raises(ValueError, match="strategy"):
        run.search("bayes")
    with pytest.raises(ValueError, match="n_iter"):
        run.search("random")
    with pytest.raises(ValueError, match="factor"):
        run.search("halving", factor=1)
    with pytest.raises(ValueError, match="select_by"):
        run.search("halving", select_by=FilteringCriteriaRegression.r2)


@pytest.fixture
def informative():
    """6 columns: 0-2 carry the signal, 3-5 are noise."""
    X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_redundant=0, shuffle=False,
                               random_state=1)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def test_forward_selection_finds_the_informative_columns(informative):
    X, y = informative
    with Classification() as run:
        run.random_state = 0
        run.set(X, y, [LogisticRegression(max_iter=500)])
        run.forward_selection(tol=0.01)
        results = run.k_folds(5)
        # path: [2] -> [1, 2]; adding a third column gains < tol, so the search stops there
        assert [step["features"] for step in run.search_log] == [1, 2, 3]
        assert [step["configurations"] for step in run.search_log] == [6, 5, 4]
        assert len(results) == 15 < 2 ** 6 - 1  # fewer subsets than brute force
        best_pair = max((r for r in results if len(r.features_index) == 2), key=lambda r: r.accuracy)
        assert set(best_pair.features_index) == {1, 2}


def test_forward_selection_respects_max_features(informative):
    X, y = informative
    with Classification() as run:
        run.set(X, y, [LogisticRegression(max_iter=500), DecisionTreeClassifier(random_state=0)])
        run.forward_selection(max_features=2)
        results = run.k_folds(3)
        assert max(len(r.features_index) for r in results) <= 2
        assert {r.name for r in results} == {"LogisticRegression", "DecisionTreeClassifier"}


def test_backward_elimination_drops_noise(informative):
    X, y = informative
    with Classification() as run:
        run.random_state = 0
        run.set(X, y, [LogisticRegression(max_iter=500)])
        run.backward_elimination(tol=0.01)
        results = run.k_folds(5)
        assert run.search_log[0] == {"round": 0, "features": 6, "configurations": 1}
        best = max(results, key=lambda r: (r.accuracy, -len(r.features_index)))
        assert len(best.features_index) < 6
        refitted = run.refit([best])[0]
        assert refitted.model.predict(X.iloc[:3].to_numpy()).shape == (3,)


def test_forward_selection_regression_uses_r2():
    X, y = make_regression(n_samples=120, n_features=5, n_informative=2, noise=1, shuffle=False, random_state=0)
    with Regression() as run:
        run.random_state = 0  # same folds every run
        run.set(pd.DataFrame(X), pd.Series(y), [LinearRegression()])
        run.forward_selection(tol=0.001)
        results = run.k_folds(3)
    # Features 0 and 1 are the informative ones: forward selection must pick them first...
    pairs = [r for r in results if len(r.features_index) == 2]
    assert set(max(pairs, key=lambda r: r.r2).features_index) == {0, 1}
    # ...then try a third feature on top of them, see none improves r2 by more than tol, and stop.
    # (Those 3-feature tries are returned too and may beat {0, 1} by ~1e-5 r2 by chance, so the
    # single highest r2 is not asserted on.)
    assert [step["features"] for step in run.search_log] == [1, 2, 3]
    assert all({0, 1} <= set(r.features_index) for r in results if len(r.features_index) == 3)
    best = max(results, key=lambda r: r.r2)
    assert {0, 1} <= set(best.features_index)


def test_greedy_selection_cannot_mix_with_random_search(informative):
    X, y = informative
    with Classification() as run:
        run.set(X, y, [LogisticRegression()])
        run.forward_selection()
        run.search("random", n_iter=3)
        with pytest.raises(ValueError, match="can't be combined"):
            run.k_folds(3)


def test_brute_force_after_forward_selection_turns_it_off(informative):
    X, y = informative
    with Classification() as run:
        run.set(X.iloc[:, :3], y, [LogisticRegression()])
        run.forward_selection()
        run.brute_force_features()
        assert len(run.k_folds(3)) == 7


# ---------------------------------------------------------------------- config
@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    data.data.assign(species=data.target).to_csv(tmp_path / "iris.csv", index=False)
    return tmp_path / "iris.csv"


def raw_config(**extra) -> dict:
    return {
        "data": {"path": "iris.csv", "target": "species", "task": "classification"},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0, "processes": 2},
        "models": [{"name": "LogisticRegression", "params": {"max_iter": 500}}, {"name": "KNeighborsClassifier"}],
        "hyper_parameters": {"C": [0.01, 0.1, 1, 10], "n_neighbors": [1, 5, 9, 15]},
        "output": {"dir": "runs", "keep": 1, "report": False},
        **extra,
    }


def test_config_halving_run(tmp_path, iris_csv):
    exp = Experiment.from_dict(raw_config(search={"strategy": "halving", "factor": 2, "min_rows": 40}),
                               base=tmp_path)
    lines = []
    folder = exp.run(log=lines.append)
    board = pd.read_csv(folder / "leaderboard.csv")
    assert 1 <= len(board) < 8
    assert any(line.startswith("  round 1") for line in lines)


def test_config_forward_selection_run(tmp_path, iris_csv):
    exp = Experiment.from_dict(raw_config(feature_selection={"method": "forward", "max_features": 2},
                                          hyper_parameters={}), base=tmp_path)
    folder = exp.run(log=lambda *_: None)
    board = pd.read_csv(folder / "leaderboard.csv")
    assert board["n_features"].max() <= 2
    assert set(board["model"]) == {"LogisticRegression", "KNeighborsClassifier"}


@pytest.mark.parametrize("extra, message", [
    ({"search": {"strategy": "optuna"}}, "strategy"),
    ({"search": {"strategy": "random"}}, "n_iter"),
    ({"search": {"strategy": "halving", "factor": 1}}, "factor"),
    ({"search": {"strategy": "halving", "rounds": 3}}, "unknown keys"),
    ({"search": {"strategy": "halving"}, "run": {"validation": "leave_one_out"}}, "k_folds or train_test_split"),
    ({"search": {"strategy": "random", "n_iter": 3}, "feature_selection": {"method": "forward"}}, "grid"),
])
def test_config_rejects_bad_search(tmp_path, extra, message):
    raw = raw_config()
    for key, value in extra.items():
        raw[key] = {**raw.get(key, {}), **value}
    with pytest.raises(ConfigError, match=message):
        Experiment.from_dict(raw, base=tmp_path)
