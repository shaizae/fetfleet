"""Smarter alternatives to trying every configuration.

A *candidate* is one model x feature subset x hyper-parameter combination. The
strategies here decide which candidates to train and on how many rows; BaseML
trains each batch in the process pool, so every round still uses every core.

- grid:     every candidate (the default, same as before).
- random:   a random sample of `n_iter` candidates from the full grid.
- halving:  successive halving. Every candidate is trained on a small share of the
            rows; the best 1/factor go on to a round with factor x more rows, until
            the last round, which uses all rows. Can start from a random sample.
- forward / backward feature selection: a greedy alternative to brute_force_features().
            Per model x hyper-parameter combination, add (or remove) the one feature
            that helps most, until no step improves the score. n^2/2 subsets, not 2^n.

`Evaluate` is evaluate(candidates, rows, description) -> results, each with
.candidate_id = its index in `candidates`; rows are the row indexes to use (None = all rows).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from ml_pipe.results.base import BaseResults
    from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression

STRATEGIES = ("grid", "random", "halving")
DIRECTIONS = ("forward", "backward")


class Candidate(NamedTuple):
    """One configuration to train.

    `model` is an index into the run's models, `features` the column indexes (None = every
    column) and `params` the hyper-parameters set on the model (only those it has).
    """
    model: int
    features: tuple[int, ...] | None
    params: dict


@dataclass
class SearchSettings:
    """How to pick the candidates to train (see the module docstring for the strategies).

    `n_iter` is the sample size for random search and an optional sample to start from for
    halving. Halving keeps 1/`factor` of the candidates per round and gives the next round
    `factor` x more rows; `min_rows` is the rows in its first round (default: from the data).
    `criteria` is the metric used to compare candidates between rounds.
    """
    strategy: str = "grid"
    n_iter: int | None = None
    factor: float = 3.0
    min_rows: int | None = None
    criteria: object = None

    def validate(self) -> None:
        """Raise ValueError if a setting is out of range or missing for the strategy."""
        if self.strategy not in STRATEGIES:
            raise ValueError(f"search strategy must be one of {list(STRATEGIES)}, not {self.strategy!r}")
        if self.strategy == "random" and self.n_iter is None:
            raise ValueError("random search needs n_iter (how many configurations to try)")
        if self.n_iter is not None and (not isinstance(self.n_iter, int) or self.n_iter < 1):
            raise ValueError("n_iter must be a whole number >= 1")
        if not self.factor > 1:
            raise ValueError("halving factor must be greater than 1")
        if self.min_rows is not None and (not isinstance(self.min_rows, int) or self.min_rows < 2):
            raise ValueError("min_rows must be a whole number >= 2")


@dataclass
class FeatureSearch:
    """Settings for greedy feature selection.

    `direction` is "forward" or "backward". Forward stops at `max_features` features and a
    step must improve the score by more than `tol`; backward stops at `min_features`.
    """
    direction: str
    max_features: int | None = None
    min_features: int = 1
    tol: float = 0.0

    def validate(self) -> None:
        """Raise ValueError if a setting is out of range."""
        if self.direction not in DIRECTIONS:
            raise ValueError(f"feature search direction must be one of {list(DIRECTIONS)}")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be >= 1")
        if self.min_features < 1:
            raise ValueError("min_features must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be >= 0")


Evaluate = Callable[[list[Candidate], "np.ndarray | None", str], list]


def score(result: BaseResults, criteria: FilteringCriteriaClassification | FilteringCriteriaRegression) -> float:
    """Higher is better, whatever the metric."""
    value = float(getattr(result, criteria.value))
    if math.isnan(value):
        return -math.inf
    return value if criteria.maximize else -value


def rank(results: list[BaseResults], criteria: FilteringCriteriaClassification | FilteringCriteriaRegression,
         n_columns: int) -> list[BaseResults]:
    """Best first; ties go to the configuration with fewer features (the simpler model)."""
    def key(r: BaseResults) -> tuple[float, int]:
        """Sort key: negated score, then number of features (None = all `n_columns`)."""
        n = len(r.features_index) if r.features_index is not None else n_columns
        return -score(r, criteria), n

    return sorted(results, key=key)


def hyper_parameter_combinations(model: BaseEstimator, grid: dict[str, list]) -> list[dict[str, Any]]:
    """Every combination of the grid values this model has a parameter for ([{}] if none)."""
    params = model.get_params(deep=False) if hasattr(model, "get_params") else vars(model)
    keys = [k for k in grid if k in params]
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


class CandidateSpace:
    """models x feature sets x hyper-parameter combinations, without building them all:
    random search draws indexes into the space, so 2^n feature subsets never sit in memory."""

    def __init__(self, models: list[BaseEstimator], feature_sets: list[Sequence[int] | None],
                 grid: dict[str, list]) -> None:
        """Index the space; `feature_sets` are column index lists (None = every column)."""
        self.feature_sets = [tuple(int(i) for i in f) if f is not None else None for f in feature_sets]
        self.params = [hyper_parameter_combinations(m, grid) for m in models]
        self.sizes = [len(self.feature_sets) * len(p) for p in self.params]

    def __len__(self) -> int:
        """Number of candidates in the space."""
        return sum(self.sizes)

    def __getitem__(self, index: int) -> Candidate:
        """The candidate at `index`, ordered by model, then feature set, then hyper-parameters."""
        for model, size in enumerate(self.sizes):
            if index < size:
                n_params = len(self.params[model])
                return Candidate(model, self.feature_sets[index // n_params], self.params[model][index % n_params])
            index -= size
        raise IndexError(index)

    def all(self) -> list[Candidate]:
        """Every candidate, in index order."""
        return [self[i] for i in range(len(self))]

    def sample(self, n: int, rng: np.random.Generator) -> list[Candidate]:
        """`n` distinct candidates drawn with `rng`, in index order (all of them if n >= the space)."""
        if n >= len(self):
            return self.all()
        picks = rng.choice(len(self), size=n, replace=False)
        return [self[int(i)] for i in np.sort(picks)]

    def tracks(self) -> list[tuple[int, dict]]:
        """(model, hyper-parameters) pairs: the starting points of a greedy feature search."""
        return [(m, p) for m, combos in enumerate(self.params) for p in combos]


def halving_rounds(n_candidates: int, n_rows: int, min_rows: int, factor: float) -> list[int]:
    """Rows to use in each round; the last round uses all rows.

    Rounds stop while a few candidates (at most `factor`) are left for the last round,
    so it still compares several; the first round gets at least min_rows rows."""
    if n_candidates <= 1 or n_rows <= min_rows:
        return [n_rows]
    by_rows = int(math.floor(math.log(n_rows / min_rows, factor) + 1e-9))
    by_candidates = max(0, int(math.ceil(math.log(n_candidates, factor) - 1e-9)) - 1)
    n_rounds = 1 + min(by_rows, by_candidates)
    rows = [max(min_rows, int(n_rows / factor ** (n_rounds - 1 - i))) for i in range(n_rounds)]
    rows[-1] = n_rows
    return rows


def successive_halving(evaluate: Evaluate, candidates: list[Candidate], subsample: Callable[[int], np.ndarray],
                       n_rows: int, min_rows: int, factor: float,
                       criteria: FilteringCriteriaClassification | FilteringCriteriaRegression, n_columns: int,
                       log: list[dict[str, int]]) -> list[BaseResults]:
    """subsample(n) -> the row indexes of one round. Returns the last round's results
    (all of them trained on every row, so they compare fairly with each other).
    A summary of each round is appended to `log`."""
    schedule = halving_rounds(len(candidates), n_rows, min_rows, factor)
    results = []
    for i, rows_in_round in enumerate(schedule, start=1):
        last = i == len(schedule)
        rows = None if last else subsample(rows_in_round)
        desc = f"halving {i}/{len(schedule)} ({len(candidates)} configs, {rows_in_round} rows)"
        results = evaluate(candidates, rows, desc)
        log.append({"round": i, "rows": rows_in_round, "configurations": len(candidates)})
        if last:
            break
        keep = max(1, math.ceil(len(candidates) / factor))
        ranked = rank(results, criteria, n_columns)[:keep]
        candidates = [candidates[r.candidate_id] for r in ranked]
    return results


def greedy_features(evaluate: Evaluate, tracks: list[tuple[int, dict]], n_columns: int, search: FeatureSearch,
                    criteria: FilteringCriteriaClassification | FilteringCriteriaRegression,
                    log: list[dict[str, int]]) -> list[BaseResults]:
    """Sequential feature selection, run for every (model, hyper-parameters) track at once:
    each round trains every track's next-step subsets in one pool batch.

    forward:  start empty; add the feature that raises the score most; stop when no
              addition improves it by more than tol, or at max_features.
    backward: start with every column; drop the feature whose removal leaves the best
              score; stop when every removal lowers the score by more than tol below the
              best seen, or at min_features.
    tol only decides when to stop. Every subset trained is returned (all on every row),
    including the last step's, so the leaderboard ranks them all by score.

    Backward first scores the full set, which removals are compared with; comparing with
    the best score so far (not the previous step's) means small losses can't add up past
    tol. Tracks move in step, so every subset in a round has the same size. A summary of
    each round is appended to `log`."""
    forward = search.direction == "forward"
    everything = tuple(range(n_columns))
    current = {t: () if forward else everything for t in range(len(tracks))}
    best = {t: -math.inf for t in current}
    active = set(current)
    all_results = []

    if not forward:
        start = [Candidate(m, everything, p) for m, p in tracks]
        results = evaluate(start, None, "backward: all features")
        for r in results:
            best[r.candidate_id] = score(r, criteria)
        all_results += results
        log.append({"round": 0, "features": n_columns, "configurations": len(start)})

    limit = min(search.max_features or n_columns, n_columns)
    step = 0
    while active:
        step += 1
        candidates, owner = [], []
        for t in sorted(active):
            subset = current[t]
            if forward:
                if len(subset) >= limit:
                    continue
                options = [tuple(sorted(subset + (j,))) for j in everything if j not in subset]
            else:
                if len(subset) <= search.min_features:
                    continue
                options = [tuple(i for i in subset if i != j) for j in subset]
            m, p = tracks[t]
            candidates += [Candidate(m, s, p) for s in options]
            owner += [t] * len(options)
        if not candidates:
            break
        size = len(candidates[0].features)
        results = evaluate(candidates, None, f"{search.direction} step {step} ({size} features)")
        all_results += results
        log.append({"round": step, "features": size, "configurations": len(candidates)})

        step_best: dict[int, tuple[float, int]] = {}
        for r in results:
            t, s = owner[r.candidate_id], score(r, criteria)
            if t not in step_best or s > step_best[t][0]:
                step_best[t] = (s, r.candidate_id)
        for t in list(active):
            if t not in step_best:
                active.discard(t)
                continue
            s, cid = step_best[t]
            if forward and s > best[t] + search.tol:
                current[t], best[t] = candidates[cid].features, s
            elif not forward and s >= best[t] - search.tol:
                current[t], best[t] = candidates[cid].features, max(s, best[t])
            else:
                active.discard(t)
    return all_results
