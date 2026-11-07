"""Smarter alternatives to trying every configuration.

A *candidate* is one model x feature subset x hyper-parameter combination. The
strategies here decide which candidates to train and on how many rows; BaseML
trains each batch in the process pool, so every round still uses every core.

- grid:     every candidate (the default, same as before).
- random:   a random sample of `n_iter` candidates from the full grid.
- halving:  successive halving. Every candidate is trained on a small share of the
            rows; the best 1/factor go on to a round with factor x more rows, until
            the last round, which uses all rows. Can start from a random sample.
- optuna:   Bayesian optimisation (TPE) with Optuna: each batch of trials (one per
            worker) is chosen from the scores of all earlier ones. Hyper-parameters can
            be ranges (Range(0.001, 100, log=True)), not only lists. Needs optuna.
- forward / backward feature selection: a greedy alternative to brute_force_features().
            Per model x hyper-parameter combination, add (or remove) the one feature
            that helps most, until no step improves the score. n^2/2 subsets, not 2^n.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Callable, NamedTuple

import numpy as np

STRATEGIES = ("grid", "random", "halving", "optuna")
DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class Range:
    """A hyper-parameter range for optuna search, instead of a list of values.

    Range(0.001, 100, log=True)   a float, sampled on a log scale
    Range(2, 20, int=True)        a whole number
    """
    low: float
    high: float
    log: bool = False
    int: bool = False
    step: float | None = None

    def __post_init__(self):
        if not self.low < self.high:
            raise ValueError(f"Range needs low < high, got {self.low}, {self.high}")
        if self.log and self.low <= 0:
            raise ValueError("a log Range needs low > 0")
        if self.log and self.step is not None:
            raise ValueError("a Range can't have both log and step")

    @classmethod
    def from_dict(cls, raw: dict) -> "Range":
        unknown = set(raw) - {"low", "high", "log", "int", "step"}
        if unknown or "low" not in raw or "high" not in raw:
            raise ValueError(f"a range needs low and high (optional: log, int, step), got {raw}")
        return cls(**raw)

    def suggest(self, trial, name: str):
        if self.int:
            step = int(self.step) if self.step else 1
            return trial.suggest_int(name, int(self.low), int(self.high), step=step, log=self.log)
        return trial.suggest_float(name, self.low, self.high, step=self.step, log=self.log)


def has_ranges(grid: dict) -> bool:
    return any(isinstance(v, Range) for v in grid.values())


class Candidate(NamedTuple):
    model: int  # index into the run's models
    features: tuple[int, ...] | None  # column indexes; None = every column
    params: dict  # hyper-parameters set on the model (only those it has)


@dataclass
class SearchSettings:
    strategy: str = "grid"
    n_iter: int | None = None  # random: sample size; halving: optional sample to start from; optuna: trials
    factor: float = 3.0  # halving: keep 1/factor per round, factor x more rows next round
    min_rows: int | None = None  # halving: rows in the first round (default: from the data)
    criteria: object = None  # metric used to compare candidates between rounds

    def validate(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"search strategy must be one of {list(STRATEGIES)}, not {self.strategy!r}")
        if self.strategy in ("random", "optuna") and self.n_iter is None:
            raise ValueError(f"{self.strategy} search needs n_iter (how many configurations to try)")
        if self.n_iter is not None and (not isinstance(self.n_iter, int) or self.n_iter < 1):
            raise ValueError("n_iter must be a whole number >= 1")
        if not self.factor > 1:
            raise ValueError("halving factor must be greater than 1")
        if self.min_rows is not None and (not isinstance(self.min_rows, int) or self.min_rows < 2):
            raise ValueError("min_rows must be a whole number >= 2")


@dataclass
class FeatureSearch:
    direction: str  # forward | backward
    max_features: int | None = None  # forward: stop at this many features
    min_features: int = 1  # backward: stop at this many features
    tol: float = 0.0  # forward: a step must improve the score by more than this

    def validate(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"feature search direction must be one of {list(DIRECTIONS)}")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be >= 1")
        if self.min_features < 1:
            raise ValueError("min_features must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be >= 0")


# evaluate(candidates, rows, description) -> results, each with .candidate_id = its index in
# `candidates`. rows = the row indexes to use (None = all rows).
Evaluate = Callable[[list[Candidate], "np.ndarray | None", str], list]


# ---------------------------------------------------------------------- scoring
def score(result, criteria) -> float:
    """Higher is better, whatever the metric."""
    value = float(getattr(result, criteria.value))
    if math.isnan(value):
        return -math.inf
    return value if criteria.maximize else -value


def rank(results, criteria, n_columns: int) -> list:
    """Best first; ties go to the configuration with fewer features (the simpler model)."""
    def key(r):
        n = len(r.features_index) if r.features_index is not None else n_columns
        return -score(r, criteria), n

    return sorted(results, key=key)


# ---------------------------------------------------------------------- candidate space
def hyper_parameter_combinations(model, grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid values this model has a parameter for ([{}] if none)."""
    params = model.get_params(deep=False) if hasattr(model, "get_params") else vars(model)
    keys = [k for k in grid if k in params]
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


class CandidateSpace:
    """models x feature sets x hyper-parameter combinations, without building them all:
    random search draws indexes into the space, so 2^n feature subsets never sit in memory."""

    def __init__(self, models: list, feature_sets: list, grid: dict[str, list]):
        self.feature_sets = [tuple(int(i) for i in f) if f is not None else None for f in feature_sets]
        self.params = [hyper_parameter_combinations(m, grid) for m in models]
        self.sizes = [len(self.feature_sets) * len(p) for p in self.params]

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, index: int) -> Candidate:
        for model, size in enumerate(self.sizes):
            if index < size:
                n_params = len(self.params[model])
                return Candidate(model, self.feature_sets[index // n_params], self.params[model][index % n_params])
            index -= size
        raise IndexError(index)

    def all(self) -> list[Candidate]:
        return [self[i] for i in range(len(self))]

    def sample(self, n: int, rng: np.random.Generator) -> list[Candidate]:
        if n >= len(self):
            return self.all()
        picks = rng.choice(len(self), size=n, replace=False)
        return [self[int(i)] for i in np.sort(picks)]

    def tracks(self) -> list[tuple[int, dict]]:
        """(model, hyper-parameters) pairs: the starting points of a greedy feature search."""
        return [(m, p) for m, combos in enumerate(self.params) for p in combos]


# ---------------------------------------------------------------------- successive halving
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


def successive_halving(evaluate: Evaluate, candidates: list[Candidate], subsample, n_rows: int, min_rows: int,
                       factor: float, criteria, n_columns: int, log: list) -> list:
    """subsample(n) -> the row indexes of one round. Returns the last round's results
    (all of them trained on every row, so they compare fairly with each other)."""
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


# ---------------------------------------------------------------------- greedy feature search
def greedy_features(evaluate: Evaluate, tracks: list[tuple[int, dict]], n_columns: int, search: FeatureSearch,
                    criteria, log: list) -> list:
    """Sequential feature selection, run for every (model, hyper-parameters) track at once:
    each round trains every track's next-step subsets in one pool batch.

    forward:  start empty; add the feature that raises the score most; stop when no
              addition improves it by more than tol, or at max_features.
    backward: start with every column; drop the feature whose removal leaves the best
              score; stop when every removal lowers the score by more than tol below the
              best seen, or at min_features.
    tol only decides when to stop. Every subset trained is returned (all on every row),
    including the last step's, so the leaderboard ranks them all by score."""
    forward = search.direction == "forward"
    everything = tuple(range(n_columns))
    current = {t: () if forward else everything for t in range(len(tracks))}
    best = {t: -math.inf for t in current}
    active = set(current)
    all_results = []

    if not forward:  # score the full set first: it is what removals are compared with
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
        size = len(candidates[0].features)  # tracks move in step, so every subset has this size
        results = evaluate(candidates, None, f"{search.direction} step {step} ({size} features)")
        all_results += results
        log.append({"round": step, "features": size, "configurations": len(candidates)})

        step_best: dict[int, tuple[float, int]] = {}  # track -> (score, candidate id)
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
                # compare with the best score so far, so small losses can't add up past tol
                current[t], best[t] = candidates[cid].features, max(s, best[t])
            else:
                active.discard(t)
    return all_results


# ---------------------------------------------------------------------- optuna
def import_optuna():
    try:
        import optuna
    except ImportError as e:
        raise ImportError("optuna search needs the optuna package: pip install optuna "
                          "(or pip install ml_pipe[optuna])") from e
    return optuna


def optuna_search(evaluate: Evaluate, models: list, feature_sets: list | None, n_columns: int, grid: dict,
                  n_trials: int, batch_size: int, criteria, seed, log: list) -> list:
    """Ask optuna for `batch_size` trials, train them together in the pool, report the
    scores, repeat until n_trials. TPE's constant liar keeps a batch from piling onto
    one point. The search space is built per trial:

    - the model (if there is more than one), then only the hyper-parameters it has;
      lists are categorical choices, Range values are sampled from the range.
    - feature_sets = None (brute force): one include / exclude switch per column, so
      the 2^n subsets are never listed. Several sets (SelectKBest k values): pick one.

    A configuration optuna proposes twice is scored from the first run, not retrained.
    Returns every configuration trained, all of them on every row."""
    optuna = import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # multivariate / group are "experimental"
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    names = [f"{i}_{type(m).__name__}" for i, m in enumerate(models)]
    model_params = [m.get_params(deep=False) if hasattr(m, "get_params") else vars(m) for m in models]

    def propose(trial) -> Candidate | None:
        m = trial.suggest_categorical("model", names) if len(models) > 1 else names[0]
        index = names.index(m)
        params = {}
        for key, values in grid.items():
            if key not in model_params[index]:
                continue
            name = f"{m}.{key}"
            if isinstance(values, Range):
                params[key] = values.suggest(trial, name)
            else:  # choose by position: values may be None, tuples or objects
                params[key] = values[trial.suggest_categorical(name, list(range(len(values))))]
        if feature_sets is None:
            features = tuple(i for i in range(n_columns) if trial.suggest_categorical(f"use_x{i}", [True, False]))
            if not features:
                return None
        elif len(feature_sets) > 1:
            chosen = feature_sets[trial.suggest_categorical("feature_set", list(range(len(feature_sets))))]
            features = tuple(chosen) if chosen is not None else None
        else:
            features = tuple(feature_sets[0]) if feature_sets[0] is not None else None
        return Candidate(index, features, params)

    def key(c: Candidate):
        return c.model, c.features, tuple(sorted((k, repr(v)) for k, v in c.params.items()))

    scores: dict = {}  # candidate key -> score, so a repeat is not retrained
    results, done, batch_number = [], 0, 0
    best = -math.inf
    while done < n_trials:
        batch_number += 1
        trials = [study.ask() for _ in range(min(batch_size, n_trials - done))]
        done += len(trials)
        proposed = [propose(t) for t in trials]
        new, queued = [], set()
        for candidate in proposed:
            if candidate is not None and key(candidate) not in scores and key(candidate) not in queued:
                queued.add(key(candidate))
                new.append(candidate)
        batch = evaluate(new, None, f"optuna {done}/{n_trials}") if new else []
        for r in batch:
            scores[key(new[r.candidate_id])] = score(r, criteria)
        results += batch
        for trial, candidate in zip(trials, proposed):
            value = scores[key(candidate)] if candidate is not None else -math.inf
            if math.isinf(value):  # no feature switched on, or the metric was NaN
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, value)
                best = max(best, value)
        log.append({"round": batch_number, "trials": len(trials), "trained": len(new),
                    "best": round(best if criteria.maximize else -best, 6)})
    return results
