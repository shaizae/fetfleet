"""Benchmark for the training pipeline.

Run from the repo root:  python -m benchmarks.bench [--repeat N]
Uses only the public API, so it works on any commit for before/after comparisons.
"""
import argparse
import os
import pickle
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from external_classes.Classificetion import Classification

GRID = {"n_estimators": [50, 100, 120], "min_samples_split": [2, 3, 6], "C": [1, 2, 3]}


def models():
    return [RandomForestClassifier(random_state=0), AdaBoostClassifier(random_state=0),
            DecisionTreeClassifier(random_state=0), SVC(random_state=0)]


def iris():
    d = load_iris()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)


def synthetic():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, n_classes=3, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y)


def many_small_tasks():
    # brute-force features on iris -> 14 subsets x 4 models, cheap fits: overhead-dominated
    cls = Classification()
    cls.set(*iris(), [DecisionTreeClassifier(random_state=0), SVC(random_state=0)])
    cls.brut_force_features()
    return cls, lambda: cls.k_folds(5)


def grid_kfold():
    cls = Classification()
    cls.set(*iris(), models())
    cls.set_hyper_parameters(GRID)
    return cls, lambda: cls.k_folds(5)


def big_data_grid():
    cls = Classification()
    cls.set(*synthetic(), models())
    cls.set_hyper_parameters({"n_estimators": [50, 100], "min_samples_split": [2, 6], "C": [1, 3]})
    return cls, lambda: cls.k_folds(5)


def feature_selection():
    from sklearn.feature_selection import f_classif
    cls = Classification()
    cls.set(*synthetic(), [DecisionTreeClassifier(random_state=0)])
    return cls, lambda: cls.fetcher_selection(f_classif, list(range(1, 21)))


SCENARIOS = {"many_small_tasks": many_small_tasks, "grid_kfold": grid_kfold,
             "big_data_grid": big_data_grid, "feature_selection": feature_selection}


def run(name, repeat):
    times = []
    for _ in range(repeat):
        cls, fn = SCENARIOS[name]()
        tracemalloc.start()
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    results = cls.results if out is None else out
    n = len(results) if results is not None else 0
    size = len(pickle.dumps(results)) if n else 0
    return dict(scenario=name, runs=n, best_s=min(times), median_s=float(np.median(times)),
                parent_peak_mb=peak / 2**20, results_mb=size / 2**20)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", nargs="*", default=list(SCENARIOS))
    a = p.parse_args()
    print(f"cpus={os.cpu_count()} workers={Classification._process_limit}")
    rows = [run(s, a.repeat) for s in a.only]
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.2f"))
