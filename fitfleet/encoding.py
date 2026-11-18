"""Per-model encoding of a mixed table, fitted inside every split.

The features reach the workers as one float matrix (see fitfleet.schema): categories as
integer codes, missing values as NaN, dates as seconds. Different models need that matrix in
different shapes:

* linear models, SVMs, KNN, MLPs ... read numbers as ordered quantities, so a code "city 7"
  must not be "more" than "city 3": categorical columns are one-hot encoded - or, when a column
  has more than `max_onehot` categories, target encoded (each category -> the average target of
  its rows, cross-fitted by scikit-learn's TargetEncoder so a row never sees its own target).
* tree models split on thresholds and cope with arbitrary codes, so codes are kept as they
  are (one column per category column, no blow-up).
* HistGradientBoosting has native categorical support: its `categorical_features` is
  pointed at the code columns.
* models that cannot take NaN get an imputer (median + a "was missing" indicator column for
  numbers, a separate code for categories); models that can take NaN get the NaN as is.
* datetime columns (seconds since 1970) become calendar features: year, month, day, weekday,
  day of year, hour (when the data has times) and days since 1970 (DateParts).
* text columns arrive already embedded (see fitfleet.schema) and pass through as numbers.

Every step here is a scikit-learn transformer inside the model's Pipeline, so it is fitted
on the training rows of each split only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

if TYPE_CHECKING:
    from fitfleet.schema import ColumnSchema

ENCODINGS = ("auto", "onehot", "ordinal", "target")
# hyper-parameter name that searches over ENCODINGS (applies to every model; mixed tables only)
ENCODING_PARAM = "categorical_encoding"
MAX_ONEHOT = 40  # auto: categorical columns with more categories are target encoded for non-tree models
_SKLEARN = tuple(int(p) for p in sklearn.__version__.split(".")[:2] if p.isdigit())

# module prefixes of estimators that split on thresholds, so integer codes are fine for them
_TREE_MODULES = ("sklearn.tree", "sklearn.ensemble._forest", "sklearn.ensemble._gb",
                 "sklearn.ensemble._hist_gradient_boosting", "xgboost", "lightgbm", "catboost")


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def is_tree_model(model) -> bool:
    return type(final_estimator(model)).__module__.startswith(_TREE_MODULES)


def is_hist_gradient_boosting(model) -> bool:
    return type(final_estimator(model)).__name__ in ("HistGradientBoostingClassifier",
                                                     "HistGradientBoostingRegressor")


def allows_nan(estimator) -> bool:
    """Whether the estimator accepts NaN in its input (scikit-learn's input tags)."""
    try:
        from sklearn.utils import get_tags
        return bool(get_tags(estimator).input_tags.allow_nan)
    except Exception:  # not a scikit-learn estimator, or tags unavailable
        return False


class CategoryNames:
    """OneHotEncoder `feature_name_combiner`: names one-hot columns "column=label" with the
    real category label instead of its code. A class (not a lambda) so it pickles."""

    def __init__(self, schema: ColumnSchema):
        self.schema = schema

    def __call__(self, column, code) -> str:
        return f"{column}={self.schema.label(str(column), code) if column in self.schema.categories else code}"


class DateParts(TransformerMixin, BaseEstimator):
    """Calendar features from datetime columns stored as seconds since 1970-01-01.

    Per column: year, month, day, weekday (0 = Monday), day of year, days since 1970 and - only
    when the training rows have times other than midnight - hour. Missing stays NaN."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        seconds = np.nan_to_num(X % 86400)
        self.with_hour_ = [bool(seconds[:, j].any()) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for j in range(X.shape[1]):
            col = X[:, j]
            known = ~np.isnan(col)
            stamps = pd.to_datetime(np.where(known, col, 0), unit="s")
            parts = [stamps.year, stamps.month, stamps.day, stamps.day_of_week, stamps.day_of_year]
            if self.with_hour_[j]:
                parts.append(stamps.hour)
            block = np.column_stack([np.asarray(p, dtype=float) for p in parts] + [col / 86400])
            block[~known] = np.nan
            out.append(block)
        return np.hstack(out) if out else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        names = []
        for j, col in enumerate(input_features):
            parts = ["year", "month", "day", "weekday", "day_of_year"] + (["hour"] if self.with_hour_[j] else [])
            names += [f"{col}_{p}" for p in parts] + [f"{col}_days"]
        return np.asarray(names, dtype=object)


def target_encoder(regression: bool, random_state: int | None = 0) -> TargetEncoder:
    """scikit-learn's TargetEncoder with a fixed internal split, so runs are reproducible.
    Its fit_transform (what a Pipeline calls) cross-fits: each row is encoded from the other folds."""
    kwargs = {"target_type": "continuous" if regression else "auto"}
    seed = 0 if random_state is None else random_state
    if _SKLEARN >= (1, 9):  # shuffle / random_state are deprecated in favour of a splitter
        kwargs["cv"] = KFold(5, shuffle=True, random_state=seed)
    else:
        kwargs.update(shuffle=True, random_state=seed)
    return TargetEncoder(**kwargs)


def _number_steps(preprocessing: list, model_allows_nan: bool, has_missing: bool) -> list:
    """Imputer (when needed) followed by the user's preprocessing steps, for number columns."""
    steps = [clone(step) for step in preprocessing or []]
    needs_imputer = has_missing and (not model_allows_nan or not all(allows_nan(s) for s in steps))
    if needs_imputer:
        steps.insert(0, SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    return steps


def _branch(steps: list):
    if not steps:
        return "passthrough"
    if len(steps) == 1:
        return steps[0]
    return Pipeline([(f"step{i}", s) for i, s in enumerate(steps)])


def build_encoder(schema: ColumnSchema, columns, model, preprocessing: list | None,
                  encoding: str = "auto", *, max_onehot: int = MAX_ONEHOT, regression: bool = False,
                  random_state: int | None = 0):
    """(encoder, model) for the given original column indexes (None = all columns).

    The encoder is a ColumnTransformer: categorical columns first (codes, one-hot or target
    encoded), then numeric / boolean / ordinal ones and the calendar features of datetime
    columns (the user's preprocessing is applied to these numbers only). `model` is returned as
    is, except HistGradientBoosting, whose clone gets `categorical_features` set to the code
    columns. Returns encoder None when the selected columns need nothing at all.

    encoding: "auto" (per model, see the module docstring), "onehot", "ordinal" (codes for every
    model) or "target" (target encoding for every model). `regression` tells the target encoder
    what kind of target it sees."""
    if encoding not in ENCODINGS:
        raise ValueError(f"categorical encoding must be one of {list(ENCODINGS)}, not {encoding!r}")
    selected = list(range(len(schema.columns))) if columns is None else [int(c) for c in columns]
    cat_cols, date_cols, text_cols = set(schema.categorical_index), set(schema.datetime_index), set(schema.text_index)
    # position of each selected column's first matrix column in the selected rows (text takes several)
    start = list(np.cumsum([0] + [schema.width(i) for i in selected])[:-1])
    names = {start[k]: schema.columns[i] for k, i in enumerate(selected)}
    cat_pos = [int(start[k]) for k, i in enumerate(selected) if i in cat_cols]
    date_pos = [int(start[k]) for k, i in enumerate(selected) if i in date_cols]
    text_pos = [int(start[k]) + j for k, i in enumerate(selected) if i in text_cols for j in range(schema.width(i))]
    num_pos = [int(start[k]) for k, i in enumerate(selected) if i not in cat_cols | date_cols | text_cols]
    missing = set(schema.missing)
    model_nan = allows_nan(final_estimator(model))

    # Predictions on new data may hold NaN even where training data had none, so a model
    # that cannot take NaN always gets an imputer on its categorical code columns.
    transformers = []
    if cat_pos:
        wide = [p for p in cat_pos if schema.n_categories(names[p]) > max_onehot]
        if encoding == "target":
            groups = {"target": cat_pos}
        elif encoding == "onehot":
            groups = {"onehot": cat_pos}
        elif encoding == "ordinal" or is_tree_model(model):
            groups = {"codes": cat_pos}
        else:  # auto, non-tree model: one-hot, target encoding for wide columns
            groups = {"onehot": [p for p in cat_pos if p not in wide], "target": wide}
        for kind, positions in groups.items():
            if not positions:
                continue
            if kind == "onehot":
                # NaN becomes its own category; categories unseen in the training rows get all zeros
                cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                    feature_name_combiner=CategoryNames(schema))
            elif kind == "target":  # NaN and unseen categories are handled by the encoder itself
                cat = target_encoder(regression, random_state)
            elif model_nan:
                cat = "passthrough"
            else:
                cat = SimpleImputer(strategy="constant", fill_value=-1.0, keep_empty_features=True)
            transformers.append(("categorical" if kind == "codes" else f"categorical_{kind}", cat, positions))
    if num_pos:
        num_missing = any(names[p] in missing for p in num_pos)
        transformers.append(("numeric", _branch(_number_steps(preprocessing, model_nan, num_missing)), num_pos))
    if date_pos:
        date_missing = any(names[p] in missing for p in date_pos)
        steps = [DateParts()] + _number_steps(preprocessing, model_nan, date_missing)
        transformers.append(("datetime", Pipeline([(f"step{i}", s) for i, s in enumerate(steps)]), date_pos))
    if text_pos:
        transformers.append(("text", "passthrough", text_pos))
    if not cat_pos and not date_pos and not text_pos and all(t[1] == "passthrough" for t in transformers):
        return None, model

    codes = next((t[2] for t in transformers if t[0] == "categorical"), [])
    if codes and is_hist_gradient_boosting(model):
        model = _with_native_categories(model, schema, [names[p] for p in codes])
    encoder = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0,
                                verbose_feature_names_out=False)
    return encoder, model


def _with_native_categories(model, schema: ColumnSchema, cat_names: list[str]):
    """A clone of a HistGradientBoosting model that treats the code columns (the first
    len(cat_names) columns of the encoder output) as categories - unless the user set
    categorical_features themselves. Columns with more categories than max_bins stay numeric."""
    est = final_estimator(model)
    if est.get_params().get("categorical_features") not in ("from_dtype", None):
        return model
    max_bins = est.get_params().get("max_bins", 255)
    native = [i for i, name in enumerate(cat_names) if schema.n_categories(name) < max_bins]
    if not native:
        return model
    model = clone(model)
    final_estimator(model).set_params(categorical_features=np.asarray(native))
    return model
