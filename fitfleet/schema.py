"""Column types of the input table, and the conversion of that table into one numeric matrix.

fitfleet keeps the features as one numeric numpy array in shared memory, so every worker
reads them without a copy. ColumnSchema makes that possible for tables that are not purely
numeric:

* numeric columns are stored as floats; a missing value is NaN.
* boolean columns become 0.0 / 1.0 (NaN when missing).
* categorical columns (text, pandas `category`, or columns you declare) become integer codes
  0..n-1, one per category, in a fixed order kept in `categories`. A missing value is NaN; a
  value not seen when the schema was built gets the code n ("unknown").
* ordinal columns (declared with their order, e.g. size = ["S", "M", "L"]) become their position
  in that order and are then used as numbers. A value outside the order counts as missing.
* datetime columns become seconds since 1970-01-01 (UTC); calendar features (year, month,
  weekday ...) are derived from them inside the model pipeline (fitfleet.encoding.DateParts).
* text columns (free text: sentences, descriptions, product names) become `text_components`
  numbers each: character n-gram TF-IDF reduced with SVD (latent semantic analysis). This is the
  one step learned from the data before splitting - it is unsupervised (it never sees the
  target), like the vocabulary of a categorical column. A text column is one column for feature
  selection, however many numbers it takes in the matrix.

`to_pandas` turns polars (or any narwhals-supported) frames into pandas first.

The conversion learns nothing from the target and the category list is just the vocabulary
of the column, so building it on all rows leaks nothing. Everything that *learns* (one-hot or
target encoding, imputation, scaling) is fitted later, inside each split (see fitfleet.encoding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

import numpy as np
import pandas as pd


class ColumnKind(StrEnum):
    """How a column is stored and encoded."""
    numeric = "numeric"
    boolean = "boolean"
    categorical = "categorical"
    ordinal = "ordinal"
    datetime = "datetime"
    text = "text"


# kinds that are numbers once stored (go through imputation and the user's scaling)
NUMBER_KINDS = (ColumnKind.numeric, ColumnKind.boolean, ColumnKind.ordinal)

_BOOL_VALUES = {True, False}
_EPOCH = pd.Timestamp("1970-01-01")
TEXT_COMPONENTS = 30  # numbers per text column (fewer when the text is short or the table small)


def to_pandas(data):
    """A pandas DataFrame / Series for pandas, polars (or other frames with to_pandas()) input;
    anything else (numpy arrays, lists) is returned unchanged."""
    if isinstance(data, (pd.DataFrame, pd.Series)) or not hasattr(data, "to_pandas"):
        return data
    return data.to_pandas()


def looks_like_text(series: pd.Series) -> bool:
    """Free text rather than categories: strings of 3+ words on average, and many distinct values."""
    if not _is_text(series):
        return False
    values = series.dropna()
    if len(values) < 20 or not all(isinstance(v, str) for v in values.iloc[:200]):
        return False
    words = values.astype(str).str.split().str.len().mean()
    return words >= 3 and values.nunique() >= max(20, 0.2 * len(values))


def fit_text_embedding(series: pd.Series, components: int = TEXT_COMPONENTS, random_state: int = 0):
    """Character n-gram TF-IDF + truncated SVD fitted on the column (no target involved)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    text = series.fillna("").astype(str)
    matrix = tfidf = None
    for min_df in (2, 1):
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True,
                                max_features=50_000, dtype=np.float32)
        try:
            matrix = tfidf.fit_transform(text)
        except ValueError:  # empty vocabulary
            continue
        if matrix.shape[1] >= 2:
            break
    if matrix is None or matrix.shape[1] < 2:
        raise ValueError(f"text column {series.name!r} has too little text to learn from")
    k = max(1, min(components, matrix.shape[1] - 1, len(text) - 1))
    svd = TruncatedSVD(n_components=k, random_state=random_state).fit(matrix)
    return make_pipeline(tfidf, svd)


def _sorted_values(values: Iterable) -> list:
    """Distinct values in a stable order (natural order when comparable, else by text)."""
    values = list(values)
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda v: (type(v).__name__, str(v)))


def _is_object_bool(series: pd.Series) -> bool:
    values = set(series.dropna().unique())
    return bool(values) and values <= _BOOL_VALUES and all(isinstance(v, (bool, np.bool_)) for v in values)


def _is_text(series: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series))


def to_seconds(series: pd.Series, name=None) -> np.ndarray:
    """Dates / times as float seconds since 1970-01-01 UTC (NaN when missing).
    Text is parsed; a value that is present but can't be read as a date raises ValueError."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().to_numpy(dtype=np.float64, na_value=np.nan)
    values = series
    if not pd.api.types.is_datetime64_any_dtype(values):
        values = pd.to_datetime(series, errors="coerce", format="mixed")
        bad = values.isna() & series.notna()
        if bad.any():
            raise ValueError(f"column {name!r} has values that are not dates: {list(series[bad].unique()[:3])}")
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    return (values - _EPOCH).dt.total_seconds().to_numpy(dtype=np.float64, na_value=np.nan)


def id_like_columns(X: pd.DataFrame, ratio: float = 0.95, min_rows: int = 20) -> list:
    """Text / category columns where (almost) every row has its own value - customer ids, row
    keys, free-form names. As categories they can't generalize: every new row is unknown."""
    if len(X) < min_rows:
        return []
    out = []
    for name in X.columns:
        s = X[name]
        if looks_like_text(s):  # free text is unique per row too, but it carries words
            continue
        if _is_text(s) and s.notna().any() and s.nunique(dropna=True) >= ratio * s.notna().sum():
            out.append(name)
    return out


@dataclass
class ColumnSchema:
    """The kind of every feature column, the categories of the categorical / ordinal ones, and
    which columns had missing values when the schema was built."""
    columns: list[str]
    kinds: list[ColumnKind]
    categories: dict[str, list] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    text_embeddings: dict = field(default_factory=dict)  # text column -> fitted TF-IDF + SVD pipeline

    # ------------------------------------------------------------------ build
    @classmethod
    def infer(cls, X: pd.DataFrame, categorical: Iterable[str] | None = None,
              numeric: Iterable[str] | None = None, ordinal: dict[str, list] | None = None,
              datetime: Iterable[str] | None = None, text: Iterable[str] | None = None,
              text_components: int = TEXT_COMPONENTS, random_state: int | None = 0) -> ColumnSchema:
        """Detect each column's kind.

        Declared kinds win: `categorical` (e.g. a zip code stored as numbers), `numeric` (values
        must convert to numbers), `ordinal` ({column: [values, lowest first]}) and `datetime`
        (parsed as dates) and `text` (free text, embedded - see the module docstring). Otherwise:
        bool -> boolean, numbers -> numeric, dates -> datetime, sentences -> text (looks_like_text),
        other text / pandas category -> categorical (an ordered pandas category -> ordinal)."""
        X = to_pandas(X)
        declared = {"categorical": set(categorical or ()), "numeric": set(numeric or ()),
                    "ordinal": set(ordinal or {}), "datetime": set(datetime or ()), "text": set(text or ())}
        ordinal = dict(ordinal or {})
        known = set(X.columns) | set(map(str, X.columns))
        unknown = sorted({str(c) for names in declared.values() for c in names if c not in known})
        if unknown:
            raise ValueError(f"columns not in the data: {unknown}")
        seen: dict = {}
        for kind, names in declared.items():
            for c in names:
                if c in seen:
                    raise ValueError(f"column {c!r} is declared both {seen[c]} and {kind}")
                seen[c] = kind
        for c, order in ordinal.items():
            if not isinstance(order, (list, tuple)) or len(order) < 2 or len(set(order)) != len(order):
                raise ValueError(f"ordinal column {c!r} needs a list of at least 2 distinct values, lowest first")

        def is_declared(name, kind):
            return name in declared[kind] or str(name) in declared[kind]

        kinds, categories, embeddings = [], {}, {}
        for name in X.columns:
            s = X[name]
            if is_declared(name, "text") or (not any(is_declared(name, k) for k in declared)
                                             and looks_like_text(s)):
                seed = 0 if random_state is None else random_state
                embeddings[name] = fit_text_embedding(s, text_components, seed)
                kinds.append(ColumnKind.text)
                continue
            if is_declared(name, "categorical"):
                kind = ColumnKind.categorical
            elif is_declared(name, "ordinal"):
                kind = ColumnKind.ordinal
                order = ordinal.get(name, ordinal.get(str(name)))
                outside = set(s.dropna().unique()) - set(order)
                if outside:
                    raise ValueError(f"ordinal column {name!r} has values missing from its order: "
                                     f"{_sorted_values(outside)[:5]}")
                categories[name] = list(order)
            elif is_declared(name, "datetime") or pd.api.types.is_datetime64_any_dtype(s) \
                    or pd.api.types.is_timedelta64_dtype(s):
                to_seconds(s, name)  # raises for values that are not dates
                kind = ColumnKind.datetime
            elif is_declared(name, "numeric"):
                try:
                    pd.to_numeric(s, errors="raise")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"column {name!r} was declared numeric but has non-numeric values: {e}") from None
                kind = ColumnKind.numeric
            elif pd.api.types.is_bool_dtype(s) or (s.dtype == object and _is_object_bool(s)):
                kind = ColumnKind.boolean
            elif pd.api.types.is_numeric_dtype(s):
                kind = ColumnKind.numeric
            elif isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered:
                kind = ColumnKind.ordinal
                categories[name] = list(s.cat.categories)
            else:  # object, string or category dtype
                kind = ColumnKind.categorical
            if kind == ColumnKind.categorical:
                if isinstance(s.dtype, pd.CategoricalDtype):
                    categories[name] = list(s.cat.categories)
                else:
                    categories[name] = _sorted_values(s.dropna().unique())
            kinds.append(kind)
        # a missing text is just an empty one: no NaN reaches the matrix
        missing = [c for c, k in zip(X.columns, kinds) if k != ColumnKind.text and X[c].isna().any()]
        return cls(columns=list(X.columns), kinds=kinds, categories=categories, missing=missing,
                   text_embeddings=embeddings)

    # ------------------------------------------------------------------ queries
    @property
    def simple(self) -> bool:
        """Every column numeric and nothing missing: the table needs no encoding at all."""
        return not self.missing and all(k == ColumnKind.numeric for k in self.kinds)

    def index_of(self, *kinds: ColumnKind) -> list[int]:
        return [i for i, k in enumerate(self.kinds) if k in kinds]

    @property
    def categorical_index(self) -> list[int]:
        return self.index_of(ColumnKind.categorical)

    @property
    def numeric_index(self) -> list[int]:
        """Numeric, boolean and ordinal columns (all encoded as numbers)."""
        return self.index_of(*NUMBER_KINDS)

    @property
    def datetime_index(self) -> list[int]:
        return self.index_of(ColumnKind.datetime)

    @property
    def text_index(self) -> list[int]:
        return self.index_of(ColumnKind.text)

    def width(self, column: int) -> int:
        """How many matrix columns the column takes (text: its SVD components, others: 1)."""
        if self.kinds[column] == ColumnKind.text:
            return int(self.text_embeddings[self.columns[column]].steps[-1][1].n_components)
        return 1

    @property
    def n_matrix(self) -> int:
        """Columns of the encoded matrix."""
        return sum(self.width(i) for i in range(len(self.columns)))

    def matrix_index(self, columns: Iterable[int] | None = None) -> list[int]:
        """Matrix column indexes of the given (original) columns, in order; None = all."""
        starts = np.cumsum([0] + [self.width(i) for i in range(len(self.columns))])
        columns = range(len(self.columns)) if columns is None else columns
        return [j for i in columns for j in range(int(starts[int(i)]), int(starts[int(i) + 1]))]

    def matrix_names(self, columns: Iterable[int] | None = None) -> list[str]:
        """A name per matrix column of the given columns (text: name_text0, name_text1 ...)."""
        columns = range(len(self.columns)) if columns is None else columns
        names = []
        for i in columns:
            name = str(self.columns[int(i)])
            if self.kinds[int(i)] == ColumnKind.text:
                names += [f"{name}_text{k}" for k in range(self.width(int(i)))]
            else:
                names.append(name)
        return names

    def n_categories(self, column: str) -> int:
        return len(self.categories[column])

    def label(self, column: str, code: float) -> str:
        """The category behind a code (or "unknown" / "missing")."""
        if code != code:  # NaN
            return "missing"
        cats = self.categories.get(column)
        code = int(code)
        if cats is None or not 0 <= code < len(cats):
            return "unknown"
        return str(cats[code])

    def describe(self) -> dict[str, int]:
        """How many columns of each kind (for logs and reports)."""
        out = {kind.value: self.kinds.count(kind) for kind in ColumnKind if kind in self.kinds}
        if self.missing:
            out["with_missing_values"] = len(self.missing)
        return out

    # ------------------------------------------------------------------ convert
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """`X` (the schema's columns, by name) as a float64 matrix; see the module docstring."""
        X = to_pandas(X)
        absent = [c for c in self.columns if c not in X.columns]
        if absent:
            raise KeyError(f"columns missing from the data: {absent}")
        out = np.empty((len(X), self.n_matrix), dtype=np.float64)
        for i, (name, kind) in enumerate(zip(self.columns, self.kinds)):
            s = X[name]
            cols = self.matrix_index([i])
            j = cols[0]
            if kind == ColumnKind.text:
                out[:, cols] = self.text_embeddings[name].transform(s.fillna("").astype(str))
            elif kind in (ColumnKind.categorical, ColumnKind.ordinal):
                cats = self.categories[name]
                values = s.astype(object).to_numpy() if isinstance(s.dtype, pd.CategoricalDtype) else s.to_numpy()
                codes = pd.Index(cats, dtype=object).get_indexer(values).astype(np.float64)  # -1: missing or unseen
                missing = s.isna().to_numpy()
                # an unseen category gets its own code n; an unseen ordinal value has no place: missing
                codes[(codes < 0) & ~missing] = len(cats) if kind == ColumnKind.categorical else np.nan
                codes[missing] = np.nan
                out[:, j] = codes
            elif kind == ColumnKind.datetime:
                out[:, j] = to_seconds(s, name)
            elif kind == ColumnKind.boolean:
                out[:, j] = s.astype("boolean").astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                values = pd.to_numeric(s, errors="raise")
                out[:, j] = pd.Series(values).astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        return out
