from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas import Index

from utils.Fetchers import Fetchers


def small_fetchers() -> tuple[Fetchers, pd.DataFrame]:
    fetchers = Fetchers()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    fetchers.load_new_data(data)
    return fetchers, data


def test_load_new_data():
    fetchers, data = small_fetchers()

    assert np.array_equal(fetchers.data, data.values)
    assert np.array_equal(fetchers.columns, data.columns)
    assert isinstance(fetchers.data, np.ndarray)
    assert isinstance(fetchers.columns, Index)
    assert isinstance(fetchers.columns[0], str)


def test_data_shape_is_preserved():
    fetchers = Fetchers()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    fetchers.load_new_data(data)

    assert fetchers.data.shape == (3, 2)


def test_overwrite_existing_data():
    fetchers = Fetchers()

    first = pd.DataFrame({"a": [1, 2]})
    second = pd.DataFrame({"b": [3, 4]})

    fetchers.load_new_data(first)
    fetchers.load_new_data(second)

    assert np.array_equal(fetchers.data, second.values)
    assert list(fetchers.columns) == ["b"]


def test_pop_feature():
    fetchers = Fetchers()
    data = pd.DataFrame({"feature1": list(range(1, 50)), "feature2": list(range(20, 69)), })
    fetchers.load_new_data(data)
    popped = fetchers.pop_index([1, 6, 9, 7, 15])
    expected_popped = data.loc[[1, 6, 9, 7, 15]].values
    np.testing.assert_array_equal(popped, expected_popped)
    expected_remaining = data.drop([1, 6, 9, 7, 15])
    np.testing.assert_array_equal(fetchers.data, expected_remaining.values)
    np.testing.assert_array_equal(fetchers.raw_data, expected_remaining.values)


def test_iter():
    fetchers = Fetchers()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    fetchers.load_new_data(data)

    rows = list(fetchers)

    assert len(rows) == 3

    for i, row in enumerate(rows):
        np.testing.assert_array_equal(row, fetchers.data[i])


def test_zero_mean_with_mean_true(fetchers_with_data):
    fetchers_with_data.standard_scaler(with_mean=True, with_std=True)
    means = fetchers_with_data.data.mean(axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-10)


def test_unit_std_with_std_true(fetchers_with_data):
    fetchers_with_data.standard_scaler(with_mean=True, with_std=True)
    stds = fetchers_with_data.data.std(axis=0)
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_with_mean_false_preserves_original_mean(fetchers_with_data):
    original_means = fetchers_with_data.data.mean(axis=0)
    original_stds = fetchers_with_data.data.std(axis=0)
    fetchers_with_data.standard_scaler(with_mean=False, with_std=True)
    scaled_means = fetchers_with_data.data.mean(axis=0)
    # mean is NOT subtracted: it is only divided by the std along with the data
    assert not np.allclose(scaled_means, 0.0)
    np.testing.assert_allclose(scaled_means, original_means / original_stds)
    # but unit std still holds
    np.testing.assert_allclose(fetchers_with_data.data.std(axis=0), 1.0, atol=1e-10)


def test_with_std_false_preserves_scale(fetchers_with_data):
    original_stds = fetchers_with_data.data.std(axis=0)
    fetchers_with_data.standard_scaler(with_mean=True, with_std=False)
    # mean is centered at 0
    np.testing.assert_allclose(fetchers_with_data.data.mean(axis=0), 0.0, atol=1e-10)
    # std is NOT scaled — original std is preserved
    np.testing.assert_allclose(fetchers_with_data.data.std(axis=0), original_stds, atol=1e-10)


def test_both_false_leaves_data_unchanged(fetchers_with_data):
    original = fetchers_with_data.data.copy()
    fetchers_with_data.standard_scaler(with_mean=False, with_std=False)
    np.testing.assert_array_equal(fetchers_with_data.data, original)


def test_output_shape_unchanged(fetchers_with_data):
    original_shape = fetchers_with_data.data.shape
    fetchers_with_data.standard_scaler()
    assert fetchers_with_data.data.shape == original_shape


def test_single_row_does_not_raise():
    df = pd.DataFrame({"a": [5.0], "b": [99.0]})
    f = Fetchers()
    f.load_new_data(df)
    # StandardScaler with a single sample produces NaN when with_std=True;
    # verify it at least runs without raising an exception.
    f.standard_scaler(with_mean=True, with_std=True)


def test_constant_column_produces_zero_or_nan():
    df = pd.DataFrame({"a": [7.0, 7.0, 7.0], "b": [1.0, 2.0, 3.0]})
    f = Fetchers()
    f.load_new_data(df)
    f.standard_scaler(with_mean=True, with_std=True)
    # Constant column → std=0 → StandardScaler outputs 0.0 for that column
    assert np.all(f.data[:, 0] == 0.0) or np.all(np.isnan(f.data[:, 0]))


def test_modifies_data_in_place(fetchers_with_data):
    original = fetchers_with_data.data.copy()
    fetchers_with_data.standard_scaler()
    assert not np.array_equal(fetchers_with_data.data, original)


def test_default_args_equal_both_true():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    f1 = Fetchers()
    f1.load_new_data(df)
    f1.standard_scaler()

    f2 = Fetchers()
    f2.load_new_data(df)
    f2.standard_scaler(with_mean=True, with_std=True)

    np.testing.assert_array_equal(f1.data, f2.data)


def test_min_max_output_shape_unchanged(fetchers_with_data):
    original_shape = fetchers_with_data.data.shape
    fetchers_with_data.min_max_scaler()
    assert fetchers_with_data.data.shape == original_shape


def test_min_max_modifies_data_in_place(fetchers_with_data):
    original = fetchers_with_data.data.copy()
    fetchers_with_data.min_max_scaler()
    assert not np.array_equal(fetchers_with_data.data, original)


def test_min_max_scaler_output(fetchers_with_data):
    fetcher = deepcopy(fetchers_with_data)
    fetcher.min_max_scaler()
    assert fetcher.min() == 0
    assert fetcher.max() == 1

    fetcher = deepcopy(fetchers_with_data)
    fetcher.min_max_scaler(min_val=-1, max_val=2)
    assert fetcher.min() == -1
    assert fetcher.max() == 2

    try:
        fetcher = deepcopy(fetchers_with_data)
        fetcher.min_max_scaler(3, 1)
    except ValueError:
        pass
    else:
        assert False


def test_min():
    fetchers, _ = small_fetchers()
    assert fetchers.min() == 1


def test_max():
    fetchers, _ = small_fetchers()
    assert fetchers.max() == 6


def test_mean():
    fetchers, _ = small_fetchers()
    assert fetchers.mean() == np.mean([[1, 4], [2, 5], [3, 6]])


def test_std():
    fetchers, _ = small_fetchers()
    expected = np.std([[1, 4], [2, 5], [3, 6]])
    assert np.isclose(fetchers.std(), expected)


def test_median():
    fetchers, _ = small_fetchers()
    assert fetchers.median() == 3.5


def test_wrong_rows_error_message():
    fetchers, _ = small_fetchers()  # 3 x 2
    with pytest.raises(ValueError, match="new data must have same shape as existing data"):
        fetchers.update_data(np.array([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows instead of 3


def test_wrong_cols_error_message():
    fetchers, _ = small_fetchers()
    with pytest.raises(ValueError, match="new data must have same shape as existing data"):
        fetchers.update_data(np.array([[1.0], [2.0], [3.0]]))  # 1 col instead of 2


def test_data_not_updated_on_row_mismatch():
    fetchers, _ = small_fetchers()
    original = fetchers.data.copy()
    with pytest.raises(ValueError):
        fetchers.update_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(fetchers.data, original)


def test_data_not_updated_on_col_mismatch():
    fetchers, _ = small_fetchers()
    original = fetchers.data.copy()
    with pytest.raises(ValueError):
        fetchers.update_data(np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_array_equal(fetchers.data, original)


def test_extra_dimension_raises():
    fetchers, _ = small_fetchers()
    with pytest.raises(ValueError):
        fetchers.update_data(np.ones((3, 2, 2)))  # same rows and cols, one dimension too many


def test_update_data_replaces_data_and_resets_preprocessing():
    fetchers, _ = small_fetchers()
    fetchers.standard_scaler()
    new = np.array([[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]])
    fetchers.update_data(new)
    np.testing.assert_array_equal(fetchers.data, new)
    np.testing.assert_array_equal(fetchers.raw_data, new)
    assert fetchers.preprocessing == []


# -------------------------
# preprocessing is recorded so training can refit it per split
# -------------------------

def test_raw_data_is_kept_after_scaling(fetchers_with_data):
    raw = fetchers_with_data.data.copy()
    fetchers_with_data.standard_scaler()
    fetchers_with_data.min_max_scaler()
    np.testing.assert_array_equal(fetchers_with_data.raw_data, raw)


def test_preprocessing_records_steps_in_order(fetchers_with_data):
    fetchers_with_data.standard_scaler(with_mean=False)
    fetchers_with_data.polynomial_features(degree=2)
    steps = fetchers_with_data.preprocessing
    assert [type(s).__name__ for s in steps] == ["StandardScaler", "PolynomialFeatures"]
    assert steps[0].with_mean is False and steps[1].degree == 2


def test_preprocessing_returns_unfitted_copies(fetchers_with_data):
    fetchers_with_data.standard_scaler()
    first, second = fetchers_with_data.preprocessing[0], fetchers_with_data.preprocessing[0]
    assert first is not second
    assert not hasattr(first, "mean_")


def test_replaying_steps_on_raw_data_reproduces_data(fetchers_with_data):
    fetchers_with_data.min_max_scaler()
    fetchers_with_data.polynomial_features(degree=2)
    x = fetchers_with_data.raw_data
    for step in fetchers_with_data.preprocessing:
        x = step.fit_transform(x)
    np.testing.assert_allclose(x, fetchers_with_data.data)


def test_polynomial_features_adds_columns(fetchers_with_data):
    fetchers_with_data.polynomial_features(degree=2, include_bias=False)
    assert fetchers_with_data.data.shape == (150, 14)  # 4 + 4 squares + 6 pairs


def test_load_new_data_resets_preprocessing(fetchers_with_data):
    fetchers_with_data.standard_scaler()
    fetchers_with_data.load_new_data(pd.DataFrame({"a": [1.0, 2.0]}))
    assert fetchers_with_data.preprocessing == []
