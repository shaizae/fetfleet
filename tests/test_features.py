from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas import Index

from ml_pipe.features import Features


def small_features() -> tuple[Features, pd.DataFrame]:
    features = Features()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    features.load_new_data(data)
    return features, data


def test_load_new_data():
    features, data = small_features()

    assert np.array_equal(features.data, data.values)
    assert np.array_equal(features.columns, data.columns)
    assert isinstance(features.data, np.ndarray)
    assert isinstance(features.columns, Index)
    assert isinstance(features.columns[0], str)


def test_data_shape_is_preserved():
    features = Features()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    features.load_new_data(data)

    assert features.data.shape == (3, 2)


def test_overwrite_existing_data():
    features = Features()

    first = pd.DataFrame({"a": [1, 2]})
    second = pd.DataFrame({"b": [3, 4]})

    features.load_new_data(first)
    features.load_new_data(second)

    assert np.array_equal(features.data, second.values)
    assert list(features.columns) == ["b"]


def test_pop_feature():
    features = Features()
    data = pd.DataFrame({"feature1": list(range(1, 50)), "feature2": list(range(20, 69)), })
    features.load_new_data(data)
    popped = features.pop_index([1, 6, 9, 7, 15])
    expected_popped = data.loc[[1, 6, 9, 7, 15]].values
    np.testing.assert_array_equal(popped, expected_popped)
    expected_remaining = data.drop([1, 6, 9, 7, 15])
    np.testing.assert_array_equal(features.data, expected_remaining.values)
    np.testing.assert_array_equal(features.raw_data, expected_remaining.values)


def test_iter():
    features = Features()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    features.load_new_data(data)

    rows = list(features)

    assert len(rows) == 3

    for i, row in enumerate(rows):
        np.testing.assert_array_equal(row, features.data[i])


def test_zero_mean_with_mean_true(features_with_data):
    features_with_data.standard_scaler(with_mean=True, with_std=True)
    means = features_with_data.data.mean(axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-10)


def test_unit_std_with_std_true(features_with_data):
    features_with_data.standard_scaler(with_mean=True, with_std=True)
    stds = features_with_data.data.std(axis=0)
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_with_mean_false_preserves_original_mean(features_with_data):
    original_means = features_with_data.data.mean(axis=0)
    original_stds = features_with_data.data.std(axis=0)
    features_with_data.standard_scaler(with_mean=False, with_std=True)
    scaled_means = features_with_data.data.mean(axis=0)
    # mean is NOT subtracted: it is only divided by the std along with the data
    assert not np.allclose(scaled_means, 0.0)
    np.testing.assert_allclose(scaled_means, original_means / original_stds)
    # but unit std still holds
    np.testing.assert_allclose(features_with_data.data.std(axis=0), 1.0, atol=1e-10)


def test_with_std_false_preserves_scale(features_with_data):
    original_stds = features_with_data.data.std(axis=0)
    features_with_data.standard_scaler(with_mean=True, with_std=False)
    # mean is centered at 0
    np.testing.assert_allclose(features_with_data.data.mean(axis=0), 0.0, atol=1e-10)
    # std is NOT scaled — original std is preserved
    np.testing.assert_allclose(features_with_data.data.std(axis=0), original_stds, atol=1e-10)


def test_both_false_leaves_data_unchanged(features_with_data):
    original = features_with_data.data.copy()
    features_with_data.standard_scaler(with_mean=False, with_std=False)
    np.testing.assert_array_equal(features_with_data.data, original)


def test_output_shape_unchanged(features_with_data):
    original_shape = features_with_data.data.shape
    features_with_data.standard_scaler()
    assert features_with_data.data.shape == original_shape


def test_single_row_does_not_raise():
    df = pd.DataFrame({"a": [5.0], "b": [99.0]})
    f = Features()
    f.load_new_data(df)
    # StandardScaler with a single sample produces NaN when with_std=True;
    # verify it at least runs without raising an exception.
    f.standard_scaler(with_mean=True, with_std=True)


def test_constant_column_produces_zero_or_nan():
    df = pd.DataFrame({"a": [7.0, 7.0, 7.0], "b": [1.0, 2.0, 3.0]})
    f = Features()
    f.load_new_data(df)
    f.standard_scaler(with_mean=True, with_std=True)
    # Constant column → std=0 → StandardScaler outputs 0.0 for that column
    assert np.all(f.data[:, 0] == 0.0) or np.all(np.isnan(f.data[:, 0]))


def test_modifies_data_in_place(features_with_data):
    original = features_with_data.data.copy()
    features_with_data.standard_scaler()
    assert not np.array_equal(features_with_data.data, original)


def test_default_args_equal_both_true():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    f1 = Features()
    f1.load_new_data(df)
    f1.standard_scaler()

    f2 = Features()
    f2.load_new_data(df)
    f2.standard_scaler(with_mean=True, with_std=True)

    np.testing.assert_array_equal(f1.data, f2.data)


def test_min_max_output_shape_unchanged(features_with_data):
    original_shape = features_with_data.data.shape
    features_with_data.min_max_scaler()
    assert features_with_data.data.shape == original_shape


def test_min_max_modifies_data_in_place(features_with_data):
    original = features_with_data.data.copy()
    features_with_data.min_max_scaler()
    assert not np.array_equal(features_with_data.data, original)


def test_min_max_scaler_output(features_with_data):
    features = deepcopy(features_with_data)
    features.min_max_scaler()
    assert features.min() == 0
    assert features.max() == 1

    features = deepcopy(features_with_data)
    features.min_max_scaler(min_val=-1, max_val=2)
    assert features.min() == -1
    assert features.max() == 2

    try:
        features = deepcopy(features_with_data)
        features.min_max_scaler(3, 1)
    except ValueError:
        pass
    else:
        assert False


def test_min():
    features, _ = small_features()
    assert features.min() == 1


def test_max():
    features, _ = small_features()
    assert features.max() == 6


def test_mean():
    features, _ = small_features()
    assert features.mean() == np.mean([[1, 4], [2, 5], [3, 6]])


def test_std():
    features, _ = small_features()
    expected = np.std([[1, 4], [2, 5], [3, 6]])
    assert np.isclose(features.std(), expected)


def test_median():
    features, _ = small_features()
    assert features.median() == 3.5


def test_wrong_rows_error_message():
    features, _ = small_features()  # 3 x 2
    with pytest.raises(ValueError, match="new data must have same shape as existing data"):
        features.update_data(np.array([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows instead of 3


def test_wrong_cols_error_message():
    features, _ = small_features()
    with pytest.raises(ValueError, match="new data must have same shape as existing data"):
        features.update_data(np.array([[1.0], [2.0], [3.0]]))  # 1 col instead of 2


def test_data_not_updated_on_row_mismatch():
    features, _ = small_features()
    original = features.data.copy()
    with pytest.raises(ValueError):
        features.update_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(features.data, original)


def test_data_not_updated_on_col_mismatch():
    features, _ = small_features()
    original = features.data.copy()
    with pytest.raises(ValueError):
        features.update_data(np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_array_equal(features.data, original)


def test_extra_dimension_raises():
    features, _ = small_features()
    with pytest.raises(ValueError):
        features.update_data(np.ones((3, 2, 2)))  # same rows and cols, one dimension too many


def test_update_data_replaces_data_and_resets_preprocessing():
    features, _ = small_features()
    features.standard_scaler()
    new = np.array([[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]])
    features.update_data(new)
    np.testing.assert_array_equal(features.data, new)
    np.testing.assert_array_equal(features.raw_data, new)
    assert features.preprocessing == []


# -------------------------
# preprocessing is recorded so training can refit it per split
# -------------------------

def test_raw_data_is_kept_after_scaling(features_with_data):
    raw = features_with_data.data.copy()
    features_with_data.standard_scaler()
    features_with_data.min_max_scaler()
    np.testing.assert_array_equal(features_with_data.raw_data, raw)


def test_preprocessing_records_steps_in_order(features_with_data):
    features_with_data.standard_scaler(with_mean=False)
    features_with_data.polynomial_features(degree=2)
    steps = features_with_data.preprocessing
    assert [type(s).__name__ for s in steps] == ["StandardScaler", "PolynomialFeatures"]
    assert steps[0].with_mean is False and steps[1].degree == 2


def test_preprocessing_returns_unfitted_copies(features_with_data):
    features_with_data.standard_scaler()
    first, second = features_with_data.preprocessing[0], features_with_data.preprocessing[0]
    assert first is not second
    assert not hasattr(first, "mean_")


def test_replaying_steps_on_raw_data_reproduces_data(features_with_data):
    features_with_data.min_max_scaler()
    features_with_data.polynomial_features(degree=2)
    x = features_with_data.raw_data
    for step in features_with_data.preprocessing:
        x = step.fit_transform(x)
    np.testing.assert_allclose(x, features_with_data.data)


def test_polynomial_features_adds_columns(features_with_data):
    features_with_data.polynomial_features(degree=2, include_bias=False)
    assert features_with_data.data.shape == (150, 14)  # 4 + 4 squares + 6 pairs


def test_load_new_data_resets_preprocessing(features_with_data):
    features_with_data.standard_scaler()
    features_with_data.load_new_data(pd.DataFrame({"a": [1.0, 2.0]}))
    assert features_with_data.preprocessing == []
