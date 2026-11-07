import numpy as np

from ml_pipe.target import Target


def test_target_encoding():
    target = Target()

    labels = np.array(["cat", "dog", "cat", "bird"])

    target.load_new_data(labels)

    assert len(target.data) == 4

    assert set(target.data) == {0, 1, 2}

    assert set(target.decrypt_map.keys()) == {"bird", "cat", "dog", }


def test_target_output_is_numeric():
    target = Target()

    target.load_new_data(np.array(["a", "b", "c"]))

    assert np.issubdtype(target.data.dtype, np.integer)


def test_pop_target_data():
    target = Target()
    target.load_new_data(np.array(["a", "b", "c", "d", "e", "f"]))

    popped = target.pop_index([1, 5, 3])
    assert np.array_equal(
        target._encoder.inverse_transform(popped), np.array(["b", "f", "d"]))

    assert np.array_equal(target.show_original_data, np.array(["a", "c", "e"]))


def test_iter():
    target = Target()

    target.load_new_data(
        np.array(["cat", "dog", "cat", "bird"])
    )

    expected = target.data.tolist()

    result = list(target)

    assert result == expected


def test_show_original_data_before_loading_is_none():
    assert Target().show_original_data is None


def test_regression_target_is_not_encoded():
    target = Target(encode=False)
    values = np.array([1000.5, -3.25, 1000.5, 7.0])
    target.load_new_data(values)
    np.testing.assert_array_equal(target.data, values)
    assert target.data.dtype == float
    assert target.decrypt_map is None
    np.testing.assert_array_equal(target.show_original_data, values)


def test_regression_target_pop_index():
    target = Target(encode=False)
    target.load_new_data(np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(target.pop_index([1]), [20.0])
    np.testing.assert_array_equal(target.data, [10.0, 30.0])
