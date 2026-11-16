import pytest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier

import fitfleet
from fitfleet import Classification


def test_public_api():
    assert fitfleet.__version__
    for name in fitfleet.__all__:
        assert hasattr(fitfleet, name)


def test_old_names_still_work_with_a_deprecation_warning(iris_data):
    cls = Classification()
    cls.set(*iris_data, [DecisionTreeClassifier()])
    with pytest.deprecated_call():
        assert cls.fetchers is cls.features
    with pytest.deprecated_call():
        cls.fetcher_selection(f_classif, [2])
    assert cls._features_index and len(cls._features_index[0]) == 2
    with pytest.deprecated_call():
        cls.brut_force_features()
    assert len(cls._features_index) == 2 ** 4 - 1
