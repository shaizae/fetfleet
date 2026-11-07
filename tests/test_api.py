import pytest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier

import ml_pipe
from ml_pipe import Classification


def test_public_api():
    assert ml_pipe.__version__
    for name in ml_pipe.__all__:
        assert hasattr(ml_pipe, name)


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
