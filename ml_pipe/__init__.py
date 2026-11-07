"""Run many scikit-learn models x feature subsets x hyper-parameters in parallel and keep the best."""
from ml_pipe.classification import Classification
from ml_pipe.regression import Regression
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, ValidationType

__version__ = "0.2.0"

__all__ = ["Classification", "Regression", "FilteringCriteriaClassification", "FilteringCriteriaRegression",
           "ValidationType", "__version__"]
