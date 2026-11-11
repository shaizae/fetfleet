"""Run many scikit-learn models x feature subsets x hyper-parameters in parallel and keep the best."""
from ml_pipe.classification import Classification
from ml_pipe.feature_research import FeatureResearch
from ml_pipe.model import SavedModel, load
from ml_pipe.regression import Regression
from ml_pipe.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, Sound, ValidationType

__version__ = "0.2.0"

__all__ = ["Classification", "Regression", "FeatureResearch", "FilteringCriteriaClassification", "FilteringCriteriaRegression",
           "ValidationType", "Sound", "SavedModel", "load", "__version__"]
