"""Run many scikit-learn models x feature subsets x hyper-parameters in parallel and keep the best."""
from fitfleet.classification import Classification
from fitfleet.feature_research import FeatureResearch
from fitfleet.model import SavedModel, load
from fitfleet.regression import Regression
from fitfleet.tracking import ExecutionLog, ExecutionTracker
from fitfleet.utils import FilteringCriteriaClassification, FilteringCriteriaRegression, Sound, ValidationType

__version__ = "0.2.0"

__all__ = ["Classification", "Regression", "FeatureResearch", "FilteringCriteriaClassification", "FilteringCriteriaRegression",
           "ValidationType", "Sound", "SavedModel", "load", "ExecutionTracker", "ExecutionLog", "__version__"]
