"""Built-in plot types; importing this package registers them (in PDF order)."""
from ml_pipe.visualization.plots import comparison as _comparison  # noqa: F401  (overall + optimization)
from ml_pipe.visualization.plots import classification as _classification  # noqa: F401
from ml_pipe.visualization.plots import regression as _regression  # noqa: F401
from ml_pipe.visualization.plots import features as _features  # noqa: F401
