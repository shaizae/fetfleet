"""Built-in plot types; importing this package registers them (in PDF order)."""
from fitfleet.visualization.plots import comparison as _comparison  # noqa: F401  (overall + optimization)
from fitfleet.visualization.plots import classification as _classification  # noqa: F401
from fitfleet.visualization.plots import regression as _regression  # noqa: F401
from fitfleet.visualization.plots import features as _features  # noqa: F401
