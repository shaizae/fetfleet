"""Isolated plot execution: every visualization succeeds, is skipped or fails on its own.

Nothing here may raise into the caller. `VisualizationManager.run()` returns a
`VisualizationReport` even if every plot fails; `safe_call()` does the same for any
callable (used around PDF building in the experiment pipeline).
"""
from __future__ import annotations

import io
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ml_pipe.visualization.context import VisualizationContext

logger = logging.getLogger("ml_pipe.visualization")

SECTIONS = (
    "Overall Results and Model Comparison",
    "Classification Analysis",
    "Regression Analysis",
    "Feature Analysis",
    "Cross-Validation Analysis",
    "Hyperparameter Optimization",
    "Runtime and Resource Analysis",
)

DPI = 150


class VizStatus(StrEnum):
    """Outcome of one visualization."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class SkipVisualization(Exception):
    """Raised by a plot whose data does not support it; the message is the reason."""


@dataclass
class Rendered:
    """What a plot's render function returns.

    figure   the matplotlib Figure drawn (closed by the manager)
    models   the model / configuration labels the plot shows
    metrics  the values the plot was drawn from, printed under it in the PDF
    note     an optional extra caption line (e.g. "top 6 of 9 models shown")
    """
    figure: Figure
    models: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class VisualizationResult:
    """The standardized record of one visualization, whatever its outcome."""
    key: str
    title: str
    section: str
    description: str
    status: VizStatus
    image: bytes | None = None          # PNG
    size_px: tuple[int, int] | None = None
    models: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    reason: str = ""                    # why it was skipped / the error message
    error_type: str = ""
    traceback: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the plot was generated."""
        return self.status is VizStatus.SUCCESS


@dataclass
class Visualization:
    """A registered plot type.

    items(ctx) -> the things to draw one plot each for (default: one plot, item None);
    render(ctx, item) -> Rendered, or raises SkipVisualization.
    tasks: "classification" / "regression"; scopes: "experiment" (ExperimentReport PDF)
    and / or "result" (the per-model PDF).
    """
    key: str
    title: str
    section: str
    description: str
    render: Callable[[VisualizationContext, Any], Rendered]
    tasks: tuple[str, ...] = ("classification", "regression")
    scopes: tuple[str, ...] = ("experiment",)
    items: Callable[[VisualizationContext], Iterable[Any]] | None = None
    item_label: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        if self.section not in SECTIONS:
            raise ValueError(f"section must be one of {SECTIONS}, not {self.section!r}")

    def applies_to(self, ctx: VisualizationContext) -> bool:
        """Whether this plot type is meant for the context's task and scope at all."""
        return ctx.task in self.tasks and ctx.scope in self.scopes


_REGISTRY: list[Visualization] = []


def register(viz: Visualization) -> Visualization:
    """Add a plot type to the default registry (replaces one with the same key)."""
    for i, existing in enumerate(_REGISTRY):
        if existing.key == viz.key:
            _REGISTRY[i] = viz
            return viz
    _REGISTRY.append(viz)
    return viz


def registry() -> list[Visualization]:
    """The registered plot types, in registration (= PDF) order."""
    from ml_pipe.visualization import plots  # noqa: F401  (registers the built-in plots)
    return list(_REGISTRY)


@dataclass
class VisualizationReport:
    """Status tracker: every visualization attempted for one report."""
    results: list[VisualizationResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def by_status(self, status: VizStatus) -> list[VisualizationResult]:
        """The results with this status, in order."""
        return [r for r in self.results if r.status is status]

    @property
    def succeeded(self) -> list[VisualizationResult]:
        return self.by_status(VizStatus.SUCCESS)

    @property
    def skipped(self) -> list[VisualizationResult]:
        return self.by_status(VizStatus.SKIPPED)

    @property
    def failed(self) -> list[VisualizationResult]:
        return self.by_status(VizStatus.FAILED)

    def counts(self) -> dict[str, int]:
        """{status: count} for every status."""
        return {s.value: len(self.by_status(s)) for s in VizStatus}

    def summary(self) -> str:
        """e.g. "18 generated, 3 skipped, 0 failed"."""
        c = self.counts()
        return f"{c['success']} generated, {c['skipped']} skipped, {c['failed']} failed"

    def without_images(self) -> VisualizationReport:
        """The same records with the PNG bytes dropped (to keep after the PDF is written)."""
        from dataclasses import replace
        return VisualizationReport([replace(r, image=None) for r in self.results])

    def save_images(self, folder) -> list:
        """Write every generated plot as <key>.png into `folder`; returns the paths."""
        from pathlib import Path
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for r in self.succeeded:
            path = folder / (_safe_name(r.key) + ".png")
            path.write_bytes(r.image)
            paths.append(path)
        return paths


def _safe_name(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


def _figure_to_png(fig: Figure) -> tuple[bytes, tuple[int, int]]:
    """Draw with Agg (no pyplot, no GUI backend); return (RGB PNG bytes, (width, height) px).

    RGB, not RGBA: the figures are opaque, and an alpha channel would make the PDF carry a
    second (mask) image per figure."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    fig.set_dpi(DPI)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3]
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", dpi=(DPI, DPI), optimize=True)
    return buffer.getvalue(), (rgb.shape[1], rgb.shape[0])


def _release(fig) -> None:
    """Free a figure's artists; never raises."""
    if fig is None:
        return
    try:
        fig.clear()
    except Exception:  # pragma: no cover - nothing else to do
        pass


class VisualizationManager:
    """Runs plot types against a context, each item in its own error boundary."""

    def __init__(self, visualizations: Iterable[Visualization] | None = None) -> None:
        """visualizations: the plot types to run (default: the registry)."""
        self._visualizations = list(visualizations) if visualizations is not None else None

    @property
    def visualizations(self) -> list[Visualization]:
        return self._visualizations if self._visualizations is not None else registry()

    def run_for(self, build: Callable[[], VisualizationContext]) -> VisualizationReport:
        """Build the context with build(), then run(); a failure to build it is recorded, not raised."""
        try:
            ctx = build()
        except Exception as e:
            logger.warning("visualization context could not be built: %s: %s", type(e).__name__, e)
            logger.debug("traceback", exc_info=True)
            return VisualizationReport([_failure("context", "Visualization input", SECTIONS[0],
                                                 "Snapshot of the results the plots are drawn from.", e, 0.0)])
        return self.run(ctx)

    def run(self, ctx: VisualizationContext) -> VisualizationReport:
        """Every applicable plot for `ctx`. Never raises (except KeyboardInterrupt / SystemExit)."""
        report = VisualizationReport()
        try:
            specs = [v for v in self.visualizations if v.applies_to(ctx)]
        except Exception as e:  # a broken registry must not stop the report
            logger.warning("visualization registry failed: %s", e, exc_info=True)
            report.results.append(_failure("registry", "Visualization registry", SECTIONS[0], "", e, 0.0))
            return report
        for spec in specs:
            report.results.extend(self._run_spec(spec, ctx))
        logger.info("visualizations: %s", report.summary())
        return report

    def _run_spec(self, spec: Visualization, ctx: VisualizationContext) -> list[VisualizationResult]:
        start = time.perf_counter()
        try:
            items = list(spec.items(ctx)) if spec.items is not None else [None]
        except SkipVisualization as e:
            return [_record(spec, spec.key, spec.title, VizStatus.SKIPPED, reason=str(e),
                            seconds=time.perf_counter() - start)]
        except Exception as e:
            logger.warning("visualization %s failed while listing its items: %s", spec.key, e)
            logger.debug("traceback", exc_info=True)
            return [_failure(spec.key, spec.title, spec.section, spec.description, e,
                             time.perf_counter() - start)]
        if not items:
            return [_record(spec, spec.key, spec.title, VizStatus.SKIPPED, reason="nothing to plot",
                            seconds=time.perf_counter() - start)]
        return [self._run_one(spec, ctx, item) for item in items]

    @staticmethod
    def _run_one(spec: Visualization, ctx: VisualizationContext, item: Any) -> VisualizationResult:
        label = ""
        if item is not None:
            try:
                label = spec.item_label(item) if spec.item_label else str(getattr(item, "label", item))
            except Exception:
                label = "?"
        key = f"{spec.key}:{label}" if label else spec.key
        title = f"{spec.title} - {label}" if label else spec.title
        start = time.perf_counter()
        fig = None
        try:
            from ml_pipe.visualization.style import theme, wrap_titles
            with theme():
                rendered = spec.render(ctx, item)
                fig = rendered.figure
                wrap_titles(fig)
                image, size = _figure_to_png(fig)
            return _record(spec, key, title, VizStatus.SUCCESS, image=image, size_px=size,
                           models=tuple(rendered.models), metrics=dict(rendered.metrics), note=rendered.note,
                           seconds=time.perf_counter() - start)
        except SkipVisualization as e:
            logger.info("visualization %s skipped: %s", key, e)
            return _record(spec, key, title, VizStatus.SKIPPED, reason=str(e), seconds=time.perf_counter() - start)
        except ImportError as e:
            logger.info("visualization %s skipped (missing dependency): %s", key, e)
            return _record(spec, key, title, VizStatus.SKIPPED, reason=f"missing optional dependency: {e}",
                           seconds=time.perf_counter() - start)
        except Exception as e:
            logger.warning("visualization %s failed: %s: %s", key, type(e).__name__, e)
            logger.debug("traceback of %s", key, exc_info=True)
            return _failure(key, title, spec.section, spec.description, e, time.perf_counter() - start)
        finally:
            _release(fig)


def _record(spec: Visualization, key: str, title: str, status: VizStatus, **kwargs) -> VisualizationResult:
    return VisualizationResult(key=key, title=title, section=spec.section, description=spec.description,
                               status=status, **kwargs)


def _failure(key: str, title: str, section: str, description: str, error: BaseException,
             seconds: float) -> VisualizationResult:
    return VisualizationResult(key=key, title=title, section=section, description=description,
                               status=VizStatus.FAILED, reason=str(error) or type(error).__name__,
                               error_type=type(error).__name__,
                               traceback="".join(traceback.format_exception(error)), seconds=seconds)


def safe_call(func: Callable[..., Any], *args, what: str = "report", log: Callable[[str], None] | None = None,
              **kwargs) -> tuple[Any, BaseException | None]:
    """Call func(*args, **kwargs); return (value, None), or (None, error) after logging it.

    For reporting steps that must never interrupt the caller (KeyboardInterrupt / SystemExit pass)."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        logger.warning("%s failed: %s: %s", what, type(e).__name__, e)
        logger.debug("traceback of %s", what, exc_info=True)
        if log is not None:
            try:
                log(f"warning: {what} failed ({type(e).__name__}: {e}); the experiment results are unaffected")
            except Exception:
                pass
        return None, e
