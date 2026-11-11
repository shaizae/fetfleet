"""Visualization subsystem: isolated plots and PDF reports built from collected results.

Execution, aggregation, visualization and reporting are separate steps:

    results  ->  VisualizationContext (read-only snapshot)
             ->  VisualizationManager.run()  ->  VisualizationReport (SUCCESS / SKIPPED / FAILED each)
             ->  PdfReport.build()

No plot can raise into the caller; see docs/visualization.md.
"""
from ml_pipe.visualization.core import (SECTIONS, Rendered, SkipVisualization, Visualization, VisualizationManager,
                                        VisualizationReport, VisualizationResult, VizStatus, register, registry,
                                        safe_call)
from ml_pipe.visualization.context import ResultView, VisualizationContext

__all__ = ["SECTIONS", "Rendered", "SkipVisualization", "Visualization", "VisualizationManager",
           "VisualizationReport", "VisualizationResult", "VizStatus", "register", "registry", "safe_call",
           "ResultView", "VisualizationContext"]
