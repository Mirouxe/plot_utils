"""Galerie de graphiques interactifs construits sur un :class:`csvscope.Dataset`."""

from .curves import curves, envelope, explorer, grid, small_multiples
from .multivariate import parallel, radar
from .summary import bars, compare, distribution, heatmap, metrics_table, scatter

__all__ = [
    "curves",
    "grid",
    "explorer",
    "envelope",
    "small_multiples",
    "compare",
    "bars",
    "heatmap",
    "distribution",
    "scatter",
    "metrics_table",
    "parallel",
    "radar",
]
