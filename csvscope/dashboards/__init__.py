"""Dashboards d'exploration : fiche d'un CSV, comparaison de deux, et outils associés."""

from __future__ import annotations

from .common import resolve_config
from .diff import diff, diff_files, diff_figures
from .inspect import inspect, inspect_file, inspect_figures
from .tools import coverage, neighbors, outliers, quality, quantity_board, snapshot

__all__ = [
    "inspect",
    "inspect_file",
    "inspect_figures",
    "diff",
    "diff_files",
    "diff_figures",
    "quantity_board",
    "snapshot",
    "outliers",
    "neighbors",
    "coverage",
    "quality",
    "resolve_config",
]
