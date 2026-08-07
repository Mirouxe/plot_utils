"""Métriques scalaires résumant une série temporelle en un nombre par configuration.

Chaque métrique est une fonction ``(valeurs, temps) -> float`` enregistrée sous un
nom court, ce qui permet d'écrire ``ds.metrics("temperature", ["max", "t_max"])``.
Une métrique maison s'ajoute avec :class:`register_metric` ou se passe directement
sous forme de callable.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "METRICS",
    "METRIC_LABELS",
    "register_metric",
    "resolve_metric",
    "metric_label",
    "apply_metric",
]

MetricFunc = Callable[[pd.Series, pd.Series], float]

METRICS: dict[str, MetricFunc] = {}
METRIC_LABELS: dict[str, str] = {}


def register_metric(name: str, label: str | None = None):
    """Décorateur enregistrant une métrique sous un nom utilisable partout."""

    def decorator(func: MetricFunc) -> MetricFunc:
        METRICS[name] = func
        METRIC_LABELS[name] = label or name
        return func

    return decorator


def _clean(values: pd.Series, time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(t)
    return y[mask], t[mask]


@register_metric("max", "maximum")
def _max(values, time):
    y, _ = _clean(values, time)
    return float(np.max(y)) if y.size else np.nan


@register_metric("min", "minimum")
def _min(values, time):
    y, _ = _clean(values, time)
    return float(np.min(y)) if y.size else np.nan


@register_metric("mean", "moyenne")
def _mean(values, time):
    y, _ = _clean(values, time)
    return float(np.mean(y)) if y.size else np.nan


@register_metric("median", "médiane")
def _median(values, time):
    y, _ = _clean(values, time)
    return float(np.median(y)) if y.size else np.nan


@register_metric("std", "écart-type")
def _std(values, time):
    y, _ = _clean(values, time)
    return float(np.std(y, ddof=1)) if y.size > 1 else np.nan


@register_metric("rms", "valeur efficace (RMS)")
def _rms(values, time):
    y, _ = _clean(values, time)
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else np.nan


@register_metric("range", "amplitude (max - min)")
def _range(values, time):
    y, _ = _clean(values, time)
    return float(np.max(y) - np.min(y)) if y.size else np.nan


@register_metric("final", "valeur finale")
def _final(values, time):
    y, t = _clean(values, time)
    return float(y[np.argmax(t)]) if y.size else np.nan


@register_metric("initial", "valeur initiale")
def _initial(values, time):
    y, t = _clean(values, time)
    return float(y[np.argmin(t)]) if y.size else np.nan


@register_metric("integral", "intégrale temporelle")
def _integral(values, time):
    y, t = _clean(values, time)
    if y.size < 2:
        return np.nan
    order = np.argsort(t)
    return float(np.trapezoid(y[order], t[order]))


@register_metric("t_max", "instant du maximum")
def _t_max(values, time):
    y, t = _clean(values, time)
    return float(t[np.argmax(y)]) if y.size else np.nan


@register_metric("t_min", "instant du minimum")
def _t_min(values, time):
    y, t = _clean(values, time)
    return float(t[np.argmin(y)]) if y.size else np.nan


@register_metric("overshoot", "dépassement relatif au régime final")
def _overshoot(values, time):
    y, t = _clean(values, time)
    if y.size < 2:
        return np.nan
    final = y[np.argmax(t)]
    if final == 0:
        return np.nan
    return float((np.max(y) - final) / abs(final))


@register_metric("slope", "pente moyenne (régression linéaire)")
def _slope(values, time):
    y, t = _clean(values, time)
    if y.size < 2 or np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, y, 1)[0])


def resolve_metric(metric: str | MetricFunc) -> MetricFunc:
    """Retourne la fonction associée à un nom de métrique (ou le callable fourni)."""
    if callable(metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError:
        raise KeyError(
            f"Métrique inconnue : {metric!r}. Disponibles : {sorted(METRICS)}"
        ) from None


def metric_label(metric: str | MetricFunc) -> str:
    """Nom lisible d'une métrique, pour les titres et les axes."""
    if callable(metric):
        return getattr(metric, "__name__", "métrique")
    return METRIC_LABELS.get(metric, metric)


def apply_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str | MetricFunc,
    time_column: str,
) -> float:
    """Applique une métrique à une colonne d'un DataFrame de configuration."""
    func = resolve_metric(metric)
    return func(frame[quantity], frame[time_column])


def normalize_metrics(
    metrics: str | MetricFunc | Iterable[str | MetricFunc] | Mapping[str, MetricFunc],
) -> dict[str, str | MetricFunc]:
    """Uniformise l'argument ``metrics`` en dictionnaire ``nom -> métrique``."""
    if isinstance(metrics, Mapping):
        return dict(metrics)
    if isinstance(metrics, str) or callable(metrics):
        metrics = [metrics]
    result: dict[str, str | MetricFunc] = {}
    for item in metrics:
        name = item if isinstance(item, str) else getattr(item, "__name__", "metric")
        result[name] = item
    return result
