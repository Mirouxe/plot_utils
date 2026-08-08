"""Vues multi-grandeurs : coordonnées parallèles et radar comparatif."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..metrics import metric_label
from ..theme import ColorEncoder, SEQUENTIAL, sort_values_naturally
from .common import (
    finalize,
    format_number,
    resolve_quantities,
    subtitle_for,
)

__all__ = ["parallel", "radar"]


def _numeric_dimension(values: Sequence[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Convertit une colonne (numérique ou catégorielle) en axe de coordonnées."""
    series = pd.Series(list(values))
    if pd.api.types.is_numeric_dtype(series.infer_objects()):
        return series.to_numpy(dtype=float), {}
    order = sort_values_naturally(series.tolist())
    codes = series.map({value: index for index, value in enumerate(order)})
    return codes.to_numpy(dtype=float), {
        "tickvals": list(range(len(order))),
        "ticktext": [str(value) for value in order],
        "range": [-0.4, len(order) - 0.6],
    }


def parallel(
    dataset,
    quantities: Sequence[str] | str | None = None,
    metric: Any = "max",
    color: str | None = None,
    characteristics: Sequence[str] | bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 560,
    width: int | None = None,
    colorscale: Sequence[str] | None = None,
) -> go.Figure:
    """Coordonnées parallèles : caractéristiques et grandeurs sur un même graphique.

    Chaque ligne est une configuration. En glissant la souris le long d'un axe on
    sélectionne une plage et seules les configurations correspondantes restent
    visibles : c'est le moyen le plus direct de répondre à « quelles configurations
    tiennent tel critère sur plusieurs grandeurs à la fois ? ».

    >>> ds.parallel(["temperature", "pression"], metric="max", color="puissance")
    """
    quantities = resolve_quantities(dataset, quantities)
    table = dataset.table(quantities, metric)

    if characteristics is True:
        keys = [c for c in dataset.characteristics if len(dataset.values(c)) > 1]
    elif characteristics is False:
        keys = []
    else:
        keys = list(characteristics)

    dimensions = []
    for key in keys:
        values, options = _numeric_dimension(table[key].tolist())
        dimensions.append({"label": key, "values": values, **options})
    for quantity in quantities:
        unit = dataset.units.get(quantity)
        label = f"{quantity} [{unit}]" if unit else quantity
        dimensions.append(
            {"label": label, "values": table[quantity].to_numpy(dtype=float)}
        )

    if color and color in table.columns:
        color_values, _ = _numeric_dimension(table[color].tolist())
        color_title = color
    else:
        color_values = table[quantities[0]].to_numpy(dtype=float)
        color_title = f"{metric_label(metric)} {quantities[0]}"

    scale = list(colorscale or SEQUENTIAL)
    figure = go.Figure(
        go.Parcoords(
            line={
                "color": color_values,
                "colorscale": [[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
                "showscale": True,
                "colorbar": {"title": {"text": color_title, "side": "right"},
                             "thickness": 14, "outlinewidth": 0},
            },
            dimensions=dimensions,
            labelangle=0,
            labelside="top",
            unselected={"line": {"color": "#d1d5db", "opacity": 0.25}},
        )
    )
    figure.update_layout(margin={"l": 90, "r": 90, "t": 110, "b": 40})

    return finalize(
        figure,
        dataset,
        title=title or f"Coordonnées parallèles · {metric_label(metric)} des grandeurs",
        subtitle=subtitle if subtitle is not None
        else subtitle_for(dataset, "glisse sur un axe pour filtrer"),
        height=height,
        width=width,
        show_legend=False,
        interactivity=False,
    )


def radar(
    dataset,
    quantities: Sequence[str] | str | None = None,
    metric: Any = "max",
    group: str | None = None,
    normalize: bool | str = "max",
    aggregate: str = "mean",
    max_traces: int = 12,
    fill: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 620,
    width: int | None = None,
    palette: Sequence[str] | None = None,
) -> go.Figure:
    """Radar comparant plusieurs grandeurs pour des configurations ou des groupes.

    Les grandeurs n'ayant pas les mêmes ordres de grandeur, les axes sont normalisés ;
    le survol affiche toujours la valeur physique réelle.

    Args:
        group: caractéristique agrégeant les configurations (une trace par valeur).
        normalize: ``"max"`` (chaque axe rapporté à son maximum, proportions
            conservées), ``"minmax"`` (axes recalés min–max, écarts accentués)
            ou ``False`` (valeurs brutes, à réserver aux grandeurs comparables).
        max_traces: garde-fou de lisibilité quand on trace configuration par configuration.
    """
    quantities = resolve_quantities(dataset, quantities)
    if len(quantities) < 3:
        raise ValueError("Un radar demande au moins trois grandeurs.")

    table = dataset.table(quantities, metric)
    truncated = False

    if group is not None:
        grouped = table.groupby(group, dropna=False)[list(quantities)]
        raw = getattr(grouped, aggregate)()
        raw = raw.reindex(sort_values_naturally(raw.index.tolist()))
        names = [format_number(value) for value in raw.index]
        counts = grouped.size().reindex(raw.index).tolist()
    else:
        raw = table[list(quantities)]
        names = table["label"].tolist()
        counts = [1] * len(raw)
        truncated = len(raw) > max_traces
        if truncated:
            raw = raw.head(max_traces)
            names = names[:max_traces]
            counts = counts[:max_traces]

    mode = "max" if normalize is True else (None if not normalize else str(normalize))
    columns = list(quantities)
    if mode == "max":
        normalized = raw / table[columns].abs().max().replace(0, np.nan)
        hint = "valeurs rapportées au maximum observé"
        radial: dict[str, Any] = {"range": [0, 1.05], "tickformat": ".0%"}
    elif mode == "minmax":
        low, high = table[columns].min(), table[columns].max()
        # Plancher à 10 % : la configuration la plus basse reste visible.
        normalized = 0.1 + 0.9 * (raw - low) / (high - low).replace(0, np.nan)
        hint = "chaque axe recalé entre le minimum et le maximum observés"
        radial = {"range": [0, 1.05], "showticklabels": False, "showline": False}
    elif mode is None:
        normalized = raw
        hint = "valeurs brutes"
        radial = {}
    else:
        raise ValueError("normalize doit valoir True, 'max', 'minmax' ou False.")

    encoder = ColorEncoder(names, palette=palette, numeric=False)
    axis_labels = [
        f"{q} [{dataset.units[q]}]" if q in dataset.units else q for q in quantities
    ]
    closed_labels = [*axis_labels, axis_labels[0]]
    figure = go.Figure()

    for position, name in enumerate(names):
        values = normalized.iloc[position].to_numpy(dtype=float)
        real = raw.iloc[position].to_numpy(dtype=float)
        color = encoder.color(name)
        suffix = f" ({counts[position]} config.)" if group is not None else ""
        figure.add_trace(
            go.Scatterpolar(
                r=np.concatenate([values, values[:1]]),
                theta=closed_labels,
                mode="lines+markers",
                fill="toself" if fill else None,
                fillcolor=_rgba(color, 0.12),
                line={"color": color, "width": 2.2},
                marker={"size": 6, "color": color},
                name=f"{name}{suffix}",
                legendgroup=name,
                meta={"label": name},
                customdata=np.concatenate([real, real[:1]]),
                hovertemplate=(
                    f"<b>{name}</b><br>%{{theta}}<br>"
                    f"{metric_label(metric)} = %{{customdata:.4g}}"
                    + ("<br>normalisé = %{r:.0%}" if mode == "max" else "")
                    + "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        polar={
            "bgcolor": "white",
            "radialaxis": {
                "visible": True,
                "gridcolor": "#e6e6e6",
                "angle": 90,
                "tickfont": {"size": 10, "color": "#6b7280"},
                **radial,
            },
            "angularaxis": {"gridcolor": "#ececec", "linecolor": "#d1d5db"},
        }
    )

    if group is None and truncated:
        hint += f" · {max_traces} premières configurations (utilise group=… ou filter)"
    return finalize(
        figure,
        dataset,
        title=title or f"Radar · {metric_label(metric)} des grandeurs",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset, hint),
        legend_title=group or "configuration",
        height=height,
        width=width,
        interactivity=False,
    )


def _rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
