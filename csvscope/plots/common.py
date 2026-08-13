"""Briques partagées par tous les graphiques : styles de traces, légendes, titres."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from .. import interactive
from ..metrics import metric_label
from ..theme import ColorEncoder, StyleEncoder, axis_title, layout_title

__all__ = [
    "TraceStyler",
    "curve_title",
    "finalize",
    "resolve_quantities",
    "resolve_metric_spec",
    "metric_column",
    "format_number",
    "subtitle_for",
]


def format_number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


class TraceStyler:
    """Attribue couleur, style de trait et entrée de légende à chaque configuration.

    Les canaux visuels sont pilotés par des caractéristiques (``color``, ``dash``).
    La légende ne montre qu'une entrée par combinaison de canaux, et cliquer cette
    entrée masque tout le groupe : indispensable quand des dizaines de courbes se
    superposent.
    """

    def __init__(
        self,
        dataset,
        color: str | None = None,
        dash: str | None = None,
        palette: Sequence[str] | None = None,
        legend_per_config: bool | None = None,
    ):
        self.dataset = dataset
        self.color_key = color
        self.dash_key = dash

        if color is not None:
            self.color_encoder = ColorEncoder(dataset.values(color), palette=palette)
        else:
            self.color_encoder = ColorEncoder(
                dataset.configs, palette=palette, numeric=False
            )

        self.dash_encoder = (
            StyleEncoder(dataset.values(dash)) if dash is not None else None
        )
        self.legend_per_config = (
            color is None if legend_per_config is None else legend_per_config
        )
        self._seen: set[str] = set()

    @property
    def legend_title(self) -> str:
        keys = [key for key in (self.color_key, self.dash_key) if key]
        return " · ".join(keys) if keys else "configuration"

    def group_of(self, config: str) -> str:
        if self.legend_per_config:
            return self.dataset.label(config)
        row = self.dataset.meta.loc[config]
        parts = [
            format_number(row[key])
            for key in (self.color_key, self.dash_key)
            if key is not None
        ]
        return " · ".join(parts)

    def color_of(self, config: str) -> str:
        if self.color_key is None:
            return self.color_encoder.color(config)
        return self.color_encoder.color(self.dataset.meta.at[config, self.color_key])

    def dash_of(self, config: str) -> str:
        if self.dash_encoder is None:
            return "solid"
        return self.dash_encoder.style(self.dataset.meta.at[config, self.dash_key])

    def style(self, config: str, force_hidden_legend: bool = False) -> dict[str, Any]:
        group = self.group_of(config)
        first = group not in self._seen
        self._seen.add(group)
        return {
            "line": {"color": self.color_of(config), "dash": self.dash_of(config)},
            "name": group,
            "legendgroup": group,
            "showlegend": bool(first and not force_hidden_legend),
            "meta": {
                "label": self.dataset.label(config),
                "config": config,
                "group": group,
            },
        }

    def reset_legend(self) -> None:
        self._seen.clear()


def resolve_quantities(dataset, quantities: str | Sequence[str] | None) -> list[str]:
    if quantities is None:
        return list(dataset.quantities)
    if isinstance(quantities, str):
        quantities = [quantities]
    unknown = [q for q in quantities if q not in dataset.quantities]
    if unknown:
        raise KeyError(
            f"Grandeurs inconnues : {unknown}. Disponibles : {dataset.quantities}"
        )
    return list(quantities)


def resolve_metric_spec(spec: str | tuple[str, Any], default_metric: Any) -> tuple[str, Any]:
    """Accepte ``"temperature"`` ou ``("temperature", "mean")``."""
    if isinstance(spec, tuple):
        quantity, metric = spec
        return quantity, metric
    return spec, default_metric


def metric_column(quantity: str, metric_name: str, n_metrics: int) -> str:
    """Nom de colonne produit par :meth:`Dataset.table` (cohérence garantie)."""
    return quantity if n_metrics == 1 else f"{quantity}_{metric_name}"


def curve_title(dataset, quantity: str, x: str) -> str:
    """Titre lisible selon que l'abscisse soit le temps ou une autre grandeur."""
    if x == dataset.time:
        return f"Évolution de {quantity} au cours du temps"
    return f"{quantity} en fonction de {x}"


def subtitle_for(dataset, extra: str | None = None) -> str:
    base = f"{len(dataset)} configurations"
    characteristics = dataset.characteristics
    if characteristics:
        base += " · " + ", ".join(characteristics)
    return f"{extra} · {base}" if extra else base


def finalize(
    figure: go.Figure,
    dataset,
    title: str | None = None,
    subtitle: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    legend_title: str | None = None,
    height: int | None = 520,
    width: int | None = None,
    hovermode: str = "closest",
    interactivity: Mapping[str, Any] | bool | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Applique titres, tailles et interactivité de façon homogène."""
    layout: dict[str, Any] = {"hovermode": hovermode, "showlegend": show_legend}
    if title is not None:
        layout["title"] = layout_title(title, subtitle)
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    if legend_title is not None:
        layout["legend"] = {"title": {"text": legend_title}}
    figure.update_layout(**layout)

    if x_title is not None:
        figure.update_xaxes(title_text=axis_title(x_title, dataset.units))
    if y_title is not None:
        figure.update_yaxes(title_text=axis_title(y_title, dataset.units))

    if interactivity is False:
        interactive.configure(figure, highlight=False, search=False)
    elif isinstance(interactivity, Mapping):
        interactive.configure(figure, **interactivity)
    else:
        interactive.configure(figure)
    return figure


def hover_lines(dataset, config: str, extra: Iterable[str] = ()) -> str:
    parts = [f"<b>{dataset.label(config)}</b>"]
    for key, value in dataset.characteristics_of(config).items():
        parts.append(f"{key} : {format_number(value)}")
    parts.extend(extra)
    return "<br>".join(parts)


def metric_axis_title(quantity: str, metric: Any, units: Mapping[str, str]) -> str:
    unit = units.get(quantity)
    label = f"{metric_label(metric)} de {quantity}"
    return f"{label} [{unit}]" if unit else label


def characteristic_axis(figure: go.Figure, dataset, characteristic: str, axis: str = "x") -> None:
    """Force l'ordre naturel des valeurs sur un axe catégoriel."""
    values = dataset.values(characteristic)
    if pd.api.types.is_numeric_dtype(pd.Series(values).infer_objects()):
        return
    setter = figure.update_xaxes if axis == "x" else figure.update_yaxes
    setter(categoryorder="array", categoryarray=[str(v) for v in values])
