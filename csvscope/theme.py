"""Thème Plotly commun et gestion des couleurs par caractéristique.

L'objectif est que tous les graphiques de la librairie soient homogènes et lisibles
sans réglage manuel : mêmes polices, même grille discrète, titre avec sous-titre,
et surtout un encodage couleur cohérent d'un graphique à l'autre pour une même
caractéristique (une valeur de caractéristique garde la même couleur partout).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "TEMPLATE_NAME",
    "QUALITATIVE",
    "SEQUENTIAL",
    "install_template",
    "ColorEncoder",
    "sort_values_naturally",
    "axis_title",
    "layout_title",
    "adaptive_line_style",
    "adaptive_max_points",
]

TEMPLATE_NAME = "csvscope"

#: Palette qualitative pour les caractéristiques non numériques.
QUALITATIVE: tuple[str, ...] = (
    "#3b6ea5",
    "#e07a3f",
    "#4c9f70",
    "#b5495b",
    "#8a6bbe",
    "#7d6252",
    "#d081b0",
    "#6d7b8d",
    "#b8a13a",
    "#3f9fb0",
)

#: Échelle continue pour les caractéristiques numériques ordonnées.
SEQUENTIAL: tuple[str, ...] = (
    "#0d3b66",
    "#2a6f97",
    "#3f9fb0",
    "#78c6a3",
    "#d9c56a",
    "#e8963c",
    "#d1495b",
)

GRID_COLOR = "#e6e6e6"
AXIS_COLOR = "#4a4a4a"
FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def install_template(set_default: bool = True) -> go.layout.Template:
    """Enregistre (et active) le template Plotly de la librairie."""
    template = go.layout.Template(
        layout=go.Layout(
            font={"family": FONT_FAMILY, "size": 13, "color": "#22252a"},
            title={
                "font": {"size": 19, "color": "#14171a"},
                "x": 0.01,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
            },
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=list(QUALITATIVE),
            margin={"l": 70, "r": 30, "t": 90, "b": 60},
            hoverlabel={
                "bgcolor": "white",
                "bordercolor": "#c8c8c8",
                "font": {"family": FONT_FAMILY, "size": 12},
                "align": "left",
            },
            legend={
                "bgcolor": "rgba(255,255,255,0.75)",
                "bordercolor": GRID_COLOR,
                "borderwidth": 1,
                "font": {"size": 12},
                "itemsizing": "constant",
                "tracegroupgap": 4,
            },
            xaxis={
                "gridcolor": GRID_COLOR,
                "zerolinecolor": "#cfcfcf",
                "linecolor": AXIS_COLOR,
                "ticks": "outside",
                "tickcolor": GRID_COLOR,
                "showline": True,
                "mirror": False,
                "automargin": True,
            },
            yaxis={
                "gridcolor": GRID_COLOR,
                "zerolinecolor": "#cfcfcf",
                "linecolor": AXIS_COLOR,
                "ticks": "outside",
                "tickcolor": GRID_COLOR,
                "showline": True,
                "mirror": False,
                "automargin": True,
            },
            colorscale={"sequential": [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]},
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    if set_default:
        pio.templates.default = TEMPLATE_NAME
    return template


def _is_numeric_series(values: Sequence[Any]) -> bool:
    series = pd.Series(list(values))
    if series.dropna().empty:
        return False
    return pd.api.types.is_numeric_dtype(series.dropna().infer_objects())


def sort_values_naturally(values: Sequence[Any]) -> list[Any]:
    """Trie des valeurs de caractéristique en respectant l'ordre numérique."""
    unique = list(dict.fromkeys(values))
    if _is_numeric_series(unique):
        return sorted(unique, key=lambda v: (v is None, float(v)))
    return sorted(unique, key=lambda v: (v is None, str(v)))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _interpolate(colors: Sequence[str], position: float) -> str:
    if len(colors) == 1:
        return colors[0]
    position = min(max(position, 0.0), 1.0)
    scaled = position * (len(colors) - 1)
    low = int(np.floor(scaled))
    high = min(low + 1, len(colors) - 1)
    ratio = scaled - low
    c1, c2 = _hex_to_rgb(colors[low]), _hex_to_rgb(colors[high])
    blended = tuple(round(a + (b - a) * ratio) for a, b in zip(c1, c2))
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


class ColorEncoder:
    """Associe une couleur stable à chaque valeur d'une caractéristique.

    Une caractéristique numérique reçoit une échelle continue (l'ordre des valeurs
    devient lisible sur le graphique), une caractéristique textuelle une palette
    qualitative.
    """

    def __init__(
        self,
        values: Sequence[Any],
        palette: Sequence[str] | None = None,
        numeric: bool | None = None,
    ):
        self.order = sort_values_naturally(values)
        self.numeric = _is_numeric_series(self.order) if numeric is None else numeric
        if palette is not None:
            self.palette = list(palette)
        else:
            self.palette = list(SEQUENTIAL if self.numeric else QUALITATIVE)
        self._mapping = self._build_mapping()

    def _build_mapping(self) -> dict[Any, str]:
        count = len(self.order)
        if count == 0:
            return {}
        if self.numeric and count > 1:
            return {
                value: _interpolate(self.palette, index / (count - 1))
                for index, value in enumerate(self.order)
            }
        return {
            value: self.palette[index % len(self.palette)]
            for index, value in enumerate(self.order)
        }

    def color(self, value: Any) -> str:
        if value not in self._mapping:
            self.order = sort_values_naturally([*self.order, value])
            self._mapping = self._build_mapping()
        return self._mapping[value]

    def mapping(self) -> dict[Any, str]:
        return dict(self._mapping)


DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "star", "hexagon")


class StyleEncoder:
    """Associe un style de trait (ou un symbole) à chaque valeur, comme second canal."""

    def __init__(self, values: Sequence[Any], styles: Sequence[str] = DASHES):
        self.order = sort_values_naturally(values)
        self.styles = list(styles)

    def style(self, value: Any) -> str:
        if value not in self.order:
            self.order = sort_values_naturally([*self.order, value])
        return self.styles[self.order.index(value) % len(self.styles)]


def adaptive_line_style(n_configs: int) -> dict[str, float]:
    """Opacité et épaisseur de trait adaptées au nombre de courbes superposées.

    Dix courbes se lisent trace par trace ; trois cents se lisent comme une
    densité, où la courbe survolée reprend le premier plan.
    """
    if n_configs <= 12:
        return {"opacity": 0.95, "width": 2.2}
    if n_configs <= 40:
        return {"opacity": 0.85, "width": 1.8}
    if n_configs <= 120:
        return {"opacity": 0.55, "width": 1.4}
    if n_configs <= 300:
        return {"opacity": 0.35, "width": 1.1}
    return {"opacity": 0.22, "width": 1.0}


def adaptive_max_points(n_configs: int, budget: int = 160_000) -> int:
    """Nombre de points conservés par courbe pour rester fluide au total."""
    return int(min(4000, max(200, budget / max(n_configs, 1))))


def axis_title(name: str, units: Mapping[str, str] | None = None) -> str:
    """Titre d'axe enrichi de l'unité quand elle est connue."""
    unit = (units or {}).get(name)
    return f"{name} [{unit}]" if unit else name


def layout_title(title: str, subtitle: str | None = None) -> dict[str, Any]:
    """Titre principal avec sous-titre discret sur une seconde ligne."""
    if subtitle:
        text = f"{title}<br><span style='font-size:12px;color:#6b7280'>{subtitle}</span>"
    else:
        text = title
    return {"text": text}
