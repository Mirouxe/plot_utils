"""Graphiques de séries temporelles : superposition, grille, explorateur, faisceau."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import ColorEncoder, axis_title
from .common import (
    TraceStyler,
    curve_title,
    finalize,
    format_number,
    hover_lines,
    resolve_quantities,
    subtitle_for,
)

__all__ = ["curves", "grid", "explorer", "envelope", "small_multiples"]

SPIKE_STYLE = {
    "showspikes": True,
    "spikemode": "across",
    "spikethickness": 1,
    "spikedash": "dot",
    "spikecolor": "#9ca3af",
}


def _thin(frame: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None or len(frame) <= max_points:
        return frame
    step = int(np.ceil(len(frame) / max_points))
    return frame.iloc[::step]


def _hover_template(dataset, config: str, x_name: str, y_name: str) -> str:
    x_unit = dataset.units.get(x_name, "")
    y_unit = dataset.units.get(y_name, "")
    return (
        hover_lines(dataset, config)
        + f"<br><span style='color:#374151'>{x_name} = %{{x:.4g}} {x_unit}</span>"
        + f"<br><span style='color:#111827'><b>{y_name} = %{{y:.4g}} {y_unit}</b></span>"
        + "<extra></extra>"
    )


def curves(
    dataset,
    quantity: str,
    x: str | None = None,
    color: str | None = None,
    dash: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 560,
    width: int | None = None,
    line_width: float = 1.9,
    opacity: float = 0.9,
    markers: bool = False,
    log_y: bool = False,
    log_x: bool = False,
    palette: Sequence[str] | None = None,
    max_points: int | None = 4000,
    interactivity: Mapping[str, Any] | bool | None = None,
    spikes: bool = True,
) -> go.Figure:
    """Superpose une grandeur pour toutes les configurations du jeu.

    Args:
        quantity: grandeur en ordonnée.
        x: abscisse ; le temps par défaut, mais une autre grandeur est acceptée
            (diagramme de phase, courbe débit/pression, etc.).
        color: caractéristique portant la couleur (légende regroupée cliquable).
        dash: seconde caractéristique portant le style de trait.

    >>> ds.curves("temperature", color="maillage", dash="materiau")
    """
    x = x or dataset.time
    styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)
    figure = go.Figure()

    for config in dataset.configs:
        frame = _thin(dataset.frames[config], max_points)
        if quantity not in frame.columns or x not in frame.columns:
            continue
        style = styler.style(config)
        figure.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[quantity],
                mode="lines+markers" if markers else "lines",
                opacity=opacity,
                line={**style["line"], "width": line_width},
                marker={"size": 5, "color": style["line"]["color"]},
                name=style["name"],
                legendgroup=style["legendgroup"],
                showlegend=style["showlegend"],
                meta=style["meta"],
                hovertemplate=_hover_template(dataset, config, x, quantity),
            )
        )

    figure.update_xaxes(type="log" if log_x else "linear", **(SPIKE_STYLE if spikes else {}))
    figure.update_yaxes(type="log" if log_y else "linear")

    return finalize(
        figure,
        dataset,
        title=title or curve_title(dataset, quantity, x),
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        x_title=x,
        y_title=quantity,
        legend_title=styler.legend_title,
        height=height,
        width=width,
        interactivity=interactivity,
    )


def grid(
    dataset,
    quantities: Sequence[str] | None = None,
    x: str | None = None,
    color: str | None = None,
    dash: str | None = None,
    ncols: int = 1,
    title: str | None = None,
    subtitle: str | None = None,
    row_height: int = 240,
    width: int | None = None,
    line_width: float = 1.7,
    opacity: float = 0.9,
    shared_x: bool = True,
    unified_hover: bool = False,
    palette: Sequence[str] | None = None,
    max_points: int | None = 3000,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Trace plusieurs grandeurs en sous-graphiques alignés sur le même axe des temps.

    C'est la vue « planche de bord » : on lit d'un coup l'évolution de toutes les
    grandeurs d'un même lot de configurations, avec un zoom synchronisé.

    >>> ds.grid(["temperature", "pression", "debit"], color="maillage")
    """
    quantities = resolve_quantities(dataset, quantities)
    x = x or dataset.time
    nrows = int(np.ceil(len(quantities) / ncols))

    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=shared_x and ncols == 1,
        vertical_spacing=min(0.12, 0.5 / max(nrows, 1)),
        horizontal_spacing=0.08,
        subplot_titles=[axis_title(q, dataset.units) for q in quantities],
    )

    styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)

    for index, quantity in enumerate(quantities):
        row, col = index // ncols + 1, index % ncols + 1
        for config in dataset.configs:
            frame = _thin(dataset.frames[config], max_points)
            if quantity not in frame.columns:
                continue
            style = styler.style(config, force_hidden_legend=index > 0)
            figure.add_trace(
                go.Scatter(
                    x=frame[x],
                    y=frame[quantity],
                    mode="lines",
                    opacity=opacity,
                    line={**style["line"], "width": line_width},
                    name=style["name"],
                    legendgroup=style["legendgroup"],
                    showlegend=style["showlegend"],
                    meta=style["meta"],
                    hovertemplate=_hover_template(dataset, config, x, quantity),
                ),
                row=row,
                col=col,
            )
        figure.update_yaxes(title_text=None, row=row, col=col)
        if row == nrows or not shared_x:
            figure.update_xaxes(title_text=axis_title(x, dataset.units), row=row, col=col)

    figure.update_annotations(font={"size": 13, "color": "#374151"})
    figure.update_xaxes(**SPIKE_STYLE)

    return finalize(
        figure,
        dataset,
        title=title or "Vue d'ensemble des grandeurs",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        legend_title=styler.legend_title,
        height=max(320, row_height * nrows + 120),
        width=width,
        hovermode="x unified" if unified_hover else "closest",
        interactivity=interactivity,
    )


def explorer(
    dataset,
    quantities: Sequence[str] | None = None,
    x: str | None = None,
    color: str | None = None,
    color_options: Sequence[str] | None = None,
    dash: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 620,
    width: int | None = None,
    line_width: float = 1.9,
    opacity: float = 0.9,
    palette: Sequence[str] | None = None,
    max_points: int | None = 3000,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Graphique unique doté de menus pour changer de grandeur et de regroupement.

    Un seul fichier HTML permet alors de balayer toutes les grandeurs et de tester
    l'effet de chaque caractéristique, sans regénérer de figure ni lancer de serveur.

    >>> ds.explorer(color="maillage")           # menus « grandeur » et « couleur »
    """
    quantities = resolve_quantities(dataset, quantities)
    x = x or dataset.time
    color_options = list(color_options) if color_options is not None else dataset.characteristics
    if color is None and color_options:
        color = color_options[0]

    figure = go.Figure()
    configs = dataset.configs

    for index, quantity in enumerate(quantities):
        styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)
        for config in configs:
            frame = _thin(dataset.frames[config], max_points)
            style = styler.style(config)
            figure.add_trace(
                go.Scatter(
                    x=frame[x] if x in frame.columns else [],
                    y=frame[quantity] if quantity in frame.columns else [],
                    mode="lines",
                    visible=index == 0,
                    opacity=opacity,
                    line={**style["line"], "width": line_width},
                    name=style["name"],
                    legendgroup=style["legendgroup"],
                    showlegend=style["showlegend"],
                    meta=style["meta"],
                    hovertemplate=_hover_template(dataset, config, x, quantity),
                )
            )

    quantity_buttons = []
    for index, quantity in enumerate(quantities):
        visible = [
            block == index for block in range(len(quantities)) for _ in configs
        ]
        quantity_buttons.append(
            {
                "label": quantity,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "yaxis": {"title": {"text": axis_title(quantity, dataset.units)}},
                        "title": {
                            "text": curve_title(dataset, quantity, x)
                            + f"<br><span style='font-size:12px;color:#6b7280'>"
                            f"{subtitle_for(dataset)}</span>"
                        },
                    },
                ],
            }
        )

    color_buttons = []
    for option in color_options:
        styles = []
        for _ in quantities:
            styler = TraceStyler(dataset, color=option, dash=dash, palette=palette)
            styles.extend(styler.style(config) for config in configs)
        color_buttons.append(
            {
                "label": option,
                "method": "restyle",
                "args": [
                    {
                        "line.color": [s["line"]["color"] for s in styles],
                        "line.dash": [s["line"]["dash"] for s in styles],
                        "name": [s["name"] for s in styles],
                        "legendgroup": [s["legendgroup"] for s in styles],
                        "showlegend": [s["showlegend"] for s in styles],
                    }
                ],
            }
        )

    menus = [
        {
            "buttons": quantity_buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.14,
            "yanchor": "bottom",
            "bgcolor": "white",
            "bordercolor": "#d1d5db",
            "font": {"size": 12},
        }
    ]
    annotations = [
        {
            "text": "grandeur",
            "x": 0.0,
            "xref": "paper",
            "y": 1.155,
            "yref": "paper",
            "xanchor": "left",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 11, "color": "#6b7280"},
        }
    ]
    if len(color_buttons) > 1:
        menus.append(
            {
                "buttons": color_buttons,
                "direction": "down",
                "showactive": True,
                "active": color_options.index(color) if color in color_options else 0,
                "x": 0.22,
                "xanchor": "left",
                "y": 1.14,
                "yanchor": "bottom",
                "bgcolor": "white",
                "bordercolor": "#d1d5db",
                "font": {"size": 12},
            }
        )
        annotations.append(
            {
                "text": "couleur par",
                "x": 0.22,
                "xref": "paper",
                "y": 1.155,
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 11, "color": "#6b7280"},
            }
        )

    figure.update_layout(updatemenus=menus, annotations=annotations)
    figure.update_xaxes(**SPIKE_STYLE)

    return finalize(
        figure,
        dataset,
        title=title or curve_title(dataset, quantities[0], x),
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        x_title=x,
        y_title=quantities[0],
        legend_title=color or "configuration",
        height=height,
        width=width,
        interactivity=interactivity,
    )


def envelope(
    dataset,
    quantity: str,
    by: str | None = None,
    band: str = "minmax",
    x: str | None = None,
    points: int = 300,
    show_individual: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 560,
    width: int | None = None,
    palette: Sequence[str] | None = None,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Faisceau (min/max ou ±1σ) et courbe moyenne par groupe de configurations.

    Quand le nombre de configurations rend la superposition illisible, cette vue
    montre la tendance de chaque groupe et sa dispersion.

    Args:
        by: caractéristique définissant les groupes ; sans elle, un seul faisceau.
        band: ``"minmax"`` (enveloppe complète) ou ``"std"`` (±1 écart-type).
    """
    if band not in {"minmax", "std"}:
        raise ValueError("band doit valoir 'minmax' ou 'std'.")

    x = x or dataset.time
    resampled = dataset.resample(points) if x == dataset.time else dataset
    axis = resampled.frames[resampled.configs[0]][x].to_numpy(dtype=float)

    if by is None:
        groups: dict[Any, list[str]] = {"toutes configurations": list(resampled.configs)}
    else:
        groups = {
            value: [c for c in resampled.configs if resampled.meta.at[c, by] == value]
            for value in resampled.values(by)
        }

    encoder = ColorEncoder(list(groups), palette=palette)
    figure = go.Figure()

    for value, configs in groups.items():
        if not configs:
            continue
        stacked = np.vstack(
            [
                pd.to_numeric(resampled.frames[c][quantity], errors="coerce").to_numpy(float)
                for c in configs
            ]
        )
        mean = np.nanmean(stacked, axis=0)
        if band == "minmax":
            low, high = np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)
            band_label = "min–max"
        else:
            spread = np.nanstd(stacked, axis=0, ddof=1) if len(configs) > 1 else np.zeros_like(mean)
            low, high = mean - spread, mean + spread
            band_label = "moyenne ± 1σ"

        color = encoder.color(value)
        fill = _to_rgba(color, 0.18)
        label = format_number(value) if by else str(value)
        described = f"{by} = {label}" if by else label

        figure.add_trace(
            go.Scatter(
                x=np.concatenate([axis, axis[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself",
                fillcolor=fill,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name=f"{label} ({band_label})",
                legendgroup=label,
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=axis,
                y=mean,
                mode="lines",
                line={"color": color, "width": 2.6},
                name=f"{label} · {len(configs)} config.",
                legendgroup=label,
                meta={"label": label},
                hovertemplate=(
                    f"<b>{described}</b><br>{len(configs)} configurations<br>"
                    f"{x} = %{{x:.4g}}<br>moyenne {quantity} = %{{y:.4g}}<extra></extra>"
                ),
            )
        )

        if show_individual:
            for config in configs:
                frame = resampled.frames[config]
                figure.add_trace(
                    go.Scatter(
                        x=frame[x],
                        y=frame[quantity],
                        mode="lines",
                        line={"color": color, "width": 0.9},
                        opacity=0.35,
                        showlegend=False,
                        legendgroup=label,
                        meta={"label": resampled.label(config)},
                        hovertemplate=_hover_template(resampled, config, x, quantity),
                    )
                )

    figure.update_xaxes(**SPIKE_STYLE)

    return finalize(
        figure,
        dataset,
        title=title or f"Dispersion de {quantity}"
        + (f" par {by}" if by else ""),
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        x_title=x,
        y_title=quantity,
        legend_title=by or "groupe",
        height=height,
        width=width,
        interactivity=interactivity if interactivity is not None else {"highlight": False},
    )


def small_multiples(
    dataset,
    quantity: str,
    facet: str,
    color: str | None = None,
    x: str | None = None,
    ncols: int = 3,
    title: str | None = None,
    subtitle: str | None = None,
    row_height: int = 250,
    width: int | None = None,
    shared_y: bool = True,
    line_width: float = 1.7,
    opacity: float = 0.95,
    palette: Sequence[str] | None = None,
    max_points: int | None = 2000,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Une facette par valeur d'une caractéristique, pour isoler son effet.

    >>> ds.small_multiples("temperature", facet="materiau", color="puissance")
    """
    x = x or dataset.time
    values = dataset.values(facet)
    ncols = min(ncols, max(len(values), 1))
    nrows = int(np.ceil(len(values) / ncols))

    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=shared_y,
        shared_xaxes=True,
        vertical_spacing=min(0.14, 0.6 / max(nrows, 1)),
        horizontal_spacing=0.05,
        subplot_titles=[f"{facet} = {value}" for value in values],
    )

    styler = TraceStyler(dataset, color=color, palette=palette)

    for index, value in enumerate(values):
        row, col = index // ncols + 1, index % ncols + 1
        configs = [c for c in dataset.configs if dataset.meta.at[c, facet] == value]
        for config in configs:
            frame = _thin(dataset.frames[config], max_points)
            style = styler.style(config, force_hidden_legend=index > 0)
            figure.add_trace(
                go.Scatter(
                    x=frame[x],
                    y=frame[quantity],
                    mode="lines",
                    opacity=opacity,
                    line={**style["line"], "width": line_width},
                    name=style["name"],
                    legendgroup=style["legendgroup"],
                    showlegend=style["showlegend"],
                    meta=style["meta"],
                    hovertemplate=_hover_template(dataset, config, x, quantity),
                ),
                row=row,
                col=col,
            )
        if row == nrows:
            figure.update_xaxes(title_text=axis_title(x, dataset.units), row=row, col=col)
        if col == 1:
            figure.update_yaxes(
                title_text=axis_title(quantity, dataset.units), row=row, col=col
            )

    figure.update_annotations(font={"size": 13, "color": "#374151"})

    return finalize(
        figure,
        dataset,
        title=title or f"{quantity} : effet de {facet}",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        legend_title=styler.legend_title,
        height=max(320, row_height * nrows + 130),
        width=width,
        interactivity=interactivity,
    )


def _to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
