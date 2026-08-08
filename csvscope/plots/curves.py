"""Graphiques de séries temporelles : superposition, grille, explorateur, faisceau."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    ColorEncoder,
    adaptive_line_style,
    adaptive_max_points,
    axis_title,
)
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

#: Au-delà de ce volume total de points, le rendu SVG devient poussif : WebGL.
WEBGL_THRESHOLD = 60_000


def _thin(frame: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None or len(frame) <= max_points:
        return frame
    step = int(np.ceil(len(frame) / max_points))
    return frame.iloc[::step]


def _scatter_class(render: str, total_points: int):
    """Choisit le moteur de rendu des traces (SVG précis ou WebGL rapide)."""
    if render == "webgl":
        return go.Scattergl
    if render == "svg":
        return go.Scatter
    if render != "auto":
        raise ValueError("render doit valoir 'auto', 'svg' ou 'webgl'.")
    return go.Scattergl if total_points > WEBGL_THRESHOLD else go.Scatter


def _resolve_line_style(
    dataset, opacity: float | None, line_width: float | None, max_points: int | None
) -> tuple[float, float, int]:
    auto = adaptive_line_style(len(dataset))
    return (
        auto["opacity"] if opacity is None else opacity,
        auto["width"] if line_width is None else line_width,
        adaptive_max_points(len(dataset)) if max_points is None else max_points,
    )


def _total_points(dataset, quantity_count: int, max_points: int) -> int:
    return quantity_count * sum(
        min(len(frame), max_points) for frame in dataset.frames.values()
    )


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
    line_width: float | None = None,
    opacity: float | None = None,
    markers: bool = False,
    log_y: bool = False,
    log_x: bool = False,
    palette: Sequence[str] | None = None,
    max_points: int | None = None,
    render: str = "auto",
    interactivity: Mapping[str, Any] | bool | None = None,
    spikes: bool = True,
) -> go.Figure:
    """Superpose une grandeur pour toutes les configurations du jeu.

    Les réglages s'adaptent au volume : opacité et épaisseur diminuent avec le
    nombre de configurations, le sous-échantillonnage respecte un budget global
    de points, et le rendu bascule en WebGL au-delà de quelques dizaines de
    milliers de points. La légende par configuration disparaît quand elle
    deviendrait plus longue que le graphique : le survol prend le relais.

    Args:
        quantity: grandeur en ordonnée.
        x: abscisse ; le temps par défaut, mais une autre grandeur est acceptée
            (diagramme de phase, courbe débit/pression, etc.).
        color: caractéristique portant la couleur (légende regroupée cliquable).
        dash: seconde caractéristique portant le style de trait.
        render: ``"auto"`` (défaut), ``"svg"`` ou ``"webgl"``.

    >>> ds.curves("temperature", color="maillage", dash="materiau")
    """
    x = x or dataset.time
    styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)
    opacity, line_width, max_points = _resolve_line_style(
        dataset, opacity, line_width, max_points
    )
    trace_class = _scatter_class(render, _total_points(dataset, 1, max_points))
    hide_config_legend = color is None and dash is None and len(dataset) > 20
    figure = go.Figure()

    for config in dataset.configs:
        frame = _thin(dataset.frames[config], max_points)
        if quantity not in frame.columns or x not in frame.columns:
            continue
        style = styler.style(config, force_hidden_legend=hide_config_legend)
        figure.add_trace(
            trace_class(
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

    hint = "survole une courbe pour l'identifier" if hide_config_legend else None
    return finalize(
        figure,
        dataset,
        title=title or curve_title(dataset, quantity, x),
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset, hint),
        x_title=x,
        y_title=quantity,
        legend_title=styler.legend_title,
        height=height,
        width=width,
        show_legend=not hide_config_legend,
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
    line_width: float | None = None,
    opacity: float | None = None,
    shared_x: bool = True,
    unified_hover: bool = False,
    palette: Sequence[str] | None = None,
    max_points: int | None = None,
    render: str = "auto",
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Trace plusieurs grandeurs en sous-graphiques alignés sur le même axe des temps.

    C'est la vue « planche de bord » : on lit d'un coup l'évolution de toutes les
    grandeurs d'un même lot de configurations, avec un zoom synchronisé. Survoler
    une courbe met en évidence la même configuration dans tous les sous-graphiques.

    >>> ds.grid(["temperature", "pression", "debit"], color="maillage")
    """
    quantities = resolve_quantities(dataset, quantities)
    x = x or dataset.time
    nrows = int(np.ceil(len(quantities) / ncols))
    opacity, line_width, max_points = _resolve_line_style(
        dataset, opacity, line_width, max_points
    )
    max_points = max(150, max_points // max(len(quantities), 1))
    trace_class = _scatter_class(
        render, _total_points(dataset, len(quantities), max_points)
    )

    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=shared_x and ncols == 1,
        vertical_spacing=min(0.12, 0.5 / max(nrows, 1)),
        horizontal_spacing=0.08,
        subplot_titles=[axis_title(q, dataset.units) for q in quantities],
    )

    styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)
    hide_config_legend = color is None and dash is None and len(dataset) > 20

    for index, quantity in enumerate(quantities):
        row, col = index // ncols + 1, index % ncols + 1
        for config in dataset.configs:
            frame = _thin(dataset.frames[config], max_points)
            if quantity not in frame.columns:
                continue
            style = styler.style(
                config, force_hidden_legend=index > 0 or hide_config_legend
            )
            figure.add_trace(
                trace_class(
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

    hint = "survole une courbe pour l'identifier" if hide_config_legend else None
    return finalize(
        figure,
        dataset,
        title=title or "Vue d'ensemble des grandeurs",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset, hint),
        legend_title=styler.legend_title,
        height=max(320, row_height * nrows + 120),
        width=width,
        hovermode="x unified" if unified_hover else "closest",
        show_legend=not hide_config_legend,
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
    line_width: float | None = None,
    opacity: float | None = None,
    palette: Sequence[str] | None = None,
    max_points: int | None = None,
    render: str = "auto",
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

    opacity, line_width, max_points = _resolve_line_style(
        dataset, opacity, line_width, max_points
    )
    # Toutes les grandeurs cohabitent dans la figure : budget par grandeur.
    max_points = max(150, max_points // max(len(quantities), 1))
    trace_class = _scatter_class(
        render, _total_points(dataset, len(quantities), max_points)
    )
    figure = go.Figure()
    configs = dataset.configs

    for index, quantity in enumerate(quantities):
        styler = TraceStyler(dataset, color=color, dash=dash, palette=palette)
        for config in configs:
            frame = _thin(dataset.frames[config], max_points)
            style = styler.style(config)
            figure.add_trace(
                trace_class(
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
    band: str = "quantiles",
    quantiles: tuple[float, float] = (0.1, 0.9),
    center: str = "median",
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
    """Courbe centrale et bande de dispersion par groupe de configurations.

    C'est la vue de synthèse quand la superposition brute devient illisible : la
    tendance de chaque groupe, et l'étendue du faisceau autour d'elle. La bande
    par quantiles (défaut) reste lisible même quand quelques configurations
    extrêmes écraseraient une enveloppe min–max.

    Args:
        by: caractéristique définissant les groupes ; sans elle, un seul faisceau.
        band: ``"quantiles"`` (défaut), ``"minmax"`` ou ``"std"`` (±1 écart-type).
        quantiles: bornes de la bande quand ``band="quantiles"``.
        center: ``"median"`` (défaut) ou ``"mean"``.
    """
    if band not in {"quantiles", "minmax", "std"}:
        raise ValueError("band doit valoir 'quantiles', 'minmax' ou 'std'.")
    if center not in {"median", "mean"}:
        raise ValueError("center doit valoir 'median' ou 'mean'.")

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
    center_label = "médiane" if center == "median" else "moyenne"
    if band == "quantiles":
        band_label = f"P{quantiles[0] * 100:g}–P{quantiles[1] * 100:g}"
    elif band == "minmax":
        band_label = "min–max"
    else:
        band_label = "±1σ"
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
        middle = np.nanmedian(stacked, axis=0) if center == "median" else np.nanmean(stacked, axis=0)
        if band == "quantiles":
            low = np.nanquantile(stacked, quantiles[0], axis=0)
            high = np.nanquantile(stacked, quantiles[1], axis=0)
        elif band == "minmax":
            low, high = np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)
        else:
            mean = np.nanmean(stacked, axis=0)
            spread = np.nanstd(stacked, axis=0, ddof=1) if len(configs) > 1 else np.zeros_like(mean)
            low, high = mean - spread, mean + spread

        color = encoder.color(value)
        label = format_number(value) if by else str(value)
        described = f"{by} = {label}" if by else label

        figure.add_trace(
            go.Scatter(
                x=np.concatenate([axis, axis[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself",
                fillcolor=_to_rgba(color, 0.16),
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
                y=middle,
                mode="lines",
                line={"color": color, "width": 2.6},
                name=f"{label} · {len(configs)} config.",
                legendgroup=label,
                meta={"label": label},
                hovertemplate=(
                    f"<b>{described}</b><br>{len(configs)} configurations · bande {band_label}<br>"
                    f"{x} = %{{x:.4g}}<br>{center_label} {quantity} = %{{y:.4g}}<extra></extra>"
                ),
            )
        )

        if show_individual:
            style = adaptive_line_style(len(resampled))
            for config in configs:
                frame = resampled.frames[config]
                figure.add_trace(
                    go.Scatter(
                        x=frame[x],
                        y=frame[quantity],
                        mode="lines",
                        line={"color": color, "width": 0.8},
                        opacity=min(0.35, style["opacity"]),
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
        title=title or f"Dispersion de {quantity}" + (f" par {by}" if by else ""),
        subtitle=subtitle if subtitle is not None
        else subtitle_for(dataset, f"{center_label} et bande {band_label}"),
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
    line_width: float | None = None,
    opacity: float | None = None,
    palette: Sequence[str] | None = None,
    max_points: int | None = None,
    render: str = "auto",
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Une facette par valeur d'une caractéristique, pour isoler son effet.

    >>> ds.small_multiples("temperature", facet="materiau", color="puissance")
    """
    x = x or dataset.time
    values = dataset.values(facet)
    ncols = min(ncols, max(len(values), 1))
    nrows = int(np.ceil(len(values) / ncols))
    opacity, line_width, max_points = _resolve_line_style(
        dataset, opacity, line_width, max_points
    )
    trace_class = _scatter_class(render, _total_points(dataset, 1, max_points))

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
                trace_class(
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
