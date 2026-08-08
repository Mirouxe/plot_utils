"""Comparaisons scalaires : une valeur par configuration, croisée aux caractéristiques."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..metrics import metric_label, normalize_metrics
from ..theme import ColorEncoder, SEQUENTIAL, sort_values_naturally
from .common import (
    characteristic_axis,
    finalize,
    format_number,
    metric_axis_title,
    resolve_metric_spec,
    subtitle_for,
)

__all__ = ["compare", "bars", "heatmap", "distribution", "scatter", "pareto", "metrics_table"]


def _default_characteristic(dataset, exclude: Sequence[str] = ()) -> str:
    for characteristic in dataset.characteristics:
        if characteristic in exclude:
            continue
        if len(dataset.values(characteristic)) > 1:
            return characteristic
    remaining = [c for c in dataset.characteristics if c not in exclude]
    if not remaining:
        raise ValueError("Aucune caractéristique disponible pour cet axe.")
    return remaining[0]


def _hover_customdata(table: pd.DataFrame, dataset) -> tuple[np.ndarray, str]:
    keys = ["label", *dataset.characteristics]
    keys = [k for k in keys if k in table.columns]
    data = table[keys].astype(object).to_numpy()
    lines = [f"<b>%{{customdata[0]}}</b>"]
    for index, key in enumerate(keys[1:], start=1):
        lines.append(f"{key} : %{{customdata[{index}]}}")
    return data, "<br>".join(lines)


def compare(
    dataset,
    quantity: str,
    metric: Any = "max",
    x: str | None = None,
    color: str | None = None,
    aggregate: str | None = "mean",
    show_points: bool = True,
    show_spread: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 540,
    width: int | None = None,
    log_y: bool = False,
    palette: Sequence[str] | None = None,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Évolution d'une métrique en fonction d'une caractéristique.

    C'est la vue de sensibilité : « comment le maximum de température varie-t-il
    avec la puissance, et cela dépend-il du matériau ? ».

    Args:
        metric: métrique scalaire (``"max"``, ``"mean"``, ``"integral"``, …).
        x: caractéristique en abscisse.
        color: seconde caractéristique, tracée en séries distinctes.
        aggregate: ``"mean"``, ``"median"`` ou ``None`` (pas de courbe agrégée).
    """
    x = x or _default_characteristic(dataset)
    table = dataset.table(quantity, metric).reset_index()
    encoder_values = dataset.values(color) if color else ["toutes"]
    encoder = ColorEncoder(encoder_values, palette=palette)
    figure = go.Figure()

    for value in encoder_values:
        subset = table if color is None else table[table[color] == value]
        if subset.empty:
            continue
        line_color = encoder.color(value)
        label = format_number(value) if color else "ensemble"
        described = f"{color} = {label}" if color else label

        if aggregate:
            grouped = subset.groupby(x, dropna=False)[quantity]
            aggregated = getattr(grouped, aggregate)()
            spread = grouped.std(ddof=1) if show_spread else None
            order = sort_values_naturally(aggregated.index.tolist())
            aggregated = aggregated.reindex(order)
            error = None
            if spread is not None:
                spread = spread.reindex(order).fillna(0.0)
                error = {"type": "data", "array": spread.to_numpy(), "visible": True,
                         "thickness": 1.2, "width": 6, "color": line_color}
            figure.add_trace(
                go.Scatter(
                    x=[str(v) for v in order] if not _numeric(order) else order,
                    y=aggregated.to_numpy(),
                    mode="lines+markers",
                    line={"color": line_color, "width": 2.4},
                    marker={"size": 9, "color": line_color, "line": {"color": "white", "width": 1}},
                    error_y=error,
                    name=label,
                    legendgroup=label,
                    meta={"label": label},
                    hovertemplate=(
                        f"<b>{described}</b><br>{x} = %{{x}}<br>"
                        f"{aggregate} = %{{y:.4g}}<extra></extra>"
                    ),
                )
            )

        if show_points:
            customdata, hover_head = _hover_customdata(subset, dataset)
            figure.add_trace(
                go.Scatter(
                    x=subset[x] if _numeric(subset[x].tolist()) else subset[x].astype(str),
                    y=subset[quantity],
                    mode="markers",
                    marker={
                        "size": 7,
                        "color": line_color,
                        "opacity": 0.55,
                        "line": {"color": line_color, "width": 1},
                    },
                    name=f"{label} (configurations)",
                    legendgroup=label,
                    showlegend=not aggregate,
                    customdata=customdata,
                    meta={"label": label},
                    hovertemplate=(
                        hover_head
                        + f"<br><b>{metric_label(metric)} {quantity} = %{{y:.4g}}</b><extra></extra>"
                    ),
                )
            )

    characteristic_axis(figure, dataset, x, axis="x")
    figure.update_yaxes(type="log" if log_y else "linear")

    return finalize(
        figure,
        dataset,
        title=title or f"{metric_label(metric).capitalize()} de {quantity} selon {x}",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        x_title=x,
        y_title=None,
        legend_title=color or "série",
        height=height,
        width=width,
        interactivity=interactivity if interactivity is not None else {"search": False},
    ).update_yaxes(title_text=metric_axis_title(quantity, metric, dataset.units))


def bars(
    dataset,
    quantity: str,
    metric: Any = "max",
    color: str | None = None,
    top: int | None = 20,
    ascending: bool = False,
    orientation: str = "h",
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = None,
    width: int | None = None,
    palette: Sequence[str] | None = None,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Classement des configurations sur une métrique.

    Seules les ``top`` premières sont affichées (20 par défaut) : un classement
    de plusieurs centaines de barres n'apprend rien. ``top=None`` les garde toutes.

    >>> ds.bars("temperature", metric="max", color="materiau", top=15)
    """
    table = dataset.table(quantity, metric).reset_index()
    table = table.sort_values(quantity, ascending=ascending)
    truncated = top is not None and len(table) > top
    if top:
        table = table.head(top)
    table = table.iloc[::-1] if orientation == "h" else table

    encoder = ColorEncoder(dataset.values(color) if color else table["label"].tolist(),
                           palette=palette)
    colors = [
        encoder.color(row[color] if color else row["label"])
        for _, row in table.iterrows()
    ]
    customdata, hover_head = _hover_customdata(table, dataset)
    labels = table["label"].astype(str)
    values = table[quantity]

    figure = go.Figure(
        go.Bar(
            x=values if orientation == "h" else labels,
            y=labels if orientation == "h" else values,
            orientation=orientation,
            marker={"color": colors, "line": {"color": "white", "width": 0.5}},
            customdata=customdata,
            text=[format_number(v) for v in values],
            textposition="outside" if orientation == "h" else "auto",
            textfont={"size": 11},
            cliponaxis=False,
            hovertemplate=(
                hover_head
                + f"<br><b>{metric_label(metric)} {quantity} = "
                + ("%{x:.4g}" if orientation == "h" else "%{y:.4g}")
                + "</b><extra></extra>"
            ),
        )
    )

    axis_label = metric_axis_title(quantity, metric, dataset.units)
    if orientation == "h":
        figure.update_xaxes(title_text=axis_label)
        figure.update_yaxes(title_text=None, showgrid=False, automargin=True)
    else:
        figure.update_yaxes(title_text=axis_label)
        figure.update_xaxes(title_text=None, tickangle=-35, showgrid=False)

    computed_height = height or max(340, 26 * len(table) + 160)

    hint = f"{len(table)} premières sur {len(dataset)}" if truncated else None
    return finalize(
        figure,
        dataset,
        title=title or f"Classement des configurations · {metric_label(metric)} de {quantity}",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset, hint),
        height=computed_height,
        width=width,
        show_legend=False,
        interactivity=interactivity if interactivity is not None else False,
    )


def heatmap(
    dataset,
    quantity: str,
    x: str | None = None,
    y: str | None = None,
    metric: Any = "max",
    aggregate: str = "mean",
    text: bool = True,
    colorscale: Sequence[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 520,
    width: int | None = None,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Carte d'une métrique sur le croisement de deux caractéristiques.

    Les cases vides signalent les combinaisons non encore calculées, ce qui aide
    aussi à piloter les campagnes de simulation.

    >>> ds.heatmap("temperature", x="puissance", y="materiau", metric="max")
    """
    x = x or _default_characteristic(dataset)
    y = y or _default_characteristic(dataset, exclude=[x])
    table = dataset.table(quantity, metric).reset_index()

    pivot = table.pivot_table(index=y, columns=x, values=quantity, aggfunc=aggregate)
    pivot = pivot.reindex(index=sort_values_naturally(pivot.index.tolist()),
                          columns=sort_values_naturally(pivot.columns.tolist()))
    counts = table.pivot_table(index=y, columns=x, values=quantity, aggfunc="count")
    counts = counts.reindex(index=pivot.index, columns=pivot.columns)

    scale = list(colorscale or SEQUENTIAL)
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
            customdata=counts.to_numpy(),
            hovertemplate=(
                f"{x} = %{{x}}<br>{y} = %{{y}}<br>"
                f"<b>{aggregate} {metric_label(metric)} {quantity} = %{{z:.4g}}</b>"
                "<br>%{customdata} configuration(s)<extra></extra>"
            ),
            colorbar={
                "title": {"text": metric_axis_title(quantity, metric, dataset.units),
                          "side": "right"},
                "thickness": 14,
                "outlinewidth": 0,
            },
            texttemplate="%{z:.3g}" if text else None,
            textfont={"size": 11},
            hoverongaps=False,
        )
    )
    figure.update_xaxes(title_text=x, type="category")
    figure.update_yaxes(title_text=y, type="category", showgrid=False)

    return finalize(
        figure,
        dataset,
        title=title or f"{metric_label(metric).capitalize()} de {quantity} : {y} × {x}",
        subtitle=subtitle if subtitle is not None
        else subtitle_for(dataset, f"agrégation : {aggregate}"),
        height=height,
        width=width,
        show_legend=False,
        interactivity=interactivity if interactivity is not None else False,
    )


def distribution(
    dataset,
    quantity: str,
    by: str | None = None,
    metric: Any = "max",
    kind: str = "box",
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 520,
    width: int | None = None,
    palette: Sequence[str] | None = None,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Distribution d'une métrique, globalement ou par groupe.

    Args:
        kind: ``"box"``, ``"violin"``, ``"strip"``, ``"hist"`` ou ``"ecdf"``.
    """
    if kind not in {"box", "violin", "strip", "hist", "ecdf"}:
        raise ValueError("kind doit valoir box, violin, strip, hist ou ecdf.")

    table = dataset.table(quantity, metric).reset_index()
    groups = dataset.values(by) if by else ["ensemble"]
    encoder = ColorEncoder(groups, palette=palette)
    figure = go.Figure()

    for value in groups:
        subset = table if by is None else table[table[by] == value]
        if subset.empty:
            continue
        color = encoder.color(value)
        label = format_number(value) if by else "ensemble"
        values = subset[quantity].to_numpy(dtype=float)
        customdata, hover_head = _hover_customdata(subset, dataset)
        hover = hover_head + f"<br><b>{metric_label(metric)} {quantity} = %{{x:.4g}}</b><extra></extra>"

        if kind in {"box", "violin"}:
            # Au-delà de quelques dizaines de points par groupe, le nuage
            # complet devient un trait plein : seuls les extrêmes restent utiles.
            points_mode = "all" if len(subset) <= 40 else "outliers"
            if kind == "box":
                trace_type: Any = go.Box
                extra: dict[str, Any] = {"boxmean": True, "boxpoints": points_mode}
            else:
                trace_type = go.Violin
                extra = {
                    "meanline": {"visible": True},
                    "box": {"visible": True},
                    "points": points_mode,
                }
            figure.add_trace(
                trace_type(
                    x=values,
                    name=label,
                    orientation="h",
                    marker={"color": color, "size": 6},
                    line={"color": color},
                    fillcolor=_rgba(color, 0.25),
                    jitter=0.35,
                    pointpos=0,
                    customdata=customdata,
                    hovertemplate=hover,
                    **extra,
                )
            )
        elif kind == "strip":
            figure.add_trace(
                go.Scatter(
                    x=values,
                    y=[label] * len(values),
                    mode="markers",
                    marker={"color": color, "size": 9, "opacity": 0.65,
                            "line": {"color": "white", "width": 1}},
                    name=label,
                    customdata=customdata,
                    hovertemplate=hover,
                )
            )
        elif kind == "hist":
            figure.add_trace(
                go.Histogram(
                    x=values,
                    name=label,
                    marker={"color": _rgba(color, 0.65), "line": {"color": color, "width": 1}},
                    hovertemplate=f"<b>{label}</b><br>{quantity} ∈ %{{x}}<br>%{{y}} configurations<extra></extra>",
                )
            )
        else:
            ordered = np.sort(values)
            figure.add_trace(
                go.Scatter(
                    x=ordered,
                    y=np.arange(1, len(ordered) + 1) / len(ordered),
                    mode="lines+markers",
                    line={"color": color, "width": 2, "shape": "hv"},
                    marker={"size": 5, "color": color},
                    name=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>{quantity} ≤ %{{x:.4g}}<br>"
                        "fraction = %{y:.0%}<extra></extra>"
                    ),
                )
            )

    if kind == "hist":
        figure.update_layout(barmode="overlay")
        figure.update_traces(opacity=0.75 if by else 1.0)

    axis_label = metric_axis_title(quantity, metric, dataset.units)
    figure.update_xaxes(title_text=axis_label)
    if kind == "ecdf":
        figure.update_yaxes(title_text="fraction cumulée", tickformat=".0%")
    elif kind == "hist":
        figure.update_yaxes(title_text="nombre de configurations")
    else:
        figure.update_yaxes(title_text=by or "", showgrid=False)

    return finalize(
        figure,
        dataset,
        title=title or f"Distribution du {metric_label(metric)} de {quantity}"
        + (f" par {by}" if by else ""),
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        legend_title=by or "groupe",
        height=height,
        width=width,
        interactivity=interactivity if interactivity is not None else False,
    )


def scatter(
    dataset,
    x: str | tuple[str, Any],
    y: str | tuple[str, Any],
    metric: Any = "max",
    color: str | None = None,
    size: str | tuple[str, Any] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 560,
    width: int | None = None,
    palette: Sequence[str] | None = None,
    trend: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Compromis entre deux grandeurs, une configuration par point.

    ``x`` et ``y`` acceptent ``"grandeur"`` ou ``("grandeur", "métrique")``, ce qui
    permet par exemple de croiser le maximum de contrainte et la moyenne de débit.

    >>> ds.scatter(("temperature", "max"), ("rendement", "mean"), color="materiau")
    """
    table, x_quantity, x_metric, y_column, y_metric, y_quantity = _merge_metric_axes(
        dataset, x, y, metric
    )

    sizes = None
    if size is not None:
        size_quantity, size_metric = resolve_metric_spec(size, metric)
        size_values = dataset.table([size_quantity], size_metric)[size_quantity]
        raw = table["config"].map(size_values).to_numpy(dtype=float)
        span = np.nanmax(raw) - np.nanmin(raw)
        sizes = 9 + 22 * ((raw - np.nanmin(raw)) / span if span else np.zeros_like(raw))

    groups = dataset.values(color) if color else ["ensemble"]
    encoder = ColorEncoder(groups, palette=palette)
    figure = go.Figure()

    for value in groups:
        subset = table if color is None else table[table[color] == value]
        if subset.empty:
            continue
        point_color = encoder.color(value)
        label = format_number(value) if color else "configurations"
        customdata, hover_head = _hover_customdata(subset, dataset)
        marker: dict[str, Any] = {
            "color": point_color,
            "size": 11 if sizes is None else sizes[subset.index.to_numpy()],
            "opacity": 0.8,
            "line": {"color": "white", "width": 1.2},
        }
        figure.add_trace(
            go.Scatter(
                x=subset[x_quantity],
                y=subset[y_column],
                mode="markers",
                marker=marker,
                name=label,
                meta={"label": label},
                customdata=customdata,
                hovertemplate=(
                    hover_head
                    + f"<br>{metric_label(x_metric)} {x_quantity} = %{{x:.4g}}"
                    + f"<br><b>{metric_label(y_metric)} {y_quantity} = %{{y:.4g}}</b>"
                    + "<extra></extra>"
                ),
            )
        )

    if trend and len(table) > 2:
        coefficients = np.polyfit(table[x_quantity], table[y_column], 1)
        span = np.linspace(table[x_quantity].min(), table[x_quantity].max(), 50)
        figure.add_trace(
            go.Scatter(
                x=span,
                y=np.polyval(coefficients, span),
                mode="lines",
                line={"color": "#6b7280", "width": 1.5, "dash": "dash"},
                name="tendance linéaire",
                hoverinfo="skip",
            )
        )

    figure.update_xaxes(
        title_text=metric_axis_title(x_quantity, x_metric, dataset.units),
        type="log" if log_x else "linear",
    )
    figure.update_yaxes(
        title_text=metric_axis_title(y_quantity, y_metric, dataset.units),
        type="log" if log_y else "linear",
    )

    return finalize(
        figure,
        dataset,
        title=title or f"{y_quantity} en fonction de {x_quantity} (par configuration)",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset),
        legend_title=color or "série",
        height=height,
        width=width,
        interactivity=interactivity if interactivity is not None else {"search": False},
    )


def _merge_metric_axes(
    dataset, x: str | tuple[str, Any], y: str | tuple[str, Any], metric: Any
) -> tuple[pd.DataFrame, str, Any, str, Any, str]:
    """Table à deux colonnes de métriques, avec gestion du cas x == y."""
    x_quantity, x_metric = resolve_metric_spec(x, metric)
    y_quantity, y_metric = resolve_metric_spec(y, metric)

    table = dataset.table([x_quantity], x_metric).reset_index()
    y_table = dataset.table([y_quantity], y_metric).reset_index()[["config", y_quantity]]
    if x_quantity == y_quantity:
        y_table = y_table.rename(columns={y_quantity: f"{y_quantity}__y"})
        y_column = f"{y_quantity}__y"
    else:
        y_column = y_quantity
    table = table.merge(y_table, on="config")
    return table, x_quantity, x_metric, y_column, y_metric, y_quantity


def _pareto_front(
    x_values: np.ndarray, y_values: np.ndarray, sense: tuple[str, str]
) -> np.ndarray:
    """Indices des points non dominés, triés le long du front."""
    for direction in sense:
        if direction not in {"min", "max"}:
            raise ValueError("sense doit contenir 'min' ou 'max' pour chaque axe.")
    tx = x_values if sense[0] == "min" else -x_values
    ty = y_values if sense[1] == "min" else -y_values

    order = np.lexsort((ty, tx))
    front: list[int] = []
    best = np.inf
    for index in order:
        if ty[index] < best:
            front.append(int(index))
            best = ty[index]
    return np.asarray(front, dtype=int)


def pareto(
    dataset,
    x: str | tuple[str, Any],
    y: str | tuple[str, Any],
    sense: tuple[str, str] = ("min", "max"),
    metric: Any = "max",
    color: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = 560,
    width: int | None = None,
    palette: Sequence[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    interactivity: Mapping[str, Any] | bool | None = None,
) -> go.Figure:
    """Front de Pareto entre deux critères : les compromis qu'aucune autre
    configuration ne bat sur les deux axes à la fois.

    C'est la vue à ouvrir quand aucun critère unique ne s'impose : le front
    réduit des centaines de configurations aux seules candidates rationnelles.

    Args:
        x, y: ``"grandeur"`` ou ``("grandeur", "métrique")``.
        sense: sens d'optimisation de chaque axe, ``"min"`` ou ``"max"``.

    >>> ds.pareto(("temperature", "max"), ("rendement", "mean"), sense=("min", "max"))
    """
    table, x_quantity, x_metric, y_column, y_metric, y_quantity = _merge_metric_axes(
        dataset, x, y, metric
    )
    x_values = table[x_quantity].to_numpy(dtype=float)
    y_values = table[y_column].to_numpy(dtype=float)
    front = _pareto_front(x_values, y_values, sense)
    on_front = np.zeros(len(table), dtype=bool)
    on_front[front] = True

    groups = dataset.values(color) if color else ["ensemble"]
    encoder = ColorEncoder(groups, palette=palette)
    figure = go.Figure()

    for value in groups:
        mask = np.ones(len(table), dtype=bool) if color is None else (table[color] == value).to_numpy()
        subset = table[mask & ~on_front]
        if subset.empty:
            continue
        point_color = encoder.color(value)
        label = format_number(value) if color else "configurations dominées"
        customdata, hover_head = _hover_customdata(subset, dataset)
        figure.add_trace(
            go.Scatter(
                x=subset[x_quantity],
                y=subset[y_column],
                mode="markers",
                marker={"size": 8, "color": point_color, "opacity": 0.45,
                        "line": {"color": "white", "width": 1}},
                name=label,
                legendgroup=str(value),
                meta={"label": label},
                customdata=customdata,
                hovertemplate=(
                    hover_head
                    + f"<br>{metric_label(x_metric)} {x_quantity} = %{{x:.4g}}"
                    + f"<br>{metric_label(y_metric)} {y_quantity} = %{{y:.4g}}"
                    + "<extra>dominée</extra>"
                ),
            )
        )

    front_table = table.iloc[front]
    front_colors = [
        encoder.color(front_table[color].iloc[i] if color else "ensemble")
        for i in range(len(front_table))
    ]
    customdata, hover_head = _hover_customdata(front_table, dataset)
    arrow = {"min": "↓", "max": "↑"}
    figure.add_trace(
        go.Scatter(
            x=front_table[x_quantity],
            y=front_table[y_column],
            mode="lines+markers",
            line={"color": "#22252a", "width": 1.6, "dash": "dot"},
            marker={"size": 13, "color": front_colors,
                    "line": {"color": "#22252a", "width": 2}, "symbol": "diamond"},
            name=f"front de Pareto · {len(front_table)} config.",
            meta={"label": "front de Pareto"},
            customdata=customdata,
            hovertemplate=(
                "<b>★ front de Pareto</b><br>"
                + hover_head
                + f"<br>{metric_label(x_metric)} {x_quantity} = %{{x:.4g}}"
                + f"<br>{metric_label(y_metric)} {y_quantity} = %{{y:.4g}}"
                + "<extra></extra>"
            ),
        )
    )

    figure.update_xaxes(
        title_text=f"{arrow[sense[0]]} " + metric_axis_title(x_quantity, x_metric, dataset.units),
        type="log" if log_x else "linear",
    )
    figure.update_yaxes(
        title_text=f"{arrow[sense[1]]} " + metric_axis_title(y_quantity, y_metric, dataset.units),
        type="log" if log_y else "linear",
    )

    direction = (
        f"{'minimiser' if sense[0] == 'min' else 'maximiser'} {x_quantity}, "
        f"{'minimiser' if sense[1] == 'min' else 'maximiser'} {y_quantity}"
    )
    return finalize(
        figure,
        dataset,
        title=title or f"Front de Pareto : {y_quantity} contre {x_quantity}",
        subtitle=subtitle if subtitle is not None else subtitle_for(dataset, direction),
        legend_title=color or "série",
        height=height,
        width=width,
        interactivity=interactivity if interactivity is not None else {"search": False},
    )


def metrics_table(
    dataset,
    quantities: Sequence[str] | str | None = None,
    metrics: Any = ("max", "mean"),
    sort_by: str | None = None,
    ascending: bool = False,
    digits: int = 4,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Tableau récapitulatif triable des métriques par configuration."""
    metric_map = normalize_metrics(metrics)
    table = dataset.table(quantities, metrics, flat=False).reset_index(drop=True)
    if sort_by and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=ascending)

    columns = ["label", *[c for c in table.columns if c != "label"]]
    values = []
    for column in columns:
        series = table[column]
        if pd.api.types.is_numeric_dtype(series):
            values.append([f"{v:.{digits}g}" if pd.notna(v) else "—" for v in series])
        else:
            values.append([str(v) for v in series])

    # « temperature_max » tient mieux sur deux lignes dans un en-tête de tableau.
    headers = []
    for column in columns:
        for metric_name in metric_map:
            suffix = f"_{metric_name}"
            if column.endswith(suffix):
                column = f"{column[: -len(suffix)]}<br>{metric_name}"
                break
        headers.append(f"<b>{column}</b>")

    # Au-delà de ~30 lignes, le tableau garde une hauteur fixe et défile.
    computed_height = height or min(max(300, 26 * len(table) + 140), 920)

    figure = go.Figure(
        go.Table(
            columnwidth=[2.2] + [1] * (len(columns) - 1),
            header={
                "values": headers,
                "fill_color": "#eef2f7",
                "align": "left",
                "font": {"size": 12, "color": "#111827"},
                "height": 40,
            },
            cells={
                "values": values,
                "align": "left",
                "font": {"size": 11},
                "height": 24,
                "fill_color": [
                    ["white", "#f9fafb"] * (len(table) // 2 + 1),
                ],
            },
        )
    )
    metric_names = ", ".join(metric_label(m) for m in metric_map.values())
    return finalize(
        figure,
        dataset,
        title=title or "Récapitulatif par configuration",
        subtitle=subtitle_for(dataset, f"métriques : {metric_names}"),
        height=computed_height,
        width=width,
        show_legend=False,
        interactivity=False,
    )


def _numeric(values: Sequence[Any]) -> bool:
    series = pd.Series(list(values)).dropna()
    return bool(len(series)) and pd.api.types.is_numeric_dtype(series.infer_objects())


def _rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
