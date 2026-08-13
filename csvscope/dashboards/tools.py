"""Petits outils d'exploration : une grandeur, un instant, aberrantes, voisins, couverture, qualité."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..dataset import Dataset
from ..metrics import metric_label
from ..plots import metrics_table
from ..plots.common import finalize, format_number
from ..theme import QUALITATIVE, sort_values_naturally
from .common import (
    as_dataset,
    filename_of,
    format_compact,
    pick_characteristics,
    pick_grouping,
    resolve_config,
    sampling_stats,
)
from .page import Dashboard

__all__ = [
    "coverage",
    "neighbors",
    "outliers",
    "quality",
    "quantity_board",
    "snapshot",
]


def quantity_board(
    source: Dataset | str | Path,
    quantity: str,
    path: str | Path = "grandeur.html",
    metric: Any = "max",
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Toutes les vues utiles d'une grandeur sur la campagne."""
    dataset = as_dataset(source, **load_kwargs)
    if quantity not in dataset.quantities:
        raise KeyError(f"Grandeur inconnue : {quantity!r}. Disponibles : {dataset.quantities}")
    picked = pick_characteristics(dataset, 2)
    group = pick_grouping(dataset)
    x = picked[0] if picked else None
    color = picked[1] if len(picked) > 1 else group

    page = Dashboard(
        title=f"Exploration de {quantity}",
        subtitle=f"{len(dataset)} configurations · métrique : {metric_label(metric)}",
        eyebrow="Dashboard grandeur",
        footer="Dashboard généré avec csvscope.quantity_board.",
    )
    page.tab("Courbes")
    page.figure(dataset.curves(quantity, color=group or color))
    page.tab("Faisceau")
    page.figure(dataset.envelope(quantity, by=group))
    if x:
        page.tab("Sensibilité")
        page.figure(dataset.compare(quantity, metric=metric, x=x, color=group if group != x else None))
        if group:
            page.figure(dataset.distribution(quantity, by=group, metric=metric, kind="box"))
    page.tab("Classement")
    page.figure(dataset.bars(quantity, metric=metric, color=group))
    if x and group and x != group:
        page.tab("Carte")
        page.figure(dataset.heatmap(quantity, x=x, y=group, metric=metric))
    page.tab("Récapitulatif")
    page.figure(metrics_table(dataset, [quantity], metrics=("min", "max", "mean", "rms", "t_max", "integral")))
    return page.write(path, plotlyjs=plotlyjs)


def snapshot(
    source: Dataset | str | Path,
    at: float | str = "final",
    path: str | Path = "instant.html",
    quantities: Sequence[str] | None = None,
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Valeurs de toutes les configurations à un instant donné.

    ``at`` accepte un temps numérique, ``"initial"``, ``"final"``, ou
    ``"t_max:grandeur"`` (instant du maximum de cette grandeur, médian sur la campagne).
    """
    dataset = as_dataset(source, **load_kwargs)
    quantities = list(quantities or dataset.quantities)
    time_value, label = _resolve_instant(dataset, at, quantities[0])
    table = _values_at(dataset, time_value, quantities)

    picked = pick_characteristics(dataset, 2)
    color = picked[0] if picked else None
    page = Dashboard(
        title=f"Instantané à {dataset.time} = {format_compact(time_value)}",
        subtitle=f"{label} · {len(dataset)} configurations",
        eyebrow="Coupe temporelle",
        footer="Dashboard généré avec csvscope.snapshot.",
    )
    page.tab("Valeurs")
    page.kpis(
        [
            {"label": "Instant", "value": format_compact(time_value), "hint": dataset.time},
            {"label": "Configurations", "value": str(len(dataset))},
            {"label": "Grandeurs", "value": str(len(quantities))},
            {
                "label": f"Médiane {quantities[0]}",
                "value": format_compact(float(table[quantities[0]].median())),
            },
        ]
    )
    rows = [
        [dataset.label(config), *[format_compact(table.at[config, q]) for q in quantities]]
        for config in table.index
    ]
    page.table(["configuration", *quantities], rows)

    page.tab("Graphiques")
    page.figure(_snapshot_bars(dataset, table, quantities[0], color))
    if len(quantities) >= 2:
        page.figure(_snapshot_scatter(dataset, table, quantities[0], quantities[1], color))
    if len(picked) >= 2:
        page.figure(
            _snapshot_heatmap(dataset, table, quantities[0], picked[0], picked[1])
        )
    return page.write(path, plotlyjs=plotlyjs)


def outliers(
    source: Dataset | str | Path,
    path: str | Path = "aberrantes.html",
    quantities: Sequence[str] | None = None,
    metric: Any = "max",
    z: float = 2.5,
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Configurations dont une métrique s'écarte fortement de la campagne."""
    dataset = as_dataset(source, **load_kwargs)
    quantities = list(quantities or dataset.quantities)
    table = dataset.table(quantities, metric)
    scores = pd.DataFrame(index=table.index)
    for quantity in quantities:
        col = table[quantity]
        std = col.std(ddof=1)
        scores[quantity] = 0.0 if not std or not np.isfinite(std) else (col - col.mean()) / std
    scores["|z| max"] = scores[quantities].abs().max(axis=1)
    scores["grandeur"] = scores[quantities].abs().idxmax(axis=1)
    flagged = scores[scores["|z| max"] >= z].sort_values("|z| max", ascending=False)
    color = pick_characteristics(dataset, 1)
    color = color[0] if color else None

    page = Dashboard(
        title="Configurations aberrantes",
        subtitle=f"seuil |z| ≥ {z:g} sur {metric_label(metric)} · {len(dataset)} configurations",
        eyebrow="Détection d'écarts",
        footer="Dashboard généré avec csvscope.outliers.",
    )
    page.tab("Classement")
    page.kpis(
        [
            {"label": "Aberrantes", "value": str(len(flagged)), "tone": "warn" if len(flagged) else "ok"},
            {"label": "Seuil |z|", "value": f"{z:g}"},
            {"label": "Métrique", "value": metric_label(metric)},
            {
                "label": "|z| max observé",
                "value": format_compact(float(scores["|z| max"].max())),
            },
        ]
    )
    if flagged.empty:
        page.note("Aucune configuration ne dépasse le seuil.", tone="")
    rows, classes = [], []
    for config in flagged.index:
        row = [
            dataset.label(config),
            flagged.at[config, "grandeur"],
            format_compact(float(flagged.at[config, "|z| max"])),
            *[format_compact(float(scores.at[config, q])) for q in quantities],
        ]
        rows.append(row)
        classes.append(["", "", "bad"] + [""] * len(quantities))
    if rows:
        page.table(
            ["configuration", "grandeur la plus extrême", "|z| max", *[f"z {q}" for q in quantities]],
            rows,
            cell_classes=classes,
        )
    page.tab("Contexte")
    page.figure(dataset.bars(quantities[0], metric=metric, color=color, top=min(30, len(dataset))))
    if len(quantities) >= 2:
        page.figure(dataset.scatter((quantities[0], metric), (quantities[1], metric), color=color))
    if not flagged.empty:
        subset = dataset.select(list(flagged.index[:12]))
        page.tab("Courbes")
        page.note("Les configurations signalées, superposées sur la première grandeur.")
        page.figure(subset.curves(quantities[0], color=color))
    return page.write(path, plotlyjs=plotlyjs)


def neighbors(
    source: Dataset | str | Path,
    config: str | int,
    path: str | Path = "proches.html",
    k: int = 8,
    quantities: Sequence[str] | None = None,
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Configurations dont les séries temporelles sont les plus proches d'une fiche."""
    dataset = as_dataset(source, **load_kwargs)
    query = resolve_config(dataset, config)
    quantities = list(quantities or dataset.quantities)
    ranking = _nearest(dataset, query, quantities, k=k)
    keys = [query, *[row["config"] for row in ranking]]
    subset = dataset.select(keys)

    page = Dashboard(
        title=f"Voisines de {dataset.label(query)}",
        subtitle=f"distance = RMSE moyen, normalisé par grandeur · k = {k}",
        eyebrow="Configurations proches",
        footer="Dashboard généré avec csvscope.neighbors.",
    )
    page.tab("Classement")
    page.kpis(
        [
            {"label": "Requête", "value": dataset.label(query), "hint": filename_of(dataset, query)},
            {"label": "Voisines", "value": str(len(ranking))},
            {
                "label": "Plus proche",
                "value": dataset.label(ranking[0]["config"]) if ranking else "—",
                "hint": f"d = {format_compact(ranking[0]['distance'])}" if ranking else "",
            },
        ]
    )
    rows = [
        [
            dataset.label(row["config"]),
            format_compact(row["distance"]),
            ", ".join(
                f"{key}={format_compact(dataset.meta.at[row['config'], key])}"
                for key in dataset.characteristics
                if dataset.meta.at[row["config"], key] != dataset.meta.at[query, key]
            )
            or "identiques",
        ]
        for row in ranking
    ]
    page.table(["configuration", "distance", "caractéristiques différentes"], rows)
    page.tab("Courbes")
    page.figure(subset.grid(quantities[: min(4, len(quantities))], row_height=200))
    if dataset.characteristics:
        page.tab("Caractéristiques")
        headers = ["configuration", *dataset.characteristics, "distance"]
        char_rows, classes = [], []
        for key in keys:
            dist = 0.0 if key == query else next(r["distance"] for r in ranking if r["config"] == key)
            char_rows.append(
                [
                    dataset.label(key),
                    *[format_compact(dataset.meta.at[key, c]) for c in dataset.characteristics],
                    format_compact(dist),
                ]
            )
            classes.append(
                [""]
                + [
                    "diff" if dataset.meta.at[key, c] != dataset.meta.at[query, c] else ""
                    for c in dataset.characteristics
                ]
                + [""]
            )
        page.table(headers, char_rows, cell_classes=classes)
    return page.write(path, plotlyjs=plotlyjs)


def coverage(
    source: Dataset | str | Path,
    path: str | Path = "couverture.html",
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Couverture de la campagne : combinaisons présentes, manquantes, marges."""
    dataset = as_dataset(source, **load_kwargs)
    varying = [c for c in dataset.characteristics if len(dataset.values(c)) > 1]
    sizes = [len(dataset.values(c)) for c in varying]
    factorial = int(np.prod(sizes)) if sizes else len(dataset)
    ratio = len(dataset) / factorial if factorial else 1.0

    page = Dashboard(
        title="Couverture de la campagne",
        subtitle=f"{len(dataset)} configurations · {len(varying)} caractéristiques variables",
        eyebrow="Plan d'expérience",
        footer="Dashboard généré avec csvscope.coverage.",
    )
    page.tab("Synthèse")
    page.kpis(
        [
            {"label": "Configurations", "value": str(len(dataset))},
            {"label": "Factoriel complet", "value": str(factorial), "hint": " × ".join(str(s) for s in sizes)},
            {
                "label": "Couverture",
                "value": f"{100 * ratio:.0f} %",
                "tone": "ok" if ratio >= 0.99 else ("warn" if ratio >= 0.5 else "bad"),
            },
            {"label": "Caractéristiques", "value": str(len(dataset.characteristics))},
        ]
    )
    page.chips(
        [
            {"label": c, "value": f"{len(dataset.values(c))} valeurs", "tone": "same"}
            for c in dataset.characteristics
        ]
    )
    if len(varying) >= 2:
        page.tab("Croisements")
        x, y = varying[0], varying[1]
        page.figure(_count_heatmap(dataset, x, y))
        if len(varying) >= 3:
            page.figure(_count_heatmap(dataset, varying[0], varying[2]))
        missing = _missing_pairs(dataset, x, y)
        if missing:
            page.note(f"{len(missing)} combinaisons {x} × {y} absentes.", tone="warn")
            page.table([x, y], [[format_compact(a), format_compact(b)] for a, b in missing[:80]])
        else:
            page.note(f"Toutes les combinaisons {x} × {y} sont présentes.")
    page.tab("Liste")
    rows = [
        [dataset.label(c), filename_of(dataset, c), *[format_compact(dataset.meta.at[c, k]) for k in dataset.characteristics]]
        for c in dataset.configs
    ]
    page.table(["configuration", "fichier", *dataset.characteristics], rows)
    return page.write(path, plotlyjs=plotlyjs)


def quality(
    source: Dataset | str | Path,
    path: str | Path = "qualite.html",
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Contrôle qualité des CSV : sampling, valeurs manquantes, durées."""
    dataset = as_dataset(source, **load_kwargs)
    records = []
    for config in dataset.configs:
        stats = sampling_stats(dataset.frames[config], dataset.time)
        stats["config"] = config
        records.append(stats)
    frame = pd.DataFrame(records).set_index("config")
    median_n = float(frame["n"].median())
    issues = []
    for config, row in frame.iterrows():
        flags = []
        if row["n_nan"] > 0:
            flags.append("NaN")
        if np.isfinite(row["dt_cv"]) and row["dt_cv"] > 0.05:
            flags.append("pas irrégulier")
        if median_n and abs(row["n"] - median_n) / median_n > 0.1:
            flags.append("longueur atypique")
        if flags:
            issues.append((config, flags, row))

    page = Dashboard(
        title="Qualité des CSV",
        subtitle=f"{len(dataset)} fichiers · {len(issues)} signalés",
        eyebrow="Contrôle",
        footer="Dashboard généré avec csvscope.quality.",
    )
    page.tab("Synthèse")
    page.kpis(
        [
            {"label": "Fichiers", "value": str(len(dataset))},
            {"label": "Signalés", "value": str(len(issues)), "tone": "warn" if issues else "ok"},
            {"label": "Points (médiane)", "value": format_compact(median_n)},
            {
                "label": "NaN totaux",
                "value": format_compact(float(frame["n_nan"].sum())),
                "tone": "bad" if frame["n_nan"].sum() else "ok",
            },
        ]
    )
    page.figure(_quality_hist(dataset, frame["n"], "Nombre de points par fichier"))
    page.figure(_quality_hist(dataset, frame["dt_cv"].replace([np.inf, -np.inf], np.nan), "Coefficient de variation du pas de temps"))
    page.tab("Détail")
    rows, classes = [], []
    for config, row in frame.iterrows():
        flag = next((f for c, f, _ in issues if c == config), [])
        rows.append(
            [
                dataset.label(config),
                int(row["n"]),
                format_compact(row["duration"]),
                format_compact(row["dt"]),
                format_compact(row["dt_cv"]),
                int(row["n_nan"]),
                ", ".join(flag) or "ok",
            ]
        )
        classes.append(["", "", "", "", "", "bad" if row["n_nan"] else "ok", "bad" if flag else "ok"])
    page.table(
        ["configuration", "points", "durée", "Δt médian", "CV(Δt)", "NaN", "statut"],
        rows,
        cell_classes=classes,
    )
    return page.write(path, plotlyjs=plotlyjs)


# ------------------------------------------------------------------ internes


def _resolve_instant(dataset: Dataset, at: float | str, quantity: str) -> tuple[float, str]:
    times = []
    for frame in dataset.frames.values():
        series = pd.to_numeric(frame[dataset.time], errors="coerce").dropna()
        times.append((float(series.min()), float(series.max())))
    t0, t1 = min(t[0] for t in times), max(t[1] for t in times)
    if at == "final":
        return t1, "fin de série (maximum des temps)"
    if at == "initial":
        return t0, "début de série (minimum des temps)"
    if isinstance(at, str) and at.startswith("t_max"):
        target = at.split(":", 1)[1] if ":" in at else quantity
        peaks = dataset.table(target, "t_max")[target]
        return float(peaks.median()), f"instant médian du maximum de {target}"
    value = float(at)
    return value, "instant imposé"


def _values_at(dataset: Dataset, time_value: float, quantities: Sequence[str]) -> pd.DataFrame:
    rows = []
    for config, frame in dataset.frames.items():
        t = pd.to_numeric(frame[dataset.time], errors="coerce").to_numpy(dtype=float)
        row: dict[str, Any] = {"config": config}
        for quantity in quantities:
            y = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(t) & np.isfinite(y)
            row[quantity] = float(np.interp(time_value, t[mask], y[mask])) if mask.sum() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def _snapshot_bars(dataset: Dataset, table: pd.DataFrame, quantity: str, color: str | None) -> go.Figure:
    del color  # réservé si l'on colore les barres plus tard
    order = table[quantity].sort_values()
    keep = order.tail(min(40, len(order)))
    figure = go.Figure(
        go.Bar(
            x=keep.to_numpy(dtype=float),
            y=[dataset.label(c) for c in keep.index],
            orientation="h",
            marker={"color": QUALITATIVE[0]},
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    return finalize(
        figure,
        dataset,
        title=f"{quantity} à cet instant" + (f" (top {len(keep)})" if len(keep) < len(table) else ""),
        y_title="",
        x_title=quantity,
        height=max(340, 22 * len(keep) + 120),
        show_legend=False,
        interactivity=False,
    )


def _snapshot_scatter(
    dataset: Dataset, table: pd.DataFrame, x: str, y: str, color: str | None
) -> go.Figure:
    merged = table.join(dataset.meta)
    figure = go.Figure()
    groups = dataset.values(color) if color else [None]
    for index, value in enumerate(groups):
        subset = merged if color is None else merged[merged[color] == value]
        figure.add_trace(
            go.Scatter(
                x=subset[x],
                y=subset[y],
                mode="markers",
                name=format_number(value) if color else "configurations",
                marker={"size": 9, "color": QUALITATIVE[index % len(QUALITATIVE)], "opacity": 0.8},
                text=[dataset.label(c) for c in subset.index],
                hovertemplate="<b>%{text}</b><br>" + f"{x} = %{{x:.4g}}<br>{y} = %{{y:.4g}}<extra></extra>",
            )
        )
    return finalize(
        figure,
        dataset,
        title=f"{y} en fonction de {x} à cet instant",
        x_title=x,
        y_title=y,
        legend_title=color or "série",
        interactivity={"search": False},
    )


def _snapshot_heatmap(dataset: Dataset, table: pd.DataFrame, quantity: str, x: str, y: str) -> go.Figure:
    merged = table.join(dataset.meta)
    pivot = merged.pivot_table(index=y, columns=x, values=quantity, aggfunc="mean")
    pivot = pivot.reindex(
        index=sort_values_naturally(pivot.index.tolist()),
        columns=sort_values_naturally(pivot.columns.tolist()),
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}<br>{quantity} = %{{z:.4g}}<extra></extra>",
            colorbar={"title": {"text": quantity, "side": "right"}, "thickness": 14, "outlinewidth": 0},
        )
    )
    figure.update_xaxes(type="category", title_text=x)
    figure.update_yaxes(type="category", title_text=y, showgrid=False)
    return finalize(
        figure,
        dataset,
        title=f"{quantity} à cet instant : {y} × {x}",
        height=480,
        show_legend=False,
        interactivity=False,
    )


def _nearest(
    dataset: Dataset, query: str, quantities: Sequence[str], k: int
) -> list[dict[str, Any]]:
    resampled = dataset.resample(min(250, max(len(next(iter(dataset.frames.values()))), 50)))
    ref = {q: resampled.frames[query][q].to_numpy(dtype=float) for q in quantities}
    scales = {
        q: max(float(np.nanstd(np.concatenate([resampled.frames[c][q].to_numpy(float) for c in resampled.configs]))), 1e-12)
        for q in quantities
    }
    ranking = []
    for config in resampled.configs:
        if config == query:
            continue
        parts = []
        for quantity in quantities:
            ya, yb = ref[quantity], resampled.frames[config][quantity].to_numpy(dtype=float)
            mask = np.isfinite(ya) & np.isfinite(yb)
            if mask.sum() < 2:
                continue
            parts.append(float(np.sqrt(np.mean((ya[mask] - yb[mask]) ** 2))) / scales[quantity])
        if not parts:
            continue
        ranking.append({"config": config, "distance": float(np.mean(parts))})
    ranking.sort(key=lambda row: row["distance"])
    return ranking[:k]


def _count_heatmap(dataset: Dataset, x: str, y: str) -> go.Figure:
    table = dataset.meta.reset_index()
    pivot = table.pivot_table(index=y, columns=x, values="config", aggfunc="count").fillna(0)
    pivot = pivot.reindex(
        index=sort_values_naturally(pivot.index.tolist()),
        columns=sort_values_naturally(pivot.columns.tolist()),
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale=[[0, "#f7f8fa"], [1, QUALITATIVE[0]]],
            hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}<br>%{{z}} configuration(s)<extra></extra>",
            colorbar={"title": {"text": "n", "side": "right"}, "thickness": 14, "outlinewidth": 0},
            texttemplate="%{z:.0f}",
            textfont={"size": 12},
        )
    )
    figure.update_xaxes(type="category", title_text=x)
    figure.update_yaxes(type="category", title_text=y, showgrid=False)
    return finalize(
        figure,
        dataset,
        title=f"Nombre de configurations : {y} × {x}",
        height=420,
        show_legend=False,
        interactivity=False,
    )


def _missing_pairs(dataset: Dataset, x: str, y: str) -> list[tuple[Any, Any]]:
    present = set(zip(dataset.meta[x], dataset.meta[y]))
    missing = [
        (a, b)
        for a, b in product(dataset.values(x), dataset.values(y))
        if (a, b) not in present
    ]
    return missing


def _quality_hist(dataset: Dataset, values: pd.Series, title: str) -> go.Figure:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    figure = go.Figure(
        go.Histogram(
            x=clean,
            marker={"color": QUALITATIVE[0]},
            hovertemplate="%{x:.4g} · %{y} fichiers<extra></extra>",
        )
    )
    return finalize(
        figure,
        dataset,
        title=title,
        height=320,
        show_legend=False,
        interactivity=False,
    )
