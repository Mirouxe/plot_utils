"""Dashboard de comparaison de deux configurations (deux CSV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..dataset import Dataset, load
from ..metrics import metric_label
from .common import (
    align_pair,
    as_dataset,
    delta_dataset,
    filename_of,
    format_compact,
    highlight_on_envelope,
    pick_grouping,
    resolve_config,
    series_scores,
)
from .page import Dashboard, identity_card, split_html

__all__ = ["diff", "diff_files", "diff_figures"]

COMPARE_METRICS = ("max", "mean", "rms", "final", "integral")


def _differing_keys(dataset: Dataset, a: str, b: str) -> list[str]:
    left, right = dataset.characteristics_of(a), dataset.characteristics_of(b)
    keys = list(dict.fromkeys([*left, *right]))
    return [key for key in keys if left.get(key) != right.get(key)]


def diff_figures(
    dataset: Dataset,
    a: str,
    b: str,
    quantities: Sequence[str] | None = None,
    points: int = 400,
) -> dict[str, Any]:
    quantities = list(quantities or dataset.quantities)
    pair = dataset.select([a, b])
    time, series = align_pair(dataset, a, b, points=points)
    delta = delta_dataset(dataset, a, b, time, {q: series[q] for q in quantities if q in series})
    figures: dict[str, Any] = {
        "superposition": pair.grid(quantities, row_height=210),
        "ecarts": delta.grid(quantities, row_height=210, title="Écart (A − B)"),
    }
    if len(quantities) >= 3:
        figures["radar"] = pair.radar(quantities, metric="max", normalize="minmax")
    if len(dataset) > 2:
        figures["campagne"] = highlight_on_envelope(
            dataset, [a, b], quantities[0], by=pick_grouping(dataset)
        )
    return figures


def diff(
    source: Dataset | str | Path,
    a: str | int,
    b: str | int,
    path: str | Path = "comparaison.html",
    quantities: Sequence[str] | None = None,
    points: int = 400,
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Dashboard récapitulatif de la comparaison de deux CSV.

    >>> ds.diff("alu · 20 kW", "cuivre · 20 kW", "comparaison.html")
    >>> cs.diff_files("a.csv", "b.csv", "comparaison.html")
    """
    dataset = as_dataset(source, **load_kwargs)
    key_a, key_b = resolve_config(dataset, a), resolve_config(dataset, b)
    if key_a == key_b:
        raise ValueError("Les deux configurations sont identiques.")

    quantities = list(quantities or dataset.quantities)
    differing = _differing_keys(dataset, key_a, key_b)
    time, series = align_pair(dataset, key_a, key_b, points=points)
    scores = {q: series_scores(*series[q]) for q in quantities if q in series}
    finite = [s for s in scores.values() if np.isfinite(s["rmse"])]
    mean_corr = float(np.nanmean([s["corr"] for s in finite])) if finite else float("nan")
    mean_rmse = float(np.nanmean([s["rmse_norm"] for s in finite])) if finite else float("nan")
    max_rel = float(np.nanmax([s["rel_max"] for s in finite])) if finite else float("nan")
    figures = diff_figures(dataset, key_a, key_b, quantities, points=points)

    label_a, label_b = dataset.label(key_a), dataset.label(key_b)
    page = Dashboard(
        title=f"{label_a}  vs  {label_b}",
        subtitle=f"{filename_of(dataset, key_a)}  ·  {filename_of(dataset, key_b)}",
        eyebrow="Comparaison de deux configurations",
        footer="Comparaison générée avec csvscope.diff.",
    )

    page.tab("Identité")
    page.kpis(
        [
            {
                "label": "Caractéristiques différentes",
                "value": str(len(differing)),
                "hint": ", ".join(differing) if differing else "aucune — mêmes paramètres",
                "tone": "warn" if differing else "ok",
            },
            {
                "label": "Corrélation moyenne",
                "value": format_compact(mean_corr),
                "hint": "séries ré-échantillonnées",
                "tone": "ok" if np.isfinite(mean_corr) and mean_corr > 0.95 else "",
            },
            {
                "label": "RMSE normalisé",
                "value": format_compact(mean_rmse),
                "hint": "moyenne sur les grandeurs",
            },
            {
                "label": "Écart relatif max",
                "value": f"{100 * max_rel:.2g} %" if np.isfinite(max_rel) else "—",
            },
            {"label": "Points alignés", "value": str(len(time))},
        ]
    )
    chips = []
    left, right = dataset.characteristics_of(key_a), dataset.characteristics_of(key_b)
    for key in list(dict.fromkeys([*left, *right])):
        if key in differing:
            chips.append(
                {
                    "label": key,
                    "value": f"{format_compact(left.get(key))} → {format_compact(right.get(key))}",
                    "tone": "diff",
                }
            )
        else:
            chips.append({"label": key, "value": format_compact(left.get(key)), "tone": "same"})
    page.chips(chips)
    page.add(
        split_html(
            identity_card(label_a, filename_of(dataset, key_a), left, differing),
            identity_card(label_b, filename_of(dataset, key_b), right, differing),
        )
    )

    page.tab("Superposition")
    page.note(
        f"Courbes de <b>{label_a}</b> et <b>{label_b}</b> sur les mêmes axes. "
        "Survole une courbe pour l'isoler."
    )
    page.figure(figures["superposition"])

    page.tab("Écarts")
    page.note("Différence point par point après ré-échantillonnage sur une grille commune : A − B.")
    page.figure(figures["ecarts"])
    score_rows = [
        [
            quantity,
            format_compact(scores[quantity]["rmse"]),
            format_compact(scores[quantity]["corr"]),
            format_compact(scores[quantity]["max_abs"]),
            f"{100 * scores[quantity]['rel_max']:.2g} %"
            if np.isfinite(scores[quantity]["rel_max"])
            else "—",
        ]
        for quantity in quantities
        if quantity in scores
    ]
    page.table(
        ["Grandeur", "RMSE", "Corrélation", "|Δ| max", "Δ relatif max"],
        score_rows,
    )

    page.tab("Métriques")
    table_a = dataset.select([key_a]).table(quantities, COMPARE_METRICS, flat=False)
    table_b = dataset.select([key_b]).table(quantities, COMPARE_METRICS, flat=False)
    headers = ["Grandeur", "Métrique", label_a, label_b, "Δ (A − B)", "Δ %"]
    rows, classes = [], []
    for quantity in quantities:
        for metric in COMPARE_METRICS:
            col = f"{quantity}_{metric}"
            if col not in table_a.columns:
                continue
            va, vb = float(table_a.at[key_a, col]), float(table_b.at[key_b, col])
            delta = va - vb
            rel = 100 * delta / vb if vb else float("nan")
            rows.append(
                [
                    quantity,
                    metric_label(metric),
                    format_compact(va),
                    format_compact(vb),
                    format_compact(delta),
                    f"{rel:.2g} %" if np.isfinite(rel) else "—",
                ]
            )
            mark = ["", "", "", "", "diff" if abs(delta) > 0 else "", "diff" if abs(delta) > 0 else ""]
            classes.append(mark)
    page.table(headers, rows, cell_classes=classes)
    if "radar" in figures:
        page.figure(figures["radar"])

    if "campagne" in figures:
        page.tab("Dans la campagne")
        page.note(
            "Les deux configurations, posées sur le faisceau des "
            f"{len(dataset)} configurations de la campagne."
        )
        page.figure(figures["campagne"])

    return page.write(path, plotlyjs=plotlyjs)


def diff_files(
    file_a: str | Path,
    file_b: str | Path,
    path: str | Path = "comparaison.html",
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Raccourci : comparer deux fichiers CSV isolés."""
    dataset = load([file_a, file_b], **load_kwargs)
    return diff(dataset, dataset.configs[0], dataset.configs[1], path=path, plotlyjs=plotlyjs)
