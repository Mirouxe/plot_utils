"""Dashboard récapitulatif d'une configuration (un CSV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..dataset import Dataset
from ..metrics import metric_label
from ..plots import metrics_table
from .common import (
    as_dataset,
    correlation_heatmap,
    filename_of,
    format_compact,
    format_duration,
    highlight_on_envelope,
    nan_report,
    percentile_ranks,
    pick_grouping,
    resolve_config,
    sampling_stats,
    value_histograms,
)
from .page import Dashboard, identity_card

__all__ = ["inspect", "inspect_file", "inspect_figures"]

DEFAULT_METRICS = ("min", "max", "mean", "rms", "t_max", "final", "integral")


def inspect_figures(
    dataset: Dataset,
    config: str,
    quantities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Figures utilisées par le dashboard ; exposées pour les captures PNG."""
    quantities = list(quantities or dataset.quantities)
    subset = dataset.select([config])
    figures: dict[str, Any] = {
        "grille": subset.grid(quantities, row_height=210),
        "correlations": correlation_heatmap(dataset, config, quantities),
        "distributions": value_histograms(dataset, config, quantities),
        "metriques": metrics_table(subset, quantities, metrics=DEFAULT_METRICS),
    }
    if len(dataset) > 1:
        figures["campagne"] = highlight_on_envelope(
            dataset, [config], quantities[0], by=pick_grouping(dataset)
        )
    if len(quantities) >= 2:
        figures["phase"] = subset.curves(
            quantities[1], x=quantities[0], title=f"{quantities[1]} en fonction de {quantities[0]}"
        )
    return figures


def inspect(
    source: Dataset | str | Path,
    config: str | int | None = None,
    path: str | Path = "fiche.html",
    quantities: Sequence[str] | None = None,
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Dashboard récapitulatif d'un CSV / d'une configuration.

    >>> ds.inspect("alu · 20 kW · 1.6 kg/s", "fiche.html")
    >>> cs.inspect_file("cas_materiau=alu_....csv", "fiche.html")
    """
    dataset = as_dataset(source, **load_kwargs)
    if config is None:
        if len(dataset) != 1:
            raise ValueError(
                f"{len(dataset)} configurations chargées : précise config=... "
                "(clé, indice, nom de fichier ou sous-chaîne unique)."
            )
        config_key = dataset.configs[0]
    else:
        config_key = resolve_config(dataset, config)

    quantities = list(quantities or dataset.quantities)
    frame = dataset.frames[config_key]
    stats = sampling_stats(frame, dataset.time)
    missing = nan_report(frame)
    figures = inspect_figures(dataset, config_key, quantities)
    chars = dataset.characteristics_of(config_key)

    page = Dashboard(
        title=dataset.label(config_key),
        subtitle=filename_of(dataset, config_key),
        eyebrow="Fiche configuration",
        footer="Fiche générée avec csvscope.inspect.",
    )

    page.tab("Vue d'ensemble")
    page.kpis(
        [
            {"label": "Points", "value": f"{int(stats['n'])}"},
            {
                "label": "Durée",
                "value": format_duration(stats["duration"]),
                "hint": f"{dataset.time} = {format_compact(stats['t0'])} → {format_compact(stats['t1'])}",
            },
            {
                "label": "Pas de temps",
                "value": format_compact(stats["dt"]),
                "hint": f"CV = {format_compact(stats['dt_cv'])}" if stats["dt_cv"] == stats["dt_cv"] else "",
                "tone": "warn" if (stats["dt_cv"] == stats["dt_cv"] and stats["dt_cv"] > 0.05) else "",
            },
            {"label": "Grandeurs", "value": str(len(quantities))},
            {
                "label": "Valeurs manquantes",
                "value": str(int(stats["n_nan"])),
                "tone": "bad" if stats["n_nan"] else "ok",
                "hint": ", ".join(f"{k}: {v}" for k, v in missing.items()) if missing else "aucune",
            },
        ]
    )
    page.chips(
        [{"label": key, "value": format_compact(value), "tone": "same"} for key, value in chars.items()]
    )
    if missing:
        page.note(
            "Des valeurs manquantes ont été détectées : "
            + ", ".join(f"<code>{k}</code> ({v})" for k, v in missing.items())
            + ".",
            tone="warn",
        )
    page.add(
        identity_card(
            dataset.label(config_key),
            filename_of(dataset, config_key),
            chars,
        )
    )
    page.figure(figures["grille"])

    page.tab("Métriques")
    page.figure(figures["metriques"])
    if len(dataset) > 1:
        ranks = percentile_ranks(dataset, config_key, quantities)
        rows = [
            [
                row.grandeur,
                format_compact(row.valeur),
                f"{row.percentile:.0f} %",
            ]
            for row in ranks.itertuples(index=False)
        ]
        page.note(
            "Percentile : part des configurations de la campagne dont le maximum "
            "de la grandeur est <b>inférieur ou égal</b> à celui de cette fiche "
            f"({len(dataset)} configurations)."
        )
        page.table(["Grandeur", f"{metric_label('max')}", "Percentile campagne"], rows)

    page.tab("Distributions")
    page.figure(figures["distributions"])
    page.figure(figures["correlations"])
    if "phase" in figures:
        page.figure(figures["phase"])

    if "campagne" in figures:
        page.tab("Dans la campagne")
        page.note(
            "Le faisceau (médiane et bande P10–P90) résume les "
            f"{len(dataset)} configurations ; la courbe mise en avant est celle de cette fiche."
        )
        page.figure(figures["campagne"])

    return page.write(path, plotlyjs=plotlyjs)


def inspect_file(
    source: str | Path,
    path: str | Path = "fiche.html",
    plotlyjs: str = "cdn",
    **load_kwargs: Any,
) -> Path:
    """Raccourci : charger un CSV isolé et en faire la fiche."""
    dataset = as_dataset(source, **load_kwargs)
    return inspect(dataset, dataset.configs[0], path=path, plotlyjs=plotlyjs)
