"""Helpers partagés par les dashboards : résolution, stats, alignement, figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..dataset import Dataset, load, load_frames
from ..plots.common import finalize, format_number
from ..theme import QUALITATIVE, SEQUENTIAL, axis_title

__all__ = [
    "align_pair",
    "as_dataset",
    "correlation_heatmap",
    "delta_dataset",
    "filename_of",
    "format_compact",
    "format_duration",
    "highlight_on_envelope",
    "nan_report",
    "percentile_ranks",
    "pick_characteristics",
    "pick_grouping",
    "resolve_config",
    "sampling_stats",
    "series_scores",
    "value_histograms",
]


def as_dataset(source: Dataset | str | Path | Sequence[str | Path], **load_kwargs) -> Dataset:
    if isinstance(source, Dataset):
        return source
    return load(source, **load_kwargs)


def resolve_config(dataset: Dataset, spec: str | int) -> str:
    """Retrouve une configuration par clé, indice, nom de fichier ou sous-chaîne unique."""
    configs = dataset.configs
    if isinstance(spec, int):
        if spec < 0:
            spec = len(configs) + spec
        if not 0 <= spec < len(configs):
            raise IndexError(
                f"Indice {spec} hors limites ({len(configs)} configurations)."
            )
        return configs[spec]

    spec = str(spec)
    if spec in dataset.frames:
        return spec

    stems = []
    for config in configs:
        path = Path(dataset.file(config))
        stems.append((config, path.name, path.stem, dataset.label(config)))
        if spec in {path.name, path.stem}:
            return config

    lowered = spec.lower()
    matches = [
        config
        for config, name, stem, label in stems
        if lowered in config.lower()
        or lowered in name.lower()
        or lowered in stem.lower()
        or lowered in str(label).lower()
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        sample = ", ".join(dataset.label(c) for c in configs[:6])
        more = "…" if len(configs) > 6 else ""
        raise KeyError(
            f"Aucune configuration ne correspond à {spec!r}. "
            f"Exemples : {sample}{more}"
        )
    shown = ", ".join(dataset.label(c) for c in unique[:8])
    raise KeyError(
        f"{spec!r} est ambigu ({len(unique)} configurations) : {shown}"
        + ("…" if len(unique) > 8 else "")
        + ". Affine la recherche."
    )


def pick_grouping(dataset: Dataset) -> str | None:
    """Caractéristique à peu de modalités, adaptée aux faisceaux et couleurs de groupe."""
    varying = [c for c in dataset.characteristics if len(dataset.values(c)) > 1]
    if not varying:
        return None
    return min(varying, key=lambda c: (len(dataset.values(c)), c))


def pick_characteristics(dataset: Dataset, count: int = 2) -> list[str]:
    ranked = sorted(
        (c for c in dataset.characteristics if len(dataset.values(c)) > 1),
        key=lambda c: (-len(dataset.values(c)), c),
    )
    if not ranked:
        ranked = list(dataset.characteristics)
    return ranked[:count]


def sampling_stats(frame: pd.DataFrame, time_column: str) -> dict[str, float]:
    time = pd.to_numeric(frame[time_column], errors="coerce").dropna().to_numpy(dtype=float)
    n_nan = int(frame.isna().sum().sum())
    if time.size == 0:
        return {
            "n": float(len(frame)),
            "t0": float("nan"),
            "t1": float("nan"),
            "duration": float("nan"),
            "dt": float("nan"),
            "dt_cv": float("nan"),
            "n_nan": float(n_nan),
        }
    time = np.sort(time)
    duration = float(time[-1] - time[0]) if time.size > 1 else 0.0
    deltas = np.diff(time)
    dt = float(np.median(deltas)) if deltas.size else float("nan")
    dt_cv = float(np.std(deltas) / dt) if deltas.size and dt else float("nan")
    return {
        "n": float(time.size),
        "t0": float(time[0]),
        "t1": float(time[-1]),
        "duration": duration,
        "dt": dt,
        "dt_cv": dt_cv,
        "n_nan": float(n_nan),
    }


def nan_report(frame: pd.DataFrame) -> dict[str, int]:
    return {str(col): int(frame[col].isna().sum()) for col in frame.columns if frame[col].isna().any()}


def align_pair(
    dataset: Dataset, a: str, b: str, points: int = 400
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Ré-échantillonne deux configurations sur une grille commune."""
    pair = dataset.select([a, b]).resample(points)
    time = pair.frames[a][dataset.time].to_numpy(dtype=float)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for quantity in pair.quantities:
        series[quantity] = (
            pair.frames[a][quantity].to_numpy(dtype=float),
            pair.frames[b][quantity].to_numpy(dtype=float),
        )
    return time, series


def series_scores(ya: np.ndarray, yb: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ya) & np.isfinite(yb)
    if mask.sum() < 2:
        return {"rmse": float("nan"), "corr": float("nan"), "max_abs": float("nan"), "rel_max": float("nan")}
    da, db = ya[mask], yb[mask]
    delta = da - db
    span = max(float(np.ptp(np.concatenate([da, db]))), 1e-12)
    floor = max(0.05 * span, 1e-12)
    scale = np.maximum(np.abs(db), floor)
    corr = float(np.corrcoef(da, db)[0, 1]) if da.size > 1 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "corr": corr,
        "max_abs": float(np.max(np.abs(delta))),
        "rel_max": float(np.max(np.abs(delta) / scale)),
        "rmse_norm": float(np.sqrt(np.mean(delta**2)) / span),
    }


def correlation_heatmap(dataset: Dataset, config: str, quantities: Sequence[str] | None = None) -> go.Figure:
    quantities = list(quantities or dataset.quantities)
    frame = dataset.frames[config][quantities]
    corr = frame.corr(numeric_only=True)
    scale = list(SEQUENTIAL)
    figure = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(dtype=float),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#3b6ea5"], [0.5, "#f7f8fa"], [1, "#d1495b"]],
            hovertemplate="%{y} × %{x} = %{z:.3f}<extra></extra>",
            colorbar={"title": {"text": "corr.", "side": "right"}, "thickness": 14, "outlinewidth": 0},
        )
    )
    figure.update_xaxes(side="bottom")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return finalize(
        figure,
        dataset,
        title="Corrélation entre grandeurs",
        subtitle=dataset.label(config),
        height=max(360, 70 * len(quantities) + 120),
        show_legend=False,
        interactivity=False,
    )


def value_histograms(dataset: Dataset, config: str, quantities: Sequence[str] | None = None) -> go.Figure:
    quantities = list(quantities or dataset.quantities)
    ncols = min(3, max(len(quantities), 1))
    nrows = int(np.ceil(len(quantities) / ncols))
    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=quantities,
        vertical_spacing=0.12 if nrows > 1 else 0.08,
        horizontal_spacing=0.08,
    )
    frame = dataset.frames[config]
    color = QUALITATIVE[0]
    for index, quantity in enumerate(quantities):
        row, col = index // ncols + 1, index % ncols + 1
        values = pd.to_numeric(frame[quantity], errors="coerce")
        figure.add_trace(
            go.Histogram(
                x=values,
                marker={"color": color, "line": {"width": 0}},
                opacity=0.85,
                name=quantity,
                showlegend=False,
                hovertemplate=f"{quantity} = %{{x:.4g}}<br>%{{y}} points<extra></extra>",
            ),
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text=axis_title(quantity, dataset.units), row=row, col=col)
        figure.update_yaxes(title_text="points" if col == 1 else "", row=row, col=col)
    figure.update_annotations(font={"size": 13, "color": "#374151"})
    return finalize(
        figure,
        dataset,
        title="Distribution des valeurs le long de la série",
        subtitle=dataset.label(config),
        height=max(280, 220 * nrows + 80),
        show_legend=False,
        interactivity=False,
    )


def highlight_on_envelope(
    dataset: Dataset,
    configs: Sequence[str],
    quantity: str,
    by: str | None = None,
) -> go.Figure:
    """Faisceau de la campagne, avec une ou deux configurations en surimpression."""
    figure = dataset.envelope(quantity, by=by)
    palette = list(QUALITATIVE)
    for index, config in enumerate(configs):
        frame = dataset.frames[config]
        figure.add_trace(
            go.Scatter(
                x=frame[dataset.time],
                y=frame[quantity],
                mode="lines",
                line={"color": palette[index % len(palette)], "width": 2.8},
                name=dataset.label(config),
                meta={"label": dataset.label(config), "config": config},
                hovertemplate=(
                    f"<b>{dataset.label(config)}</b><br>"
                    f"{dataset.time} = %{{x:.4g}}<br>{quantity} = %{{y:.4g}}<extra></extra>"
                ),
            )
        )
    return figure


def percentile_ranks(
    dataset: Dataset,
    config: str,
    quantities: Sequence[str],
    metric: str = "max",
) -> pd.DataFrame:
    table = dataset.table(quantities, metric)
    ranks = table[list(quantities)].rank(pct=True)
    row = table.loc[config, quantities]
    rank_row = ranks.loc[config, quantities]
    return pd.DataFrame(
        {
            "grandeur": quantities,
            "valeur": [float(row[q]) for q in quantities],
            "percentile": [float(rank_row[q]) * 100 for q in quantities],
        }
    )


def delta_dataset(
    dataset: Dataset,
    a: str,
    b: str,
    time: np.ndarray,
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> Dataset:
    name = f"{dataset.label(a)} − {dataset.label(b)}"
    data = {dataset.time: time}
    for quantity, (ya, yb) in series.items():
        data[quantity] = ya - yb
    return load_frames(
        {name: pd.DataFrame(data)},
        {name: {"comparaison": name}},
        time=dataset.time,
        units=dataset.units,
        label="{comparaison}",
    )


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "—"
    if abs(seconds) < 1:
        return f"{seconds:.3g} s"
    if seconds < 180:
        return f"{seconds:.3g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    return f"{seconds / 3600:.3g} h"


def format_compact(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return format_number(value)


def filename_of(dataset: Dataset, config: str) -> str:
    return os.path.basename(dataset.file(config))
