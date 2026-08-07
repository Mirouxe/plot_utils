"""Chargement d'un lot de CSV en un jeu de configurations exploitable.

Un :class:`Dataset` associe :

- ``frames`` : une série temporelle (DataFrame) par configuration ;
- ``meta``   : une ligne par configuration décrivant ses caractéristiques,
  lues dans le nom du fichier.

Tous les graphiques de la librairie travaillent sur cet objet, ce qui évite de
réécrire la boucle « je lis mes fichiers, je décode le nom, je trace » à chaque
nouvelle question posée aux données.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import naming
from .metrics import apply_metric, metric_label, normalize_metrics

__all__ = ["Dataset", "load", "load_frames"]

TIME_CANDIDATES = ("time", "temps", "t", "instant", "timestamp", "sec", "seconds")
UNIT_IN_NAME_RE = re.compile(r"^(?P<name>.+?)\s*[\[\(](?P<unit>[^\]\)]+)[\]\)]\s*$")

META_FILE = "__file__"
META_LABEL = "__label__"
RESERVED = (META_FILE, META_LABEL)


def _clean_columns(columns: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Sépare le nom de la colonne de son unité (``"debit [kg/s]"``).

    Retourne le renommage à appliquer et les unités trouvées. En cas de collision
    de noms après nettoyage, la colonne d'origine est conservée telle quelle.
    """
    renames: dict[str, str] = {}
    units: dict[str, str] = {}
    taken = set(columns)

    for column in columns:
        match = UNIT_IN_NAME_RE.match(str(column))
        if not match:
            continue
        name, unit = match.group("name").strip(), match.group("unit").strip()
        if not name or name in taken:
            continue
        renames[column] = name
        units[name] = unit
        taken.add(name)

    return renames, units


def _resolve_files(
    source: str | Path | Iterable[str | Path],
    pattern: str = "*.csv",
    recursive: bool = False,
) -> list[Path]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            files = sorted(globber(pattern))
        else:
            files = sorted(Path(p) for p in glob.glob(str(source), recursive=recursive))
    else:
        files = [Path(item) for item in source]

    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier trouvé pour source={source!r} avec pattern={pattern!r}."
        )
    return files


def _detect_time_column(frame: pd.DataFrame) -> str:
    lowered = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in TIME_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            return column
    raise ValueError(
        "Impossible de détecter la colonne de temps ; précise-la avec time=..."
    )


def load(
    source: str | Path | Iterable[str | Path],
    pattern: str = "*.csv",
    time: str | None = None,
    template: str | None = None,
    regex: str | re.Pattern[str] | None = None,
    parser: Callable[[str], Mapping[str, Any]] | None = None,
    rename: Mapping[str, str] | None = None,
    casters: Mapping[str, Callable[[str], Any]] | None = None,
    label: str | None = None,
    separators: str = "_ ",
    strict: bool = False,
    strip_units: bool = True,
    recursive: bool = False,
    drop_constant_tags: bool = True,
    read_csv_kwargs: Mapping[str, Any] | None = None,
) -> "Dataset":
    """Charge un dossier (ou une liste) de CSV en un :class:`Dataset`.

    Args:
        source: dossier, motif glob, ou liste de chemins.
        pattern: motif appliqué quand ``source`` est un dossier.
        time: colonne de temps ; détectée automatiquement si omise.
        template: gabarit de nom de fichier, ex. ``"run_{materiau}_P{puissance}"``.
        regex: expression régulière à groupes nommés (alternative au gabarit).
        parser: fonction ``nom -> dict`` (alternative la plus libre).
        rename: renommage des caractéristiques détectées.
        casters: conversions par caractéristique, ex. ``{"puissance": float}``.
        label: gabarit d'étiquette lisible, ex. ``"{materiau} — {puissance} kW"``.
        strip_units: extrait l'unité des noms de colonnes du style ``"T [K]"``.
        drop_constant_tags: écarte les jetons non identifiés (``tag1``, ``tag2``, …)
            identiques dans tous les noms de fichiers, en général un simple préfixe.
    """
    files = _resolve_files(source, pattern=pattern, recursive=recursive)
    parse = naming.make_parser(
        template=template,
        regex=regex,
        parser=parser,
        rename=rename,
        casters=casters,
        separators=separators,
        strict=strict,
    )

    frames: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    units: dict[str, str] = {}
    time_column = time

    for path in files:
        frame = pd.read_csv(path, **dict(read_csv_kwargs or {}))
        if strip_units:
            renames, found = _clean_columns(list(frame.columns))
            if renames:
                frame = frame.rename(columns=renames)
            units.update(found)

        frame.columns = [str(col).strip() for col in frame.columns]

        if time_column is None:
            time_column = _detect_time_column(frame)
        if time_column not in frame.columns:
            raise ValueError(
                f"Colonne de temps {time_column!r} absente de {path.name} "
                f"(colonnes : {list(frame.columns)})."
            )

        config = path.stem
        if config in frames:
            config = f"{config}#{len(frames)}"
        frames[config] = frame.sort_values(time_column).reset_index(drop=True)

        characteristics = parse(path)
        for reserved in RESERVED:
            characteristics.pop(reserved, None)
        records.append({"config": config, META_FILE: str(path), **characteristics})

    meta = pd.DataFrame(records).set_index("config")
    if drop_constant_tags:
        useless = [
            column
            for column in meta.columns
            if re.fullmatch(r"tag\d+", str(column)) and meta[column].nunique() <= 1
        ]
        meta = meta.drop(columns=useless)

    dataset = Dataset(frames=frames, meta=meta, time=str(time_column), units=units)
    return dataset.sort().with_labels(label)


def load_frames(
    frames: Mapping[str, pd.DataFrame],
    meta: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
    time: str = "time",
    units: Mapping[str, str] | None = None,
    label: str | None = None,
) -> "Dataset":
    """Construit un :class:`Dataset` depuis des DataFrames déjà en mémoire."""
    if isinstance(meta, pd.DataFrame):
        meta_df = meta.copy()
    else:
        meta_df = pd.DataFrame.from_dict(dict(meta), orient="index")
    meta_df.index.name = "config"
    meta_df = meta_df.loc[[key for key in frames if key in meta_df.index]]
    dataset = Dataset(
        frames={key: frames[key] for key in meta_df.index},
        meta=meta_df,
        time=time,
        units=dict(units or {}),
    )
    return dataset.with_labels(label)


class Dataset:
    """Lot de configurations partageant la même structure de séries temporelles."""

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        meta: pd.DataFrame,
        time: str,
        units: Mapping[str, str] | None = None,
        label_template: str | None = None,
    ):
        self.frames: dict[str, pd.DataFrame] = dict(frames)
        self.meta: pd.DataFrame = meta.copy()
        self.time: str = time
        self.units: dict[str, str] = dict(units or {})
        self.label_template = label_template
        if META_LABEL not in self.meta.columns:
            self.meta[META_LABEL] = list(self.meta.index)

    # ------------------------------------------------------------------ bases

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[str]:
        return iter(self.meta.index)

    def __getitem__(self, config: str) -> pd.DataFrame:
        return self.frames[config]

    def __repr__(self) -> str:
        return (
            f"<Dataset {len(self)} configurations | temps={self.time!r} | "
            f"caractéristiques={self.characteristics} | "
            f"grandeurs={self.quantities[:6]}{'...' if len(self.quantities) > 6 else ''}>"
        )

    @property
    def configs(self) -> list[str]:
        return list(self.meta.index)

    @property
    def characteristics(self) -> list[str]:
        """Caractéristiques lues dans les noms de fichiers."""
        return [c for c in self.meta.columns if c not in RESERVED]

    @property
    def quantities(self) -> list[str]:
        """Grandeurs numériques présentes dans toutes les configurations."""
        shared: set[str] | None = None
        for frame in self.frames.values():
            numeric = {
                str(col)
                for col in frame.columns
                if pd.api.types.is_numeric_dtype(frame[col])
            }
            shared = numeric if shared is None else (shared & numeric)
        ordered = [
            str(col)
            for col in next(iter(self.frames.values())).columns
            if str(col) in (shared or set()) and str(col) != self.time
        ]
        return ordered

    def label(self, config: str) -> str:
        return str(self.meta.at[config, META_LABEL])

    def file(self, config: str) -> str:
        return str(self.meta.at[config, META_FILE]) if META_FILE in self.meta else config

    def values(self, characteristic: str) -> list[Any]:
        """Valeurs distinctes d'une caractéristique, triées naturellement."""
        from .theme import sort_values_naturally

        if characteristic not in self.meta.columns:
            raise KeyError(
                f"Caractéristique {characteristic!r} inconnue. "
                f"Disponibles : {self.characteristics}"
            )
        return sort_values_naturally(self.meta[characteristic].tolist())

    def characteristics_of(self, config: str) -> dict[str, Any]:
        row = self.meta.loc[config]
        return {key: row[key] for key in self.characteristics}

    # ------------------------------------------------------- transformations

    def _clone(
        self,
        frames: Mapping[str, pd.DataFrame] | None = None,
        meta: pd.DataFrame | None = None,
        units: Mapping[str, str] | None = None,
    ) -> "Dataset":
        return Dataset(
            frames=self.frames if frames is None else frames,
            meta=self.meta if meta is None else meta,
            time=self.time,
            units=self.units if units is None else units,
            label_template=self.label_template,
        )

    def with_labels(self, template: str | None = None, keys: Sequence[str] | None = None) -> "Dataset":
        """Recalcule les étiquettes lisibles des configurations."""
        keys = list(keys) if keys is not None else self.characteristics
        meta = self.meta.copy()
        labels = []
        for config in meta.index:
            characteristics = {key: meta.at[config, key] for key in keys if key in meta}
            if not characteristics:
                labels.append(config)
                continue
            labels.append(
                naming.build_label(characteristics, keys=keys, template=template)
            )
        meta[META_LABEL] = labels
        clone = self._clone(meta=meta)
        clone.label_template = template
        return clone

    def filter(self, **conditions: Any) -> "Dataset":
        """Sélectionne les configurations par caractéristique.

        Une condition accepte une valeur, une liste de valeurs ou un prédicat :

        >>> ds.filter(maillage="fin", puissance=[10, 25], debit=lambda v: v > 1.0)
        """
        mask = pd.Series(True, index=self.meta.index)
        for key, condition in conditions.items():
            if key not in self.meta.columns:
                raise KeyError(
                    f"Caractéristique {key!r} inconnue. Disponibles : {self.characteristics}"
                )
            column = self.meta[key]
            if callable(condition):
                mask &= column.map(condition).astype(bool)
            elif isinstance(condition, (list, tuple, set, np.ndarray, pd.Series)):
                mask &= column.isin(list(condition))
            else:
                mask &= column == condition
        meta = self.meta[mask]
        if meta.empty:
            raise ValueError(f"Aucune configuration ne satisfait {conditions}.")
        return self._clone(frames={c: self.frames[c] for c in meta.index}, meta=meta)

    def select(self, configs: Sequence[str]) -> "Dataset":
        """Restreint le jeu à une liste de configurations."""
        missing = [c for c in configs if c not in self.frames]
        if missing:
            raise KeyError(f"Configurations inconnues : {missing}")
        meta = self.meta.loc[list(configs)]
        return self._clone(frames={c: self.frames[c] for c in configs}, meta=meta)

    def sort(self, by: str | Sequence[str] | None = None, ascending: bool = True) -> "Dataset":
        """Ordonne les configurations selon une ou plusieurs caractéristiques.

        L'ordre est « naturel » : les valeurs numériques sont comparées comme des
        nombres, pas comme du texte (``5`` avant ``12``, et non l'inverse).
        """
        from .theme import sort_values_naturally

        keys = self.characteristics if by is None else ([by] if isinstance(by, str) else list(by))
        if not keys:
            return self

        def rank(column: pd.Series) -> pd.Series:
            order = sort_values_naturally(column.tolist())
            positions = {value: index for index, value in enumerate(order)}
            return column.map(positions)

        meta = self.meta.sort_values(keys, ascending=ascending, key=rank)
        return self._clone(frames={c: self.frames[c] for c in meta.index}, meta=meta)

    def derive(
        self,
        units: Mapping[str, str] | None = None,
        **columns: str | Callable[[pd.DataFrame], Any],
    ) -> "Dataset":
        """Ajoute des grandeurs calculées dans chaque configuration.

        La valeur est soit une expression ``pandas.eval`` (chaîne), soit une
        fonction du DataFrame :

        >>> ds.derive(puissance_utile="debit * delta_T", ecart=lambda df: df.T_out - df.T_in)
        """
        frames = {}
        for config, frame in self.frames.items():
            frame = frame.copy()
            for name, recipe in columns.items():
                frame[name] = recipe(frame) if callable(recipe) else frame.eval(recipe)
            frames[config] = frame
        merged_units = {**self.units, **dict(units or {})}
        return self._clone(frames=frames, units=merged_units)

    def add_characteristic(
        self, **columns: Callable[[pd.Series], Any] | Sequence[Any] | Any
    ) -> "Dataset":
        """Ajoute une caractéristique dérivée des autres (ou d'un calcul métier).

        >>> ds.add_characteristic(regime=lambda row: "fort" if row.puissance > 20 else "faible")
        """
        meta = self.meta.copy()
        for name, recipe in columns.items():
            if callable(recipe):
                meta[name] = [recipe(meta.loc[config]) for config in meta.index]
            else:
                meta[name] = recipe
        return self._clone(meta=meta)

    def rename_characteristics(self, **mapping: str) -> "Dataset":
        return self._clone(meta=self.meta.rename(columns=mapping))

    # ---------------------------------------------------------- restitutions

    def long(
        self,
        quantities: Sequence[str] | None = None,
        include_characteristics: bool = True,
    ) -> pd.DataFrame:
        """Format long (``config``, caractéristiques, temps, ``quantity``, ``value``)."""
        quantities = list(quantities or self.quantities)
        pieces = []
        for config, frame in self.frames.items():
            available = [q for q in quantities if q in frame.columns]
            melted = frame[[self.time, *available]].melt(
                id_vars=[self.time], var_name="quantity", value_name="value"
            )
            melted.insert(0, "config", config)
            melted.insert(1, "label", self.label(config))
            if include_characteristics:
                for key, value in self.characteristics_of(config).items():
                    melted[key] = value
            pieces.append(melted)
        return pd.concat(pieces, ignore_index=True)

    def wide(self, quantities: Sequence[str] | None = None) -> pd.DataFrame:
        """Concaténation de toutes les configurations avec leurs caractéristiques."""
        quantities = list(quantities or self.quantities)
        pieces = []
        for config, frame in self.frames.items():
            available = [q for q in quantities if q in frame.columns]
            piece = frame[[self.time, *available]].copy()
            piece.insert(0, "config", config)
            piece.insert(1, "label", self.label(config))
            for key, value in self.characteristics_of(config).items():
                piece[key] = value
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    def table(
        self,
        quantities: Sequence[str] | str | None = None,
        metrics: Any = "max",
        flat: bool = True,
    ) -> pd.DataFrame:
        """Tableau des métriques scalaires par configuration.

        C'est la base de tous les graphiques de comparaison : une ligne par
        configuration, ses caractéristiques, et une colonne par couple
        (grandeur, métrique).

        >>> ds.table(["temperature", "pression"], ["max", "mean"])
        """
        if isinstance(quantities, str):
            quantities = [quantities]
        quantities = list(quantities or self.quantities)
        metric_map = normalize_metrics(metrics)

        rows = []
        for config in self.meta.index:
            frame = self.frames[config]
            row: dict[str, Any] = {
                "config": config,
                "label": self.label(config),
                **self.characteristics_of(config),
            }
            for quantity in quantities:
                if quantity not in frame.columns:
                    continue
                for metric_name, metric in metric_map.items():
                    key = quantity if (flat and len(metric_map) == 1) else f"{quantity}_{metric_name}"
                    row[key] = apply_metric(frame, quantity, metric, self.time)
            rows.append(row)

        table = pd.DataFrame(rows).set_index("config")
        table.attrs["metrics"] = {name: metric_label(m) for name, m in metric_map.items()}
        table.attrs["quantities"] = quantities
        return table

    def common_grid(self, points: int = 400) -> np.ndarray:
        """Grille temporelle commune (intersection des plages de toutes les configs)."""
        starts, ends = [], []
        for frame in self.frames.values():
            series = pd.to_numeric(frame[self.time], errors="coerce").dropna()
            starts.append(series.min())
            ends.append(series.max())
        start, end = max(starts), min(ends)
        if not np.isfinite(start) or not np.isfinite(end) or start >= end:
            start, end = min(starts), max(ends)
        return np.linspace(start, end, points)

    def resample(self, grid: Sequence[float] | int = 400) -> "Dataset":
        """Ré-échantillonne toutes les configurations sur une grille commune."""
        axis = self.common_grid(grid) if isinstance(grid, int) else np.asarray(grid, dtype=float)
        quantities = self.quantities
        frames = {}
        for config, frame in self.frames.items():
            time_values = pd.to_numeric(frame[self.time], errors="coerce").to_numpy(dtype=float)
            data = {self.time: axis}
            for quantity in quantities:
                values = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(time_values) & np.isfinite(values)
                data[quantity] = np.interp(
                    axis, time_values[mask], values[mask], left=np.nan, right=np.nan
                )
            frames[config] = pd.DataFrame(data)
        return self._clone(frames=frames)

    def overview(self) -> str:
        """Résumé textuel : configurations, caractéristiques et grandeurs disponibles."""
        lines = [
            f"{len(self)} configurations · colonne de temps : {self.time!r}",
            "",
            "Caractéristiques :",
        ]
        for characteristic in self.characteristics:
            values = self.values(characteristic)
            shown = ", ".join(str(v) for v in values[:8])
            more = f" … (+{len(values) - 8})" if len(values) > 8 else ""
            lines.append(f"  - {characteristic} ({len(values)}) : {shown}{more}")
        lines += ["", "Grandeurs :"]
        for quantity in self.quantities:
            unit = self.units.get(quantity)
            lines.append(f"  - {quantity}{f' [{unit}]' if unit else ''}")
        return "\n".join(lines)

    def hover_text(self, config: str, extra: Mapping[str, Any] | None = None) -> str:
        """Bloc de survol commun : étiquette, fichier et caractéristiques."""
        parts = [f"<b>{self.label(config)}</b>"]
        for key, value in self.characteristics_of(config).items():
            parts.append(f"{key} : {value}")
        for key, value in (extra or {}).items():
            parts.append(f"{key} : {value}")
        parts.append(
            f"<span style='color:#9ca3af'>{os.path.basename(self.file(config))}</span>"
        )
        return "<br>".join(parts)

    # ------------------------------------------------------------ graphiques

    def curves(self, *args, **kwargs):
        """Superposition des courbes de toutes les configurations (voir :func:`csvscope.plots.curves`)."""
        from .plots import curves

        return curves(self, *args, **kwargs)

    def grid(self, *args, **kwargs):
        """Plusieurs grandeurs en sous-graphiques synchronisés."""
        from .plots import grid

        return grid(self, *args, **kwargs)

    def explorer(self, *args, **kwargs):
        """Graphique unique avec menus pour changer de grandeur et de couleur."""
        from .plots import explorer

        return explorer(self, *args, **kwargs)

    def envelope(self, *args, **kwargs):
        """Faisceau min/moyenne/max par groupe de configurations."""
        from .plots import envelope

        return envelope(self, *args, **kwargs)

    def small_multiples(self, *args, **kwargs):
        """Une facette par valeur de caractéristique."""
        from .plots import small_multiples

        return small_multiples(self, *args, **kwargs)

    def compare(self, *args, **kwargs):
        """Métrique scalaire en fonction d'une caractéristique."""
        from .plots import compare

        return compare(self, *args, **kwargs)

    def bars(self, *args, **kwargs):
        """Classement des configurations sur une métrique."""
        from .plots import bars

        return bars(self, *args, **kwargs)

    def heatmap(self, *args, **kwargs):
        """Carte d'une métrique sur le croisement de deux caractéristiques."""
        from .plots import heatmap

        return heatmap(self, *args, **kwargs)

    def distribution(self, *args, **kwargs):
        """Distribution d'une métrique par groupe (boîte, violon, points, ECDF)."""
        from .plots import distribution

        return distribution(self, *args, **kwargs)

    def parallel(self, *args, **kwargs):
        """Coordonnées parallèles multi-grandeurs avec sélection interactive."""
        from .plots import parallel

        return parallel(self, *args, **kwargs)

    def radar(self, *args, **kwargs):
        """Radar comparatif de plusieurs grandeurs normalisées."""
        from .plots import radar

        return radar(self, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        """Nuage de points entre deux métriques (compromis entre grandeurs)."""
        from .plots import scatter

        return scatter(self, *args, **kwargs)

    def report(self, *args, **kwargs):
        """Rapport HTML multi-onglets rassemblant plusieurs figures."""
        from .report import report

        return report(self, *args, **kwargs)
