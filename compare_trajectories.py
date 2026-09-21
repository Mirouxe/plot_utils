"""
Comparaison de plusieurs trajectoires stockées dans des fichiers CSV.

- `compare_trajectories` : compare des critères scalaires (opérateur statistique
  appliqué à une grandeur) sur un graphique radar.
- `compare_time_series` : superpose les séries temporelles complètes de plusieurs
  trajectoires, un visuel par grandeur.

Chaque trajectoire correspond à un CSV dont le nom de fichier contient le nom
de la trajectoire.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from radar_plot import add_derived_columns


# ---------------------------------------------------------------------------
# Chargement des trajectoires
# ---------------------------------------------------------------------------

def find_trajectory_file(trajectory_name: str, csv_folder: str = ".", pattern: str = "*.csv") -> str:
    """
    Retourne l'unique CSV du dossier dont le nom contient `trajectory_name`.

    Si plusieurs fichiers correspondent, celui dont le nom (sans extension)
    est exactement `trajectory_name` est retenu ; sinon une erreur est levée.
    """
    files = sorted(glob.glob(os.path.join(csv_folder, pattern)))
    candidates = [f for f in files if trajectory_name in os.path.basename(f)]

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier '{pattern}' contenant '{trajectory_name}' dans {csv_folder}"
        )
    if len(candidates) == 1:
        return candidates[0]

    exact = [f for f in candidates if Path(f).stem == trajectory_name]
    if len(exact) == 1:
        return exact[0]

    raise ValueError(
        f"Plusieurs fichiers correspondent à la trajectoire '{trajectory_name}' : "
        f"{[os.path.basename(f) for f in candidates]}"
    )


def load_trajectories(
    trajectory_names: list[str],
    csv_folder: str = ".",
    pattern: str = "*.csv",
    derive_columns: bool = True,
) -> dict[str, pd.DataFrame]:
    """Charge le CSV de chaque trajectoire et retourne {nom: DataFrame}."""
    if not trajectory_names:
        raise ValueError("Aucune trajectoire à comparer")

    trajectories = {}
    for name in trajectory_names:
        file = find_trajectory_file(name, csv_folder=csv_folder, pattern=pattern)
        df = pd.read_csv(file)
        if derive_columns:
            df = add_derived_columns(df)
        df.attrs["source_file"] = os.path.basename(file)
        trajectories[name] = df
    return trajectories


# ---------------------------------------------------------------------------
# Critères scalaires et radar
# ---------------------------------------------------------------------------

def _integrate_over_time(df: pd.DataFrame, column: str, time_column: str) -> float:
    if time_column not in df.columns:
        raise ValueError(
            f"Colonne temporelle '{time_column}' introuvable : impossible de calculer "
            f"l'intégrale temporelle de '{column}'"
        )
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    order = np.argsort(df[time_column].values)
    t = df[time_column].values[order].astype(float)
    y = df[column].values[order].astype(float)
    return float(trapezoid(y, t))


# Opérateurs applicables à une grandeur (colonne) d'un CSV de trajectoire.
# Chaque opérateur reçoit (df, column, time_column) et renvoie un scalaire.
CRITERION_OPERATORS = {
    "mean": lambda df, col, t: float(df[col].mean()),
    "max": lambda df, col, t: float(df[col].max()),
    "min": lambda df, col, t: float(df[col].min()),
    "abs_max": lambda df, col, t: float(df[col].abs().max()),
    "median": lambda df, col, t: float(df[col].median()),
    "std": lambda df, col, t: float(df[col].std()),
    "range": lambda df, col, t: float(df[col].max() - df[col].min()),
    "rms": lambda df, col, t: float(np.sqrt(np.mean(np.square(df[col].values.astype(float))))),
    "final": lambda df, col, t: float(df[col].iloc[-1]),
    "integral": _integrate_over_time,
}


def parse_criterion(criterion) -> tuple[str, str]:
    """
    Normalise un critère en tuple (operateur, colonne).

    Formes acceptées : ("max", "temperature"), "max(temperature)" ou "max:temperature".
    """
    if isinstance(criterion, (tuple, list)):
        if len(criterion) != 2:
            raise ValueError(f"Un critère doit être un couple (operateur, colonne) : {criterion}")
        operator, column = criterion
    elif isinstance(criterion, str):
        text = criterion.strip()
        if text.endswith(")") and "(" in text:
            operator, column = text[:-1].split("(", 1)
        elif ":" in text:
            operator, column = text.split(":", 1)
        else:
            raise ValueError(
                f"Critère '{criterion}' invalide : attendu 'operateur(colonne)' ou 'operateur:colonne'"
            )
    else:
        raise TypeError(f"Type de critère non supporté : {type(criterion).__name__}")

    operator = operator.strip().lower()
    column = column.strip()
    if operator not in CRITERION_OPERATORS:
        raise ValueError(
            f"Opérateur '{operator}' inconnu. Opérateurs disponibles : {sorted(CRITERION_OPERATORS)}"
        )
    if not column:
        raise ValueError(f"Nom de colonne vide dans le critère : {criterion}")
    return operator, column


def compute_trajectory_criteria(
    df: pd.DataFrame,
    criteria: list[tuple[str, str]],
    time_column: str = "temps",
) -> dict[str, float]:
    """Applique chaque critère (operateur, colonne) au DataFrame d'une trajectoire."""
    values = {}
    for operator, column in criteria:
        if column not in df.columns:
            raise ValueError(f"Colonne '{column}' introuvable dans la trajectoire")
        values[f"{operator}({column})"] = CRITERION_OPERATORS[operator](df, column, time_column)
    return values


def _nice_step(raw_step: float) -> float:
    """Arrondit un pas de graduation au nombre « rond » supérieur (1, 2, 2.5, 5 × 10^k)."""
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / 10 ** exponent
    for nice in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice * (1 + 1e-9):
            return float(nice * 10 ** exponent)
    return float(10 ** (exponent + 1))


def nice_axis_bounds(
    vmin: float, vmax: float, n_levels: int, include_zero: bool = True
) -> tuple[float, float]:
    """
    Calcule des bornes « rondes » (lo, hi) encadrant [vmin, vmax] avec exactement
    `n_levels` intervalles réguliers.

    Avec `include_zero=True`, le zéro est toujours dans [lo, hi] : le centre du radar
    vaut 0 pour les grandeurs positives, ce qui rend les surfaces comparables.
    Avec `include_zero=False`, l'échelle est resserrée sur les valeurs observées.
    """
    lo_target = min(0.0, vmin) if include_zero else vmin
    hi_target = max(0.0, vmax) if include_zero else vmax
    span = hi_target - lo_target
    if span == 0:
        span = abs(vmax) if vmax != 0 else 1.0

    step = _nice_step(span / n_levels)
    while True:
        lo = np.floor(lo_target / step + 1e-9) * step
        hi = lo + n_levels * step
        if hi >= hi_target - 1e-9 * step:
            return float(lo), float(hi)
        step = _nice_step(step * 1.01)


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.4g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text


def _polar_label_alignment(angle: float) -> tuple[str, str]:
    """Alignement d'un texte placé à l'extérieur du radar selon l'angle (en radians)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ha = "center" if abs(cos_a) < 0.2 else ("left" if cos_a > 0 else "right")
    va = "center" if abs(sin_a) < 0.2 else ("bottom" if sin_a > 0 else "top")
    return ha, va


def plot_comparison_radar(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    normalize: bool = True,
    n_levels: int = 5,
    include_zero: bool = True,
):
    """
    Trace un radar multi-trajectoires : une ligne du tableau = une trajectoire,
    une colonne = un critère (un axe du radar).

    Avec `normalize=True`, chaque axe possède sa propre échelle : la géométrie est
    ramenée à [0, 1] pour que toutes les grandeurs soient visibles, mais les
    graduations affichées le long de chaque axe sont les vraies valeurs du critère.
    """
    labels = list(table.columns)
    if len(labels) < 3:
        print("Attention : un radar est plus lisible avec au moins 3 critères.")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    axis_titles = list(labels)

    if normalize:
        bounds = {
            label: nice_axis_bounds(table[label].min(), table[label].max(), n_levels, include_zero)
            for label in labels
        }
        lo = pd.Series({label: b[0] for label, b in bounds.items()})
        hi = pd.Series({label: b[1] for label, b in bounds.items()})
        plotted = (table - lo) / (hi - lo)

        fractions = np.linspace(0, 1, n_levels + 1)
        ax.set_ylim(0, 1)
        ax.set_yticks(fractions)
        ax.set_yticklabels([])

        # Graduations réelles le long de chaque axe. La valeur du centre (commune à
        # tous les axes en position) est reportée dans le titre de l'axe si non nulle.
        for k, (angle, label) in enumerate(zip(angles, labels)):
            if lo[label] != 0:
                axis_titles[k] = f"{label}\n(centre = {_format_tick(lo[label])})"
            for frac in fractions[1:]:
                value = lo[label] + frac * (hi[label] - lo[label])
                ax.text(
                    angle, frac, _format_tick(value),
                    fontsize=7, color="dimgray", ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                    zorder=5,
                )
        label_radius = 1.12
    else:
        plotted = table
        label_radius = None

    for i, (name, row) in enumerate(plotted.iterrows()):
        values = row.values.astype(float)
        values_closed = np.concatenate((values, [values[0]]))
        color = colors[i % len(colors)]
        ax.plot(angles_closed, values_closed, "o-", linewidth=2, color=color, label=str(name))
        ax.fill(angles_closed, values_closed, alpha=0.12, color=color)

    ax.set_xticks(angles)
    if label_radius is None:
        ax.set_xticklabels(axis_titles, fontsize=9)
    else:
        # Titres d'axes placés manuellement pour ne pas chevaucher la graduation extérieure.
        ax.set_xticklabels([])
        for angle, axis_title in zip(angles, axis_titles):
            ha, va = _polar_label_alignment(angle)
            ax.text(angle, label_radius, axis_title, fontsize=9, ha=ha, va=va)
    ax.set_title(title, pad=25)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def compare_trajectories(
    trajectory_names: list[str],
    criteria: list,
    csv_folder: str = ".",
    pattern: str = "*.csv",
    time_column: str = "temps",
    output_path: str = "radar_comparaison_trajectoires.png",
    title: str = "Comparaison des trajectoires",
    normalize: bool = True,
    n_levels: int = 5,
    include_zero: bool = True,
    derive_columns: bool = True,
) -> pd.DataFrame:
    """
    Compare plusieurs trajectoires sur un ensemble de critères et trace un radar.

    Pour chaque nom de trajectoire, le CSV du dossier dont le nom contient ce nom
    est chargé, puis chaque critère (opérateur statistique appliqué à une grandeur,
    ex. ("max", "temperature"), "mean(pression)", "integral:vitesse") est calculé.
    Le radar comparatif est sauvegardé dans `output_path`.

    Opérateurs disponibles : voir `CRITERION_OPERATORS` (mean, max, min, abs_max,
    median, std, range, rms, final, integral). L'intégrale temporelle utilise
    la méthode des trapèzes sur la colonne `time_column`.

    Avec `normalize=True` (défaut), chaque axe du radar a sa propre échelle réelle,
    graduée en `n_levels` niveaux ; la géométrie est normalisée pour que toutes les
    grandeurs restent visibles. `include_zero=True` force le zéro dans chaque échelle
    (centre = 0 pour les grandeurs positives) ; `include_zero=False` resserre chaque
    échelle sur les valeurs observées. Avec `normalize=False`, toutes les grandeurs
    partagent un même axe radial en valeurs brutes.

    Retourne un DataFrame (lignes = trajectoires, colonnes = critères) des valeurs brutes.
    """
    if not criteria:
        raise ValueError("Aucun critère à calculer")

    parsed_criteria = [parse_criterion(c) for c in criteria]
    trajectories = load_trajectories(
        trajectory_names, csv_folder=csv_folder, pattern=pattern, derive_columns=derive_columns
    )

    rows = {}
    for name, df in trajectories.items():
        try:
            rows[name] = compute_trajectory_criteria(df, parsed_criteria, time_column=time_column)
        except ValueError as exc:
            raise ValueError(f"{exc} (fichier : {df.attrs['source_file']})") from exc

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "trajectoire"

    plot_comparison_radar(
        table,
        title=title,
        output_path=Path(output_path),
        normalize=normalize,
        n_levels=n_levels,
        include_zero=include_zero,
    )
    return table


# ---------------------------------------------------------------------------
# Séries temporelles complètes
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def plot_time_series_comparison(
    trajectories: dict[str, pd.DataFrame],
    quantity: str,
    time_column: str,
    output_path: Path,
    title: str | None = None,
    line_width: float = 1.8,
):
    """Superpose la grandeur `quantity` de chaque trajectoire en fonction du temps."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, df) in enumerate(trajectories.items()):
        order = np.argsort(df[time_column].values)
        ax.plot(
            df[time_column].values[order],
            df[quantity].values[order],
            linewidth=line_width,
            color=colors[i % len(colors)],
            label=str(name),
        )

    ax.set_xlabel(time_column)
    ax.set_ylabel(quantity)
    ax.set_title(title or f"Comparaison de '{quantity}' entre trajectoires")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def compare_time_series(
    trajectory_names: list[str],
    quantities: list[str],
    csv_folder: str = ".",
    pattern: str = "*.csv",
    time_column: str = "temps",
    output_dir: str = "comparaison_series",
    image_format: str = "png",
    derive_columns: bool = True,
) -> dict[str, str]:
    """
    Compare les séries temporelles complètes de plusieurs trajectoires.

    Pour chaque nom de trajectoire, le CSV du dossier dont le nom contient ce nom
    est chargé. Pour chaque grandeur de `quantities`, un visuel superposant la
    série temporelle de toutes les trajectoires est sauvegardé dans `output_dir`
    (un fichier `<grandeur>.<image_format>` par grandeur).

    Retourne un dictionnaire {grandeur: chemin de la figure}.
    """
    if not quantities:
        raise ValueError("Aucune grandeur à comparer")

    trajectories = load_trajectories(
        trajectory_names, csv_folder=csv_folder, pattern=pattern, derive_columns=derive_columns
    )

    for name, df in trajectories.items():
        missing = [c for c in [time_column, *quantities] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes introuvables dans la trajectoire '{name}' "
                f"(fichier : {df.attrs['source_file']}) : {missing}"
            )

    output_dir = Path(output_dir)
    figures = {}
    for quantity in quantities:
        output_path = output_dir / f"{_safe_filename(quantity)}.{image_format}"
        plot_time_series_comparison(trajectories, quantity, time_column, output_path)
        figures[quantity] = str(output_path)

    return figures
