"""
Sélection des k meilleures trajectoires sur plusieurs critères par étude de Pareto.

Entrées : un dossier de CSV (un CSV = une trajectoire) et 2 à 4 critères à
compromettre (opérateur statistique appliqué à une grandeur, cf.
`compare_trajectories.CRITERION_OPERATORS`). Chaque critère est à maximiser par
défaut ; préfixe-le par "-" (ex. "-mean(pression)") ou fournis `senses` pour le
minimiser.

Sorties (dans `output_dir`) :
- `classement_pareto.csv` : toutes les trajectoires avec valeurs des critères,
  rang de Pareto, distance au point idéal, distance de crowding, score pondéré,
  ordre de classement et indicateur de sélection ;
- `trajectoires_selectionnees.csv` et `tableau_selection.png` : récapitulatif
  des k trajectoires retenues ;
- `pareto_paires.png` : nuages de points critère contre critère (front de Pareto
  et sélection mis en évidence) ;
- `pareto_3d.png` : nuage 3D (uniquement pour 3 critères) ;
- `coordonnees_paralleles.png` : profil normalisé de chaque trajectoire ;
- `radar_selection.png` : radar des trajectoires sélectionnées (à partir de 3 critères).

Usage en ligne de commande (le sens se donne via --senses) :
    python selection_pareto.py mes_csv --criteria "max(temperature)" "mean(pression)" --senses max min -k 5
"""

import argparse
import glob
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from radar_plot import add_derived_columns
from compare_trajectories import (
    compute_trajectory_criteria,
    parse_criterion,
    plot_comparison_radar,
)

SELECTION_METHODS = ("ideal", "crowding", "weighted_sum")


# ---------------------------------------------------------------------------
# Objectifs et chargement
# ---------------------------------------------------------------------------

def parse_objective(criterion, sense: str | None = None) -> tuple[str, str, str]:
    """
    Normalise un objectif en (operateur, colonne, sens) avec sens dans {"max", "min"}.

    Le sens est "max" par défaut. Un critère chaîne préfixé par "-" est minimisé
    (ex. "-mean(pression)") ; `sense`, s'il est fourni, a la priorité.
    """
    implicit_sense = "max"
    if isinstance(criterion, str):
        text = criterion.strip()
        if text.startswith("-"):
            implicit_sense = "min"
            text = text[1:]
        elif text.startswith("+"):
            text = text[1:]
        criterion = text

    operator, column = parse_criterion(criterion)
    final_sense = (sense or implicit_sense).strip().lower()
    if final_sense not in ("max", "min"):
        raise ValueError(f"Sens d'optimisation '{sense}' invalide : attendu 'max' ou 'min'")
    return operator, column, final_sense


def parse_objectives(criteria: list, senses: list[str] | None = None) -> list[tuple[str, str, str]]:
    if not criteria or len(criteria) < 2:
        raise ValueError("Une étude de Pareto nécessite au moins 2 critères")
    if len(criteria) > 4:
        print(f"Attention : {len(criteria)} critères, les graphiques par paires seront nombreux.")
    if senses is not None and len(senses) != len(criteria):
        raise ValueError("`senses` doit avoir la même longueur que `criteria`")
    senses = senses or [None] * len(criteria)
    return [parse_objective(c, s) for c, s in zip(criteria, senses)]


def objective_label(operator: str, column: str) -> str:
    return f"{operator}({column})"


def load_folder_criteria(
    csv_folder: str,
    objectives: list[tuple[str, str, str]],
    pattern: str = "*.csv",
    time_column: str = "temps",
    derive_columns: bool = True,
) -> pd.DataFrame:
    """
    Calcule les critères pour chaque CSV du dossier.

    Retourne un DataFrame indexé par nom de trajectoire (nom de fichier sans
    extension), avec une colonne `fichier` puis une colonne par critère.
    """
    files = sorted(glob.glob(os.path.join(csv_folder, pattern)))
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{pattern}' dans {csv_folder}")

    criteria = [(op, col) for op, col, _ in objectives]
    rows = {}
    for file in files:
        name = Path(file).stem
        df = pd.read_csv(file)
        if derive_columns:
            df = add_derived_columns(df)
        try:
            values = compute_trajectory_criteria(df, criteria, time_column=time_column)
        except ValueError as exc:
            raise ValueError(f"{exc} (fichier : {os.path.basename(file)})") from exc
        rows[name] = {"fichier": os.path.basename(file), **values}

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "trajectoire"
    return table


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def pareto_ranks(values: np.ndarray) -> np.ndarray:
    """
    Classement par fronts de Pareto successifs (1 = non dominé) pour des
    objectifs tous orientés « plus grand = meilleur ».
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    ranks = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    current_rank = 1

    while remaining.size > 0:
        sub = values[remaining]
        # i est dominé s'il existe j >= i partout et > i quelque part.
        ge = np.all(sub[None, :, :] >= sub[:, None, :], axis=2)
        gt = np.any(sub[None, :, :] > sub[:, None, :], axis=2)
        dominated = np.any(ge & gt, axis=1)
        front = remaining[~dominated]
        ranks[front] = current_rank
        remaining = remaining[dominated]
        current_rank += 1

    return ranks


def crowding_distance(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """
    Distance de crowding (NSGA-II) calculée front par front sur des valeurs
    normalisées ; les extrémités de chaque front reçoivent +inf.
    """
    values = np.asarray(values, dtype=float)
    n, m = values.shape
    distance = np.zeros(n)
    span = values.max(axis=0) - values.min(axis=0)
    span[span == 0] = 1.0

    for rank in np.unique(ranks):
        idx = np.where(ranks == rank)[0]
        if idx.size <= 2:
            distance[idx] = np.inf
            continue
        for j in range(m):
            order = idx[np.argsort(values[idx, j])]
            distance[order[0]] = distance[order[-1]] = np.inf
            gaps = (values[order[2:], j] - values[order[:-2], j]) / span[j]
            distance[order[1:-1]] += gaps

    return distance


def normalize_to_maximize(table: pd.DataFrame, labels: list[str], senses: list[str]) -> pd.DataFrame:
    """
    Ramène chaque critère dans [0, 1] avec 1 = meilleur, quel que soit son sens.
    Un critère constant vaut 1 partout (toutes les trajectoires sont équivalentes).
    """
    normalized = pd.DataFrame(index=table.index)
    for label, sense in zip(labels, senses):
        col = table[label].astype(float)
        lo, hi = col.min(), col.max()
        if hi == lo:
            normalized[label] = 1.0
            continue
        scaled = (col - lo) / (hi - lo)
        normalized[label] = scaled if sense == "max" else 1.0 - scaled
    return normalized


def rank_trajectories(
    table: pd.DataFrame,
    objectives: list[tuple[str, str, str]],
    method: str = "ideal",
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Enrichit le tableau des critères avec le rang de Pareto, les scores de
    départage et un ordre global de classement.

    `method` départage les trajectoires d'un même front :
    - "ideal" : distance (pondérée) au point idéal normalisé, plus petite d'abord ;
    - "crowding" : distance de crowding NSGA-II, plus grande d'abord (diversité) ;
    - "weighted_sum" : somme pondérée des critères normalisés, plus grande d'abord.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Méthode '{method}' inconnue. Méthodes : {SELECTION_METHODS}")

    labels = [objective_label(op, col) for op, col, _ in objectives]
    senses = [sense for _, _, sense in objectives]

    if weights is None:
        weights = np.ones(len(labels))
    else:
        if len(weights) != len(labels):
            raise ValueError("`weights` doit avoir la même longueur que `criteria`")
        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("Les poids doivent être positifs et non tous nuls")
    weights = weights / weights.sum()

    result = table.copy()
    values = result[labels].astype(float)
    valid = ~values.isna().any(axis=1)
    if not valid.all():
        dropped = list(result.index[~valid])
        print(f"Attention : critères non calculables (NaN), trajectoires ignorées : {dropped}")
    result = result[valid].copy()

    normalized = normalize_to_maximize(result, labels, senses)
    oriented = normalized.values

    result["rang_pareto"] = pareto_ranks(oriented)
    result["distance_ideal"] = np.sqrt(np.sum(weights * (1.0 - oriented) ** 2, axis=1))
    result["crowding"] = crowding_distance(oriented, result["rang_pareto"].values)
    result["score_pondere"] = oriented @ weights

    if method == "ideal":
        order = result.sort_values(["rang_pareto", "distance_ideal"], ascending=[True, True]).index
    elif method == "crowding":
        order = result.sort_values(["rang_pareto", "crowding"], ascending=[True, False]).index
    else:
        order = result.sort_values(["rang_pareto", "score_pondere"], ascending=[True, False]).index

    result["ordre"] = pd.Series(np.arange(1, len(order) + 1), index=order)
    result = result.sort_values("ordre")
    result.attrs["labels"] = labels
    result.attrs["senses"] = senses
    result.attrs["method"] = method
    return result


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def _axis_label(label: str, sense: str) -> str:
    return f"{label} {'↑' if sense == 'max' else '↓'}"


def _scatter_layers(ax, result: pd.DataFrame, x: str, y: str, connect_front: bool):
    front = result[result["rang_pareto"] == 1]
    selected = result[result["selectionnee"]]
    others = result[~result["selectionnee"]]

    ax.scatter(others[x], others[y], s=22, color="lightgray", edgecolor="gray",
               linewidth=0.4, label="Toutes les trajectoires", zorder=1)
    front_only = front[~front["selectionnee"]]
    ax.scatter(front_only[x], front_only[y], s=36, color="tab:blue", label="Front de Pareto",
               zorder=2)
    if connect_front and len(front) > 1:
        f = front.sort_values(x)
        ax.plot(f[x], f[y], color="tab:blue", linewidth=1, alpha=0.6, zorder=2)
    ax.scatter(selected[x], selected[y], s=110, marker="*", color="tab:red",
               edgecolor="black", linewidth=0.5, label="Sélectionnées", zorder=3)
    for name, row in selected.iterrows():
        ax.annotate(f"{int(row['ordre'])}. {name}", (row[x], row[y]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, color="darkred")
    ax.grid(alpha=0.3)


def plot_pareto_pairs(result: pd.DataFrame, output_path: Path, title: str):
    """Nuage de points pour chaque paire de critères (un seul panneau pour 2 critères)."""
    labels, senses = result.attrs["labels"], result.attrs["senses"]
    pairs = list(itertools.combinations(range(len(labels)), 2))
    n = len(pairs)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.8 * n_rows), squeeze=False)
    flat = axes.ravel()
    for ax, (i, j) in zip(flat, pairs):
        _scatter_layers(ax, result, labels[i], labels[j], connect_front=(len(labels) == 2))
        ax.set_xlabel(_axis_label(labels[i], senses[i]))
        ax.set_ylabel(_axis_label(labels[j], senses[j]))
    for ax in flat[n:]:
        ax.set_visible(False)

    handles, leg_labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def plot_pareto_3d(result: pd.DataFrame, output_path: Path, title: str):
    """Nuage 3D des trois critères (front et sélection mis en évidence)."""
    labels, senses = result.attrs["labels"], result.attrs["senses"]
    x, y, z = labels
    front = result[(result["rang_pareto"] == 1) & ~result["selectionnee"]]
    selected = result[result["selectionnee"]]
    others = result[~result["selectionnee"]]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(others[x], others[y], others[z], s=18, color="lightgray", edgecolor="gray",
               linewidth=0.4, label="Toutes les trajectoires")
    ax.scatter(front[x], front[y], front[z], s=36, color="tab:blue", label="Front de Pareto")
    ax.scatter(selected[x], selected[y], selected[z], s=120, marker="*", color="tab:red",
               edgecolor="black", linewidth=0.5, label="Sélectionnées")
    for name, row in selected.iterrows():
        ax.text(row[x], row[y], row[z], f" {int(row['ordre'])}. {name}", fontsize=7, color="darkred")

    ax.set_xlabel(_axis_label(x, senses[0]), fontsize=9)
    ax.set_ylabel(_axis_label(y, senses[1]), fontsize=9)
    ax.set_zlabel(_axis_label(z, senses[2]), fontsize=9)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def plot_parallel_coordinates(result: pd.DataFrame, output_path: Path, title: str):
    """
    Coordonnées parallèles : un axe vertical par critère (normalisé, 1 = meilleur),
    une ligne par trajectoire ; la sélection est colorée, le reste en gris.
    """
    labels, senses = result.attrs["labels"], result.attrs["senses"]
    normalized = normalize_to_maximize(result, labels, senses)
    xs = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(2.6 * len(labels) + 4, 6))
    for name, row in normalized.iterrows():
        if not result.loc[name, "selectionnee"]:
            ax.plot(xs, row.values, color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)

    selected = result[result["selectionnee"]]
    for i, (name, row) in enumerate(selected.iterrows()):
        ax.plot(xs, normalized.loc[name].values, "o-", linewidth=2.2,
                color=colors[i % len(colors)], label=f"{int(row['ordre'])}. {name}", zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([_axis_label(l, s) for l, s in zip(labels, senses)], fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Critère normalisé (1 = meilleur)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="Sélection")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def save_summary_table(selected: pd.DataFrame, output_dir: Path, title: str) -> pd.DataFrame:
    """
    Construit le tableau récapitulatif des trajectoires sélectionnées
    (CSV + image PNG) et le retourne.
    """
    labels = selected.attrs["labels"]
    summary = selected[["ordre", "fichier", *labels, "rang_pareto", "distance_ideal", "score_pondere"]].copy()
    summary["ordre"] = summary["ordre"].astype(int)

    csv_path = output_dir / "trajectoires_selectionnees.csv"
    summary.to_csv(csv_path, float_format="%.6g")
    print(f"Tableau sauvegardé : {csv_path}")

    display = summary.reset_index()
    cell_text = []
    for _, row in display.iterrows():
        cells = []
        for col in display.columns:
            v = row[col]
            cells.append(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
        cell_text.append(cells)

    fig_height = 0.6 + 0.4 * (len(display) + 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(display.columns) + 1, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=list(display.columns), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbe5f1")
    ax.set_title(title, fontsize=11, pad=10)

    png_path = output_dir / "tableau_selection.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Tableau sauvegardé : {png_path}")
    return summary


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def select_trajectories_pareto(
    csv_folder: str,
    criteria: list,
    k: int,
    senses: list[str] | None = None,
    weights: list[float] | None = None,
    method: str = "ideal",
    pattern: str = "*.csv",
    time_column: str = "temps",
    output_dir: str = "selection_pareto",
    derive_columns: bool = True,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sélectionne les `k` trajectoires réalisant le meilleur compromis entre plusieurs
    critères par étude de Pareto.

    - `csv_folder` : dossier contenant un CSV par trajectoire ;
    - `criteria` : 2 à 4 critères "operateur(colonne)" (ou tuples), maximisés par
      défaut ; préfixe "-" ou `senses=["max", "min", ...]` pour minimiser ;
    - `k` : nombre de trajectoires à retenir ;
    - `method` : départage au sein d'un front ("ideal", "crowding", "weighted_sum") ;
    - `weights` : poids des critères pour "ideal" et "weighted_sum".

    Les trajectoires sont classées par front de Pareto (rang 1 = non dominées), puis
    départagées par `method` ; les `k` premières sont sélectionnées. Les graphiques
    et tableaux sont écrits dans `output_dir`.

    Retourne (tableau récapitulatif des sélectionnées, classement complet).
    """
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1")

    objectives = parse_objectives(criteria, senses)
    table = load_folder_criteria(
        csv_folder, objectives, pattern=pattern, time_column=time_column, derive_columns=derive_columns
    )
    result = rank_trajectories(table, objectives, method=method, weights=weights)

    if k > len(result):
        print(f"Attention : k={k} supérieur au nombre de trajectoires ({len(result)}), toutes retenues.")
        k = len(result)
    result["selectionnee"] = result["ordre"] <= k
    selected = result[result["selectionnee"]].copy()
    selected.attrs.update(result.attrs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "classement_pareto.csv"
    result.to_csv(ranking_path, float_format="%.6g")
    print(f"Classement sauvegardé : {ranking_path}")

    summary = save_summary_table(selected, output_dir, title=f"Top {k} — compromis de Pareto ({method})")

    if make_plots:
        n_front = int((result["rang_pareto"] == 1).sum())
        base_title = f"Étude de Pareto — {len(result)} trajectoires, front de {n_front}, top {k}"
        plot_pareto_pairs(result, output_dir / "pareto_paires.png", base_title)
        if len(objectives) == 3:
            plot_pareto_3d(result, output_dir / "pareto_3d.png", base_title)
        plot_parallel_coordinates(result, output_dir / "coordonnees_paralleles.png", base_title)
        if len(objectives) >= 3:
            plot_comparison_radar(
                selected[result.attrs["labels"]],
                title=f"Radar des {k} trajectoires sélectionnées",
                output_path=output_dir / "radar_selection.png",
            )

    return summary, result


def main():
    parser = argparse.ArgumentParser(
        description="Sélectionne les k trajectoires réalisant le meilleur compromis de Pareto entre plusieurs critères."
    )
    parser.add_argument("csv_folder", help="Dossier contenant un CSV par trajectoire")
    parser.add_argument(
        "--criteria", nargs="+", required=True,
        help="Critères 'operateur(colonne)' (2 à 4), maximisés par défaut",
    )
    parser.add_argument(
        "--senses", nargs="+", choices=("max", "min"), default=None,
        help="Sens d'optimisation de chaque critère (même ordre que --criteria), ex. --senses max min",
    )
    parser.add_argument("-k", type=int, required=True, help="Nombre de trajectoires à sélectionner")
    parser.add_argument("--method", choices=SELECTION_METHODS, default="ideal",
                        help="Départage au sein d'un front de Pareto (défaut : ideal)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Poids des critères (même ordre que --criteria)")
    parser.add_argument("--pattern", default="*.csv", help="Motif des fichiers (défaut : *.csv)")
    parser.add_argument("--time-column", default="temps", help="Colonne temporelle (défaut : temps)")
    parser.add_argument("--output-dir", default="selection_pareto", help="Dossier de sortie")
    parser.add_argument("--no-derived", action="store_true",
                        help="Ne pas calculer les grandeurs dérivées de radar_plot.add_derived_columns")
    parser.add_argument("--no-plots", action="store_true", help="Ne pas générer les graphiques")
    args = parser.parse_args()

    summary, _ = select_trajectories_pareto(
        args.csv_folder,
        criteria=args.criteria,
        k=args.k,
        senses=args.senses,
        weights=args.weights,
        method=args.method,
        pattern=args.pattern,
        time_column=args.time_column,
        output_dir=args.output_dir,
        derive_columns=not args.no_derived,
        make_plots=not args.no_plots,
    )

    print("\nTrajectoires sélectionnées :")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)


if __name__ == "__main__":
    main()
