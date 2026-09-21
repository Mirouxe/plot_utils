import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exemple d'enrichissement du DataFrame.
    Adapte ou complète cette fonction selon tes besoins métier.
    """
    df = df.copy()

    if {"pression", "vitesse"}.issubset(df.columns):
        df["energie_proxy"] = df["pression"] * df["vitesse"]

    if {"contrainte", "deformation"}.issubset(df.columns):
        df["ratio_contrainte_deformation"] = safe_divide(
            df["contrainte"].values,
            df["deformation"].values,
        )

    if {"Ux", "Uy", "Uz"}.issubset(df.columns):
        df["norme_vitesse"] = np.sqrt(df["Ux"] ** 2 + df["Uy"] ** 2 + df["Uz"] ** 2)

    return df


def compute_maxima(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes introuvables dans le fichier : {missing}")
    return df[columns].max(numeric_only=True)


def plot_radar(max_values: pd.Series, title: str, output_path: Path | None = None):
    labels = list(max_values.index)
    values = max_values.values.astype(float)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values_closed = np.concatenate((values, [values[0]]))
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="tab:blue")
    ax.fill(angles_closed, values_closed, alpha=0.20, color="tab:blue")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(True)

    for angle, value in zip(angles, values):
        ax.text(angle, value, f" {value:.3g}", fontsize=9)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Figure sauvegardée : {output_path}")
    else:
        plt.show()


def plot_radar_from_df(
    df: pd.DataFrame,
    columns: list[str],
    output_path: str = "radar_maxima.png",
    title: str = "Radar des maxima",
) -> pd.Series:
    """
    Calcule les maxima des colonnes demandées dans un DataFrame,
    trace le radar associé, sauvegarde la figure et retourne les maxima.
    """
    max_values = compute_maxima(df, columns)
    plot_radar(max_values, title=title, output_path=Path(output_path))
    return max_values


def analyze_max_distribution(csv_folder, column_name, pattern="*.csv", plot=True):
    """
    Lit un ensemble de CSV, calcule le max d'une colonne pour chaque fichier,
    trace la distribution et retourne les statistiques principales.

    Remarque : sous hypothèse gaussienne, la règle usuelle est ~68% dans ±1σ
    et ~95% dans ±2σ autour de la moyenne.
    """
    files = glob.glob(os.path.join(csv_folder, pattern))
    if not files:
        raise ValueError(f"Aucun fichier trouvé dans {csv_folder} avec le pattern {pattern}")

    max_data = []
    for file in files:
        df = pd.read_csv(file)

        if column_name not in df.columns:
            raise ValueError(f"Colonne '{column_name}' absente dans {file}")

        max_val = df[column_name].max()
        max_data.append({
            "file": os.path.basename(file),
            "max_value": max_val,
        })

    res_df = pd.DataFrame(max_data)
    x = res_df["max_value"].values

    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    min_x = np.min(x)
    max_x = np.max(x)
    median_x = np.median(x)

    q05 = np.quantile(x, 0.05)
    q50 = np.quantile(x, 0.50)
    q66 = np.quantile(x, 0.66)
    q95 = np.quantile(x, 0.95)

    lower_bound = mean_x + std_x
    upper_bound = mean_x + 2 * std_x

    tranche_df = res_df[
        (res_df["max_value"] >= lower_bound) &
        (res_df["max_value"] <= upper_bound)
    ]

    if len(tranche_df) > 0:
        highest_point_in_slice = tranche_df.loc[tranche_df["max_value"].idxmax()]
    else:
        highest_point_in_slice = None

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(x)), x, color="steelblue", label="Max par configuration")
        plt.axhline(mean_x, color="green", linestyle="-", label=f"Moyenne = {mean_x:.3g}")
        plt.axhline(mean_x + std_x, color="orange", linestyle="--", label=f"+1σ = {lower_bound:.3g}")
        plt.axhline(mean_x + 2 * std_x, color="red", linestyle="--", label=f"+2σ = {upper_bound:.3g}")
        plt.axhline(q66, color="purple", linestyle=":", label=f"Q66 = {q66:.3g}")
        plt.axhline(q95, color="brown", linestyle=":", label=f"Q95 = {q95:.3g}")
        plt.xlabel("Configuration")
        plt.ylabel(f"Max {column_name}")
        plt.title(f"Distribution des maxima de '{column_name}'")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=15, color="lightblue", edgecolor="black", alpha=0.7)
        plt.axvline(mean_x, color="green", linestyle="-", label=f"Moyenne = {mean_x:.3g}")
        plt.axvline(lower_bound, color="orange", linestyle="--", label=f"+1σ = {lower_bound:.3g}")
        plt.axvline(upper_bound, color="red", linestyle="--", label=f"+2σ = {upper_bound:.3g}")
        plt.axvline(q66, color="purple", linestyle=":", label=f"Q66 = {q66:.3g}")
        plt.axvline(q95, color="brown", linestyle=":", label=f"Q95 = {q95:.3g}")
        plt.xlabel(f"Max {column_name}")
        plt.ylabel("Fréquence")
        plt.title("Histogramme des maxima")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    results = {
        "data": res_df,
        "mean": mean_x,
        "std": std_x,
        "min": min_x,
        "max": max_x,
        "median": median_x,
        "q05": q05,
        "q50": q50,
        "q66": q66,
        "q95": q95,
        "sigma_lower": lower_bound,
        "sigma_upper": upper_bound,
        "slice_data": tranche_df,
        "highest_point_between_1sigma_2sigma": highest_point_in_slice,
    }

    return results


def plot_csv_curves_interactive(
    folder_path,
    x_column,
    y_column,
    pattern="*.csv",
    opacity=0.35,
    line_width=2,
    highlight_opacity=1.0,
    highlight_line_width=4,
    output_html=None,
):
    """
    Lit tous les CSV d'un dossier et trace y_column en fonction de x_column
    sur un graphique interactif Plotly avec transparence des courbes.

    Si output_html est fourni, le HTML généré ajoute une mise en surbrillance
    de la courbe survolée ou cliquée.
    """
    files = glob.glob(os.path.join(folder_path, pattern))
    if not files:
        raise ValueError(f"Aucun fichier CSV trouvé dans {folder_path}")

    fig = go.Figure()

    for file in files:
        df = pd.read_csv(file)

        if x_column not in df.columns:
            raise ValueError(f"Colonne '{x_column}' absente dans {file}")
        if y_column not in df.columns:
            raise ValueError(f"Colonne '{y_column}' absente dans {file}")

        csv_name = os.path.basename(file)

        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df[y_column],
            mode="lines",
            name=csv_name,
            opacity=opacity,
            line={"width": line_width},
            hovertemplate=(
                f"<b>{csv_name}</b><br>"
                + f"{x_column} = %{{x}}<br>"
                + f"{y_column} = %{{y}}<br>"
                + "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{y_column} en fonction de {x_column}",
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode="closest",
        template="plotly_white",
    )

    if output_html:
        div_id = "plot_utils_interactive_curves"
        post_script = f"""
const gd = document.getElementById('{div_id}');
const baseOpacity = {opacity};
const baseWidth = {line_width};
const highlightOpacity = {highlight_opacity};
const highlightWidth = {highlight_line_width};
let lockedCurve = null;

function restyleCurve(curveNumber) {{
  const opacities = gd.data.map((_, i) => (i === curveNumber ? highlightOpacity : baseOpacity));
  const widths = gd.data.map((_, i) => (i === curveNumber ? highlightWidth : baseWidth));
  Plotly.restyle(gd, {{opacity: opacities, 'line.width': widths}});
}}

function resetCurves() {{
  const opacities = gd.data.map(() => baseOpacity);
  const widths = gd.data.map(() => baseWidth);
  Plotly.restyle(gd, {{opacity: opacities, 'line.width': widths}});
}}

gd.on('plotly_hover', function(evt) {{
  if (lockedCurve === null && evt.points && evt.points.length > 0) {{
    restyleCurve(evt.points[0].curveNumber);
  }}
}});

gd.on('plotly_unhover', function() {{
  if (lockedCurve === null) {{
    resetCurves();
  }}
}});

gd.on('plotly_click', function(evt) {{
  if (evt.points && evt.points.length > 0) {{
    const curve = evt.points[0].curveNumber;
    if (lockedCurve === curve) {{
      lockedCurve = null;
      resetCurves();
    }} else {{
      lockedCurve = curve;
      restyleCurve(curve);
    }}
  }}
}});
"""
        fig.write_html(output_html, div_id=div_id, post_script=post_script)
        print(f"Figure interactive sauvegardée : {output_html}")

    fig.show()
    return fig


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


def plot_comparison_radar(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    normalize: bool = True,
):
    """
    Trace un radar multi-trajectoires : une ligne du tableau = une trajectoire,
    une colonne = un critère (un axe du radar).
    """
    labels = list(table.columns)
    if len(labels) < 3:
        print("Attention : un radar est plus lisible avec au moins 3 critères.")

    if normalize:
        # Chaque axe est ramené à [-1, 1] par rapport au maximum absolu entre trajectoires,
        # sinon les critères d'unités différentes ne sont pas comparables visuellement.
        ref = table.abs().max(axis=0).replace(0, np.nan)
        plotted = (table / ref).fillna(0.0)
        tick_labels = [f"{label}\n(réf. = {ref[label]:.3g})" for label in labels]
    else:
        plotted = table
        tick_labels = labels

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, row) in enumerate(plotted.iterrows()):
        values = row.values.astype(float)
        values_closed = np.concatenate((values, [values[0]]))
        color = colors[i % len(colors)]
        ax.plot(angles_closed, values_closed, "o-", linewidth=2, color=color, label=str(name))
        ax.fill(angles_closed, values_closed, alpha=0.12, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, fontsize=9)
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

    Retourne un DataFrame (lignes = trajectoires, colonnes = critères) des valeurs brutes.
    """
    if not trajectory_names:
        raise ValueError("Aucune trajectoire à comparer")
    if not criteria:
        raise ValueError("Aucun critère à calculer")

    parsed_criteria = [parse_criterion(c) for c in criteria]

    rows = {}
    for name in trajectory_names:
        file = find_trajectory_file(name, csv_folder=csv_folder, pattern=pattern)
        df = pd.read_csv(file)
        if derive_columns:
            df = add_derived_columns(df)
        try:
            rows[name] = compute_trajectory_criteria(df, parsed_criteria, time_column=time_column)
        except ValueError as exc:
            raise ValueError(f"{exc} (fichier : {os.path.basename(file)})") from exc

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "trajectoire"

    plot_comparison_radar(table, title=title, output_path=Path(output_path), normalize=normalize)
    return table


def main():
    parser = argparse.ArgumentParser(description="Trace un radar des maxima de grandeurs physiques depuis un Excel.")
    parser.add_argument("excel_path", help="Chemin vers le fichier Excel")
    parser.add_argument(
        "--sheet",
        default=0,
        help="Nom ou index de feuille Excel (défaut : 0)",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Colonnes à afficher sur le radar après création éventuelle des grandeurs dérivées",
    )
    parser.add_argument(
        "--output",
        default="radar_maxima.png",
        help="Chemin de sortie de l'image",
    )
    parser.add_argument(
        "--title",
        default="Radar des maxima des grandeurs physiques",
        help="Titre du graphique",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    df = add_derived_columns(df)
    max_values = plot_radar_from_df(
        df,
        columns=args.columns,
        output_path=args.output,
        title=args.title,
    )

    print("Maxima retenus :")
    print(max_values)


if __name__ == "__main__":
    main()
