import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
