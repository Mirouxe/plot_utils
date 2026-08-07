#!/usr/bin/env python3
"""Démonstration complète de csvscope sur des données artificielles.

Le script :

1. écrit une campagne de CSV artificiels (une configuration par fichier, ses
   caractéristiques dans le nom du fichier) ;
2. la recharge avec csvscope ;
3. construit toute la galerie de graphiques interactifs ;
4. écrit un rapport HTML à onglets, et éventuellement les captures PNG.

    python examples/galerie_demo.py --png

Chaque bloc ci-dessous est volontairement court : c'est le code qu'il faudrait
écrire pour obtenir la figure correspondante sur tes propres données.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import csvscope as cs

RACINE = Path(__file__).resolve().parent


def construire_figures(ds: cs.Dataset) -> dict[str, object]:
    """Une entrée par figure : le nom servira de nom de fichier."""
    return {
        # Toutes les configurations superposées : couleur = puissance, trait = matériau.
        "01_courbes": ds.curves("temperature", color="puissance", dash="materiau"),

        # Plusieurs grandeurs, axes des temps synchronisés.
        "02_grille": ds.grid(
            ["temperature", "contrainte", "rendement", "vibration"], color="puissance"
        ),

        # Une seule figure, deux menus : grandeur affichée et caractéristique en couleur.
        "03_explorateur": ds.explorer(color="materiau"),

        # Faisceau min–max et courbe moyenne par matériau.
        "04_faisceau": ds.envelope("temperature", by="materiau", show_individual=True),

        # Une facette par matériau pour isoler son effet.
        "05_facettes": ds.small_multiples(
            "temperature", facet="materiau", color="puissance"
        ),

        # Sensibilité : maximum de température en fonction de la puissance.
        "06_sensibilite": ds.compare(
            "temperature", metric="max", x="puissance", color="materiau"
        ),

        # Classement des configurations les plus chaudes.
        "07_classement": ds.bars("temperature", metric="max", color="materiau", top=15),

        # Carte du croisement de deux caractéristiques.
        "08_carte": ds.heatmap("temperature", x="puissance", y="materiau", metric="max"),

        # Dispersion des maxima par matériau.
        "09_distribution": ds.distribution(
            "temperature", by="materiau", metric="max", kind="box"
        ),

        # Compromis entre deux grandeurs, un point par configuration.
        "10_compromis": ds.scatter(
            ("temperature", "max"),
            ("rendement", "mean"),
            color="materiau",
            size="contrainte",
            trend=True,
        ),

        # Filtrage multi-critères en glissant la souris sur les axes.
        "11_coordonnees_paralleles": ds.parallel(
            ["temperature", "contrainte", "vibration", "rendement"],
            metric="max",
            color="puissance",
        ),

        # Signature globale de chaque matériau.
        "12_radar": ds.radar(
            ["temperature", "pression", "contrainte", "vibration", "rendement"],
            metric="max",
            group="materiau",
            normalize="minmax",
        ),

        # Tableau récapitulatif trié.
        "13_recapitulatif": cs.metrics_table(
            ds, ["temperature", "contrainte", "rendement"], metrics=("max", "mean")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", default=str(RACINE / "donnees_demo"),
                        help="dossier où écrire les CSV artificiels")
    parser.add_argument("--sortie", default=str(RACINE / "figures"),
                        help="dossier des figures HTML")
    parser.add_argument("--points", type=int, default=400,
                        help="nombre de pas de temps par configuration")
    parser.add_argument("--png", action="store_true",
                        help="exporter aussi des captures PNG (nécessite kaleido)")
    parser.add_argument("--png-dossier", default=str(RACINE.parent / "docs" / "images"),
                        help="dossier des captures PNG")
    args = parser.parse_args()

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    # 1. Données artificielles : 48 configurations, noms de fichiers du style
    #    cas_materiau=alu_maillage=fin_puissance=20_debit=1p5.csv
    fichiers = cs.write_campaign(args.dossier, points=args.points)
    print(f"{len(fichiers)} fichiers CSV écrits dans {args.dossier}")

    # 2. Chargement : les caractéristiques sont lues dans les noms de fichiers.
    ds = cs.load(args.dossier, label="{materiau} · {puissance} kW · {debit} kg/s")
    print(ds.overview())

    # Grandeur dérivée, calculée à la volée pour toutes les configurations.
    ds = ds.derive(
        marge_thermique="180 - temperature",
        units={"marge_thermique": "°C"},
    )

    # 3. Galerie.
    figures = construire_figures(ds)
    for nom, figure in figures.items():
        chemin = cs.save(figure, sortie / f"{nom}.html")
        print(f"  {chemin.name}")

    # 4. Rapport à onglets, autonome.
    rapport = ds.report(
        sortie / "rapport.html",
        quantities=["temperature", "contrainte", "rendement", "vibration"],
        x="puissance",
        color="materiau",
        title="Campagne de refroidissement — 48 configurations",
    )
    print(f"Rapport : {rapport}")

    # Exemple de sélection : on ne garde qu'un sous-ensemble de configurations.
    fort = ds.filter(puissance=[20, 35], debit=lambda valeur: valeur > 1.0)
    cs.save(
        fort.curves("contrainte", color="materiau"),
        sortie / "14_selection_forte_puissance.html",
    )
    print(f"Sélection : {len(fort)} configurations sur {len(ds)}")

    if args.png:
        dossier_png = Path(args.png_dossier)
        dossier_png.mkdir(parents=True, exist_ok=True)
        for nom, figure in figures.items():
            largeur = 900 if nom == "12_radar" else 1150
            hauteur = min(int(figure.layout.height or 560), 1000)
            figure.write_image(dossier_png / f"{nom}.png", width=largeur, height=hauteur)
        print(f"Captures PNG dans {dossier_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
