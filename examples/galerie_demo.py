#!/usr/bin/env python3
"""Démonstration complète de csvscope sur une campagne artificielle de 288 configurations.

Le script :

1. écrit 288 CSV artificiels (une configuration par fichier, ses caractéristiques
   dans le nom du fichier : 3 matériaux × 2 maillages × 8 puissances × 6 débits) ;
2. les recharge avec csvscope ;
3. construit toute la galerie de graphiques interactifs ;
4. écrit un rapport HTML à onglets, les dashboards d'exploration, et éventuellement les captures PNG.

    python examples/galerie_demo.py --png

Chaque bloc ci-dessous est volontairement court : c'est le code qu'il faudrait
écrire pour obtenir la figure correspondante sur tes propres données.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import csvscope as cs

RACINE = Path(__file__).resolve().parent

MATERIAUX = ("alu", "cuivre", "composite")
MAILLAGES = ("moyen", "fin")
PUISSANCES = (5, 8, 12, 16, 20, 25, 30, 35)
DEBITS = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)


def construire_figures(ds: cs.Dataset) -> dict[str, object]:
    """Une entrée par figure : le nom servira de nom de fichier."""
    return {
        # 288 courbes superposées : opacité, épaisseur et rendu (WebGL) s'adaptent
        # seuls au volume ; la couleur ordonnée porte la puissance.
        "01_courbes": ds.curves("temperature", color="puissance"),

        # Plusieurs grandeurs, axes des temps synchronisés ; survoler une courbe
        # met en évidence la même configuration dans tous les sous-graphiques.
        "02_grille": ds.grid(
            ["temperature", "contrainte", "rendement", "vibration"], color="puissance"
        ),

        # Une seule figure, deux menus : grandeur affichée et caractéristique en couleur.
        "03_explorateur": ds.explorer(color="materiau"),

        # Médiane et bande P10–P90 par matériau : la vue de synthèse à ce volume.
        "04_faisceau": ds.envelope("temperature", by="materiau"),

        # Une facette par matériau pour isoler son effet.
        "05_facettes": ds.small_multiples(
            "temperature", facet="materiau", color="puissance"
        ),

        # Sensibilité : maximum de température en fonction de la puissance.
        "06_sensibilite": ds.compare(
            "temperature", metric="max", x="puissance", color="materiau"
        ),

        # Les 20 configurations les plus chaudes (sur 288).
        "07_classement": ds.bars("temperature", metric="max", color="materiau"),

        # Carte du croisement de deux caractéristiques.
        "08_carte": ds.heatmap("temperature", x="puissance", y="debit", metric="max"),

        # Dispersion des maxima par matériau.
        "09_distribution": ds.distribution(
            "temperature", by="materiau", metric="max", kind="box"
        ),

        # Compromis entre deux grandeurs, un point par configuration.
        "10_compromis": ds.scatter(
            ("temperature", "max"),
            ("vibration", "rms"),
            color="materiau",
            size="contrainte",
        ),

        # Front de Pareto à puissance fixée : refroidir coûte du pompage ; le front
        # isole les seules configurations qu'aucune autre ne bat sur les deux axes.
        "11_pareto": ds.filter(puissance=20).pareto(
            ("temperature", "max"),
            ("pompage", "mean"),
            sense=("min", "min"),
            color="materiau",
            log_y=True,
        ),

        # Filtrage multi-critères en glissant la souris sur les axes.
        "12_coordonnees_paralleles": ds.parallel(
            ["temperature", "contrainte", "vibration", "rendement"],
            metric="max",
            color="puissance",
        ),

        # Signature globale de chaque matériau.
        "13_radar": ds.radar(
            ["temperature", "pression", "contrainte", "vibration", "rendement"],
            metric="max",
            group="materiau",
            normalize="minmax",
        ),

        # Tableau récapitulatif (hauteur fixe, contenu défilant).
        "14_recapitulatif": cs.metrics_table(
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

    # 1. Données artificielles : 288 configurations, noms de fichiers du style
    #    cas_materiau=alu_maillage=fin_puissance=20_debit=1p6.csv
    fichiers = cs.write_campaign(
        args.dossier,
        materiaux=MATERIAUX,
        maillages=MAILLAGES,
        puissances=PUISSANCES,
        debits=DEBITS,
        points=args.points,
    )
    print(f"{len(fichiers)} fichiers CSV écrits dans {args.dossier}")

    # 2. Chargement : les caractéristiques sont lues dans les noms de fichiers.
    ds = cs.load(args.dossier, label="{materiau} · {puissance} kW · {debit} kg/s")
    print(ds.overview())

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
        title="Campagne de refroidissement — 288 configurations",
    )
    print(f"Rapport : {rapport}")

    # Exemple de sélection : tous les graphiques acceptent un jeu filtré.
    fort = ds.filter(puissance=lambda p: p >= 25, debit=lambda q: q >= 1.6)
    cs.save(
        fort.curves("contrainte", color="materiau"),
        sortie / "15_selection_forte_puissance.html",
    )
    print(f"Sélection : {len(fort)} configurations sur {len(ds)}")

    # 5. Dashboards d'exploration (fiche d'un CSV, comparaison de deux, etc.).
    dashboards = Path(args.sortie) / "dashboards"
    a = ds.filter(materiau="alu", maillage="fin", puissance=20, debit=lambda q: abs(float(q) - 1.6) < 1e-9).configs[0]
    b = ds.filter(materiau="cuivre", maillage="fin", puissance=20, debit=lambda q: abs(float(q) - 1.6) < 1e-9).configs[0]
    ds.inspect(a, dashboards / "fiche.html")
    ds.diff(a, b, dashboards / "comparaison.html")
    ds.quantity_board("temperature", dashboards / "grandeur.html")
    ds.snapshot(at="final", path=dashboards / "instant.html")
    ds.outliers(dashboards / "aberrantes.html")
    ds.neighbors(a, dashboards / "proches.html", k=8)
    ds.coverage(dashboards / "couverture.html")
    ds.quality(dashboards / "qualite.html")
    print(f"Dashboards : {dashboards}")

    if args.png:
        dossier_png = Path(args.png_dossier)
        dossier_png.mkdir(parents=True, exist_ok=True)
        for nom, figure in figures.items():
            largeur = 900 if "radar" in nom else 1150
            hauteur = min(int(figure.layout.height or 560), 1000)
            figure.write_image(dossier_png / f"{nom}.png", width=largeur, height=hauteur)
        from csvscope.dashboards import diff_figures, inspect_figures
        from csvscope.dashboards.tools import _count_heatmap

        fiche = inspect_figures(ds, a)
        fiche["grille"].write_image(dossier_png / "15_fiche.png", width=1150, height=900)
        cmp_figs = diff_figures(ds, a, b)
        cmp_figs["superposition"].write_image(
            dossier_png / "16_comparaison.png", width=1150, height=900
        )
        _count_heatmap(ds, "puissance", "debit").write_image(
            dossier_png / "17_couverture.png", width=1150, height=480
        )
        print(f"Captures PNG dans {dossier_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
