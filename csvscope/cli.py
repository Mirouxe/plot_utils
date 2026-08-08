"""Interface en ligne de commande : obtenir un rapport sans écrire de code Python.

    python -m csvscope infos mes_csv
    python -m csvscope rapport mes_csv --sortie rapport.html
    python -m csvscope figure mes_csv --type curves --y temperature --couleur maillage
    python -m csvscope demo --dossier donnees_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import plots
from .dataset import load
from .interactive import save
from .report import report
from .synthetic import write_campaign

FIGURE_TYPES = {
    "curves": plots.curves,
    "grid": plots.grid,
    "explorer": plots.explorer,
    "envelope": plots.envelope,
    "small_multiples": plots.small_multiples,
    "compare": plots.compare,
    "bars": plots.bars,
    "heatmap": plots.heatmap,
    "distribution": plots.distribution,
    "scatter": plots.scatter,
    "pareto": plots.pareto,
    "parallel": plots.parallel,
    "radar": plots.radar,
    "metrics_table": plots.metrics_table,
}


def _add_loading_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dossier", help="dossier de CSV (ou motif glob)")
    parser.add_argument("--motif", default="*.csv", help="motif des fichiers (défaut : *.csv)")
    parser.add_argument("--temps", default=None, help="nom de la colonne de temps")
    parser.add_argument("--gabarit", default=None, help='gabarit de nom, ex. "cas_{materiau}_P{puissance}"')
    parser.add_argument("--regex", default=None, help="expression régulière à groupes nommés")
    parser.add_argument("--etiquette", default=None, help='gabarit d\'étiquette, ex. "{materiau} {puissance} kW"')
    parser.add_argument("--recursif", action="store_true", help="parcourir les sous-dossiers")


def _load(args: argparse.Namespace):
    return load(
        args.dossier,
        pattern=args.motif,
        time=args.temps,
        template=args.gabarit,
        regex=args.regex,
        label=args.etiquette,
        recursive=args.recursif,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csvscope",
        description="Graphiques interactifs pour comparer des lots de CSV de séries temporelles.",
    )
    subparsers = parser.add_subparsers(dest="commande", required=True)

    infos = subparsers.add_parser("infos", help="décrire le lot de CSV détecté")
    _add_loading_arguments(infos)

    rapport = subparsers.add_parser("rapport", help="générer le rapport HTML complet")
    _add_loading_arguments(rapport)
    rapport.add_argument("--sortie", default="rapport.html", help="fichier HTML produit")
    rapport.add_argument("--grandeurs", nargs="+", default=None, help="grandeurs à inclure")
    rapport.add_argument("--x", default=None, help="caractéristique en abscisse des comparaisons")
    rapport.add_argument("--couleur", default=None, help="caractéristique portant la couleur")
    rapport.add_argument("--metrique", default="max", help="métrique scalaire (max, mean, integral, …)")
    rapport.add_argument(
        "--plotlyjs",
        default="inline",
        choices=["inline", "cdn"],
        help="inline : fichier autonome ; cdn : fichier léger",
    )

    figure = subparsers.add_parser("figure", help="générer une figure isolée")
    _add_loading_arguments(figure)
    figure.add_argument("--type", dest="type_figure", default="curves", choices=sorted(FIGURE_TYPES))
    figure.add_argument("--y", default=None, help="grandeur tracée")
    figure.add_argument("--grandeurs", nargs="+", default=None, help="grandeurs (types multi-grandeurs)")
    figure.add_argument("--x", default=None, help="abscisse (grandeur ou caractéristique)")
    figure.add_argument("--couleur", default=None, help="caractéristique portant la couleur")
    figure.add_argument("--facette", default=None, help="caractéristique des facettes (small_multiples)")
    figure.add_argument("--metrique", default="max", help="métrique scalaire")
    figure.add_argument("--sortie", default="figure.html", help="fichier HTML produit")

    demo = subparsers.add_parser("demo", help="créer des CSV artificiels et leur rapport")
    demo.add_argument("--dossier", default="donnees_demo", help="dossier de sortie des CSV")
    demo.add_argument("--style", default="kv", choices=["kv", "compact"], help="style des noms de fichiers")
    demo.add_argument("--points", type=int, default=400, help="nombre de pas de temps par fichier")
    demo.add_argument("--sortie", default=None, help="rapport HTML à générer (optionnel)")

    return parser


def _figure_arguments(args: argparse.Namespace, dataset) -> dict:
    name = args.type_figure
    quantities = args.grandeurs or dataset.quantities
    y = args.y or quantities[0]

    if name in {"curves", "envelope"}:
        options = {"quantity": y}
        if name == "curves":
            options.update({"color": args.couleur, "dash": None, "x": args.x})
        else:
            options.update({"by": args.couleur})
        return options
    if name in {"grid", "explorer", "parallel", "radar", "metrics_table"}:
        options = {"quantities": quantities}
        if name in {"grid", "explorer"}:
            options["color"] = args.couleur
        if name in {"parallel", "radar"}:
            options["metric"] = args.metrique
        if name == "parallel":
            options["color"] = args.couleur
        if name == "radar":
            options["group"] = args.couleur
        return options
    if name == "small_multiples":
        return {"quantity": y, "facet": args.facette or dataset.characteristics[0], "color": args.couleur}
    if name == "compare":
        return {"quantity": y, "metric": args.metrique, "x": args.x, "color": args.couleur}
    if name == "heatmap":
        return {"quantity": y, "metric": args.metrique, "x": args.x, "y": args.couleur}
    if name == "bars":
        return {"quantity": y, "metric": args.metrique, "color": args.couleur}
    if name == "distribution":
        return {"quantity": y, "metric": args.metrique, "by": args.couleur}
    if name in {"scatter", "pareto"}:
        return {"x": y, "y": quantities[1] if len(quantities) > 1 else y,
                "metric": args.metrique, "color": args.couleur}
    raise ValueError(f"Type de figure non géré : {name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.commande == "demo":
        files = write_campaign(args.dossier, style=args.style, points=args.points)
        print(f"{len(files)} fichiers CSV écrits dans {Path(args.dossier).resolve()}")
        if args.sortie:
            dataset = load(args.dossier)
            path = report(dataset, path=args.sortie)
            print(f"Rapport : {path.resolve()}")
        return 0

    dataset = _load(args)

    if args.commande == "infos":
        print(dataset.overview())
        return 0

    if args.commande == "rapport":
        path = report(
            dataset,
            path=args.sortie,
            quantities=args.grandeurs,
            x=args.x,
            color=args.couleur,
            metric=args.metrique,
            plotlyjs=args.plotlyjs,
        )
        print(f"Rapport : {path.resolve()}")
        return 0

    if args.commande == "figure":
        figure = FIGURE_TYPES[args.type_figure](dataset, **_figure_arguments(args, dataset))
        path = save(figure, args.sortie)
        print(f"Figure : {path.resolve()}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
