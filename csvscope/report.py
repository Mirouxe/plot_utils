"""Assemblage de plusieurs figures en un rapport HTML autonome à onglets.

Un fichier unique, ouvrable par un collègue sans Python ni serveur, contenant
toutes les vues d'une campagne de calculs.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go

from . import plots
from .interactive import to_div

__all__ = ["build_report", "report", "auto_sections"]

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  :root { --border:#e5e7eb; --muted:#6b7280; --accent:#3b6ea5; --bg:#f7f8fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#111827;
         font-family:Inter,"Segoe UI",Helvetica,Arial,sans-serif; }
  header { background:white; border-bottom:1px solid var(--border); padding:22px 32px 0 32px; }
  header h1 { margin:0 0 4px 0; font-size:22px; font-weight:600; }
  header p { margin:0 0 16px 0; color:var(--muted); font-size:13px; }
  nav { display:flex; gap:4px; flex-wrap:wrap; }
  nav button { background:transparent; border:none; border-bottom:2px solid transparent;
               padding:10px 14px; font-size:13.5px; color:var(--muted); cursor:pointer;
               font-family:inherit; }
  nav button:hover { color:#111827; background:#f3f4f6; }
  nav button.active { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
  main { padding:24px 32px 60px 32px; }
  section { display:none; }
  section.active { display:block; }
  .card { background:white; border:1px solid var(--border); border-radius:10px;
          padding:10px 14px 14px 14px; margin-bottom:22px;
          box-shadow:0 1px 2px rgba(16,24,40,.04); }
  .note { background:#eef4fb; border:1px solid #d6e4f5; border-radius:8px; padding:10px 14px;
          font-size:12.5px; color:#334155; margin-bottom:22px; line-height:1.55; }
  .note code { background:white; border:1px solid var(--border); border-radius:4px;
               padding:1px 5px; font-size:12px; }
  footer { color:var(--muted); font-size:12px; padding:0 32px 30px 32px; }
</style>
</head>
<body>
<header>
  <h1>$title</h1>
  <p>$subtitle</p>
  <nav>$tabs</nav>
</header>
<main>
$panes
</main>
<footer>Rapport généré avec csvscope.</footer>
<script>
  function showPane(index) {
    document.querySelectorAll('nav button').forEach(function (b, i) {
      b.classList.toggle('active', i === index);
    });
    document.querySelectorAll('main section').forEach(function (s, i) {
      s.classList.toggle('active', i === index);
    });
    var pane = document.querySelectorAll('main section')[index];
    if (pane && window.Plotly) {
      pane.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
        Plotly.Plots.resize(gd);
      });
    }
  }
  showPane(0);
</script>
</body>
</html>
"""
)

_HINT = (
    "<b>Interactions</b> : survole une courbe pour l'isoler du faisceau, clique pour la "
    "verrouiller, puis <code>réinitialiser</code> pour tout rétablir. Le champ de filtrage "
    "accepte une expression régulière sur l'étiquette des configurations. La légende reste "
    "cliquable (clic simple : masquer un groupe, double-clic : n'afficher que lui), et les "
    "axes des coordonnées parallèles se filtrent en glissant la souris."
)


def build_report(
    sections: Mapping[str, go.Figure | Sequence[go.Figure]],
    path: str | Path = "rapport.html",
    title: str = "Analyse des configurations",
    subtitle: str = "",
    hint: bool = True,
    plotlyjs: str = "inline",
) -> Path:
    """Écrit un rapport à onglets depuis un dictionnaire ``onglet -> figure(s)``.

    ``plotlyjs="inline"`` produit un fichier autonome (plus lourd de ~4 Mo),
    ``"cdn"`` un fichier léger mais qui demande un accès réseau à l'ouverture.

    >>> build_report({"Courbes": fig1, "Comparaisons": [fig2, fig3]}, "rapport.html")
    """
    if plotlyjs not in {"inline", "cdn"}:
        raise ValueError("plotlyjs doit valoir 'inline' ou 'cdn'.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tabs, panes = [], []
    first_figure = True

    for index, (name, figures) in enumerate(sections.items()):
        if isinstance(figures, go.Figure):
            figures = [figures]
        tabs.append(
            f'<button type="button" onclick="showPane({index})">{name}</button>'
        )
        cards = [f'<div class="note">{_HINT}</div>'] if (hint and index == 0) else []
        for figure in figures:
            include: str | bool = False
            if first_figure:
                include = True if plotlyjs == "inline" else "cdn"
            cards.append('<div class="card">' + to_div(figure, include_plotlyjs=include) + "</div>")
            first_figure = False
        panes.append(f'<section>{"".join(cards)}</section>')

    path.write_text(
        _PAGE.substitute(
            title=title,
            subtitle=subtitle,
            tabs="".join(tabs),
            panes="\n".join(panes),
        ),
        encoding="utf-8",
    )
    return path


def _pick_characteristics(dataset, count: int = 2) -> list[str]:
    """Choisit les caractéristiques les plus discriminantes pour les axes."""
    ranked = sorted(
        (c for c in dataset.characteristics if len(dataset.values(c)) > 1),
        key=lambda c: (-len(dataset.values(c)), c),
    )
    if not ranked:
        ranked = list(dataset.characteristics)
    return ranked[:count]


def auto_sections(
    dataset,
    quantities: Sequence[str] | None = None,
    x: str | None = None,
    color: str | None = None,
    metric: Any = "max",
) -> dict[str, list[go.Figure]]:
    """Construit un jeu de figures couvrant les questions les plus fréquentes."""
    quantities = list(quantities or dataset.quantities)
    picked = _pick_characteristics(dataset, 2)
    x = x or (picked[0] if picked else None)
    if color is None:
        color = picked[1] if len(picked) > 1 else x

    sections: dict[str, list[go.Figure]] = {
        "Vue d'ensemble": [
            plots.grid(dataset, quantities, color=color),
        ],
        "Explorateur": [
            plots.explorer(dataset, quantities, color=color),
        ],
        "Comparaisons": [
            plots.compare(dataset, quantities[0], metric=metric, x=x, color=color),
            plots.bars(dataset, quantities[0], metric=metric, color=color),
        ],
        "Dispersion": [
            plots.envelope(dataset, quantities[0], by=color),
            plots.distribution(dataset, quantities[0], by=color, metric=metric, kind="box"),
        ],
        "Multi-grandeurs": [
            plots.parallel(dataset, quantities, metric=metric, color=x),
        ],
        "Récapitulatif": [
            plots.metrics_table(dataset, quantities, metrics=("max", "mean")),
        ],
    }

    if x is not None and color is not None and x != color:
        sections["Comparaisons"].append(
            plots.heatmap(dataset, quantities[0], x=x, y=color, metric=metric)
        )
    if len(quantities) >= 3:
        sections["Multi-grandeurs"].append(
            plots.radar(dataset, quantities, metric=metric, group=color)
        )
    if len(quantities) >= 2:
        sections["Comparaisons"].append(
            plots.scatter(dataset, quantities[0], quantities[1], metric=metric, color=color)
        )
    return sections


def report(
    dataset,
    path: str | Path = "rapport.html",
    figures: Mapping[str, go.Figure | Sequence[go.Figure]] | None = None,
    quantities: Sequence[str] | None = None,
    x: str | None = None,
    color: str | None = None,
    metric: Any = "max",
    title: str | None = None,
    subtitle: str | None = None,
    plotlyjs: str = "inline",
) -> Path:
    """Rapport complet en une ligne, ou assemblage de figures choisies.

    >>> ds.report("rapport.html")                       # analyse automatique
    >>> ds.report("rapport.html", figures={"Mes courbes": fig})
    """
    sections = (
        dict(figures)
        if figures is not None
        else auto_sections(dataset, quantities=quantities, x=x, color=color, metric=metric)
    )
    default_subtitle = (
        f"{len(dataset)} configurations · "
        f"caractéristiques : {', '.join(dataset.characteristics)} · "
        f"grandeurs : {', '.join(quantities or dataset.quantities)}"
    )
    return build_report(
        sections,
        path=path,
        title=title or "Analyse des configurations",
        subtitle=subtitle if subtitle is not None else default_subtitle,
        plotlyjs=plotlyjs,
    )
