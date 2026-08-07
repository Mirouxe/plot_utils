"""Interactivité ajoutée aux figures exportées en HTML.

Plotly fournit déjà le zoom, le survol et la légende cliquable. Deux manques se
font sentir dès qu'on superpose des dizaines de configurations :

1. retrouver *une* courbe dans le faisceau ;
2. ne garder que les configurations dont le nom contient tel motif.

Le script ci-dessous ajoute donc la mise en évidence au survol (les autres
courbes s'estompent), le verrouillage au clic, et une barre de filtrage par
expression régulière sur l'étiquette des configurations. Sa configuration est
lue dans ``figure.layout.meta.csvscope``, ce qui la rend visible côté navigateur
sans réécrire le script pour chaque figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import plotly.graph_objects as go

__all__ = ["configure", "get_config", "post_script", "save", "to_div", "show"]

DEFAULTS: dict[str, Any] = {
    "highlight": True,
    "search": True,
    "baseOpacity": 0.9,
    "dimOpacity": 0.08,
    "baseWidth": 1.9,
    "focusWidth": 3.6,
    "searchPlaceholder": "filtrer les configurations (regex)…",
}

_JS = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.data) { return; }
  var cfg = (gd.layout && gd.layout.meta && gd.layout.meta.csvscope) || {};
  var targets = [];
  gd.data.forEach(function (trace, index) {
    if (trace.meta && trace.meta.label) { targets.push(index); }
  });
  if (!targets.length) { return; }

  var baseOpacity = cfg.baseOpacity == null ? 0.9 : cfg.baseOpacity;
  var dimOpacity = cfg.dimOpacity == null ? 0.08 : cfg.dimOpacity;
  var baseWidth = cfg.baseWidth == null ? 1.9 : cfg.baseWidth;
  var focusWidth = cfg.focusWidth == null ? 3.6 : cfg.focusWidth;
  var labels = targets.map(function (i) { return String(gd.data[i].meta.label); });
  var locked = null;
  var filterRe = null;

  function matches(i) { return filterRe === null || filterRe.test(labels[i]); }

  function apply(focusIndex) {
    var opacity = [], width = [], hover = [];
    for (var i = 0; i < targets.length; i++) {
      var visible = matches(i);
      var focused = focusIndex === null ? null : targets[focusIndex] === targets[i];
      if (!visible) {
        opacity.push(0); width.push(baseWidth); hover.push('skip');
      } else if (focusIndex === null) {
        opacity.push(baseOpacity); width.push(baseWidth); hover.push('all');
      } else {
        opacity.push(focused ? 1 : dimOpacity);
        width.push(focused ? focusWidth : baseWidth);
        hover.push('all');
      }
    }
    Plotly.restyle(gd, { opacity: opacity, 'line.width': width, hoverinfo: hover }, targets);
  }

  if (cfg.highlight !== false) {
    gd.on('plotly_hover', function (evt) {
      if (locked !== null || !evt.points || !evt.points.length) { return; }
      var position = targets.indexOf(evt.points[0].curveNumber);
      if (position >= 0) { apply(position); }
    });
    gd.on('plotly_unhover', function () { if (locked === null) { apply(null); } });
    gd.on('plotly_click', function (evt) {
      if (!evt.points || !evt.points.length) { return; }
      var position = targets.indexOf(evt.points[0].curveNumber);
      if (position < 0) { return; }
      locked = locked === position ? null : position;
      apply(locked);
    });
    gd.on('plotly_doubleclick', function () { locked = null; apply(null); });
  }

  if (cfg.search !== false) {
    var bar = document.createElement('div');
    bar.style.cssText = 'display:flex;gap:8px;align-items:center;margin:6px 0 2px 0;' +
      'font:13px Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#374151;';
    var input = document.createElement('input');
    input.type = 'search';
    input.placeholder = cfg.searchPlaceholder || 'filtrer (regex)…';
    input.style.cssText = 'flex:0 1 320px;padding:5px 9px;border:1px solid #d1d5db;' +
      'border-radius:6px;font-size:13px;';
    var counter = document.createElement('span');
    counter.style.cssText = 'color:#6b7280;font-size:12px;';
    counter.textContent = targets.length + ' configurations';
    bar.appendChild(input);
    bar.appendChild(counter);
    gd.parentNode.insertBefore(bar, gd);

    input.addEventListener('input', function () {
      var text = input.value.trim();
      if (!text) {
        filterRe = null;
      } else {
        try { filterRe = new RegExp(text, 'i'); } catch (err) { return; }
      }
      var kept = 0;
      for (var i = 0; i < targets.length; i++) { if (matches(i)) { kept++; } }
      counter.textContent = kept + ' / ' + targets.length + ' configurations';
      locked = null;
      apply(null);
    });
  }
})();
"""


def configure(figure: go.Figure, **options: Any) -> go.Figure:
    """Attache les réglages d'interactivité à une figure (via ``layout.meta``)."""
    meta = dict(figure.layout.meta or {})
    current = dict(DEFAULTS)
    current.update(meta.get("csvscope", {}))
    current.update({k: v for k, v in options.items() if v is not None})
    meta["csvscope"] = current
    figure.layout.meta = meta
    return figure


def get_config(figure: go.Figure) -> dict[str, Any]:
    meta = dict(figure.layout.meta or {})
    config = dict(DEFAULTS)
    config.update(meta.get("csvscope", {}))
    return config


def post_script(figure: go.Figure) -> str | None:
    """Script à injecter dans le HTML, ou ``None`` si rien n'est demandé."""
    config = get_config(figure)
    if not (config.get("highlight") or config.get("search")):
        return None
    return _JS


def save(
    figure: go.Figure,
    path: str | Path,
    include_plotlyjs: str | bool = "cdn",
    config: Mapping[str, Any] | None = None,
    open_browser: bool = False,
) -> Path:
    """Enregistre une figure interactive autonome.

    ``include_plotlyjs="cdn"`` donne un fichier léger (nécessite le réseau à
    l'ouverture), ``True`` un fichier totalement autonome (~3 Mo).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        path,
        include_plotlyjs=include_plotlyjs,
        post_script=post_script(figure),
        config=_plot_config(config),
        full_html=True,
    )
    if open_browser:  # pragma: no cover - dépend de l'environnement
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return path


def to_div(figure: go.Figure, include_plotlyjs: str | bool = False, config: Mapping[str, Any] | None = None) -> str:
    """Fragment HTML d'une figure, pour l'assemblage dans un rapport."""
    return figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        post_script=post_script(figure),
        config=_plot_config(config),
    )


def show(figure: go.Figure, config: Mapping[str, Any] | None = None) -> None:
    """Affiche la figure (notebook ou navigateur), avec la barre d'outils réduite."""
    figure.show(config=_plot_config(config))


def _plot_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    base = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }
    base.update(dict(config or {}))
    return base
