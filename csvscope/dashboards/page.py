"""Assembleur HTML pour les dashboards d'exploration.

Une page peut contenir des cartes d'indicateurs, des pastilles, des tableaux et
des figures Plotly, éventuellement répartis en onglets. Le fichier produit
s'ouvre dans un navigateur, sans serveur.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go

from ..interactive import to_div

__all__ = [
    "Dashboard",
    "chips_html",
    "kpi_html",
    "note_html",
    "split_html",
    "table_html",
]

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  :root {
    --border:#e5e7eb; --muted:#6b7280; --accent:#3b6ea5; --bg:#f4f6f8;
    --ok:#4c9f70; --warn:#d97706; --bad:#b5495b; --card:#ffffff;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#111827;
         font-family:Inter,"Segoe UI",Helvetica,Arial,sans-serif; }
  header { background:white; border-bottom:1px solid var(--border); padding:20px 28px 0 28px; }
  header .eyebrow { margin:0 0 4px 0; font-size:11px; letter-spacing:.08em;
                    text-transform:uppercase; color:var(--accent); font-weight:600; }
  header h1 { margin:0 0 4px 0; font-size:22px; font-weight:600; }
  header p { margin:0 0 14px 0; color:var(--muted); font-size:13px; }
  nav { display:flex; gap:4px; flex-wrap:wrap; }
  nav button { background:transparent; border:none; border-bottom:2px solid transparent;
               padding:10px 14px; font-size:13.5px; color:var(--muted); cursor:pointer;
               font-family:inherit; }
  nav button:hover { color:#111827; background:#f3f4f6; }
  nav button.active { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
  main { padding:22px 28px 48px 28px; }
  section { display:none; }
  section.active { display:block; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px;
          padding:12px 16px 16px 16px; margin-bottom:18px;
          box-shadow:0 1px 2px rgba(16,24,40,.04); }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
          gap:10px; margin-bottom:18px; }
  .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px;
         padding:12px 14px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
  .kpi .label { font-size:11px; color:var(--muted); text-transform:uppercase;
                letter-spacing:.04em; }
  .kpi .value { font-size:22px; font-weight:650; margin-top:4px; letter-spacing:-.02em; }
  .kpi .hint { font-size:12px; color:var(--muted); margin-top:2px; }
  .kpi.ok .value { color:var(--ok); }
  .kpi.warn .value { color:var(--warn); }
  .kpi.bad .value { color:var(--bad); }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px; }
  .chip { background:white; border:1px solid var(--border); border-radius:999px;
          padding:5px 11px; font-size:12.5px; color:#374151; }
  .chip b { font-weight:650; }
  .chip.same { background:#f0fdf4; border-color:#bbf7d0; }
  .chip.diff { background:#fff7ed; border-color:#fed7aa; }
  .chip.warn { background:#fffbeb; border-color:#fde68a; }
  .note { background:#eef4fb; border:1px solid #d6e4f5; border-radius:8px; padding:10px 14px;
          font-size:12.5px; color:#334155; margin-bottom:18px; line-height:1.55; }
  .note.warn { background:#fffbeb; border-color:#fde68a; }
  .note.bad { background:#fef2f2; border-color:#fecaca; }
  .split { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:18px; }
  @media (max-width: 900px) { .split { grid-template-columns:1fr; } }
  .id-card h3 { margin:0 0 2px 0; font-size:15px; }
  .id-card .file { margin:0 0 10px 0; font-size:12px; color:var(--muted); word-break:break-all; }
  .id-card dl { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; margin:0; font-size:13px; }
  .id-card dt { color:var(--muted); }
  .id-card dd { margin:0; font-weight:600; }
  .id-card dd.diff { color:var(--warn); }
  table.data { width:100%; border-collapse:collapse; font-size:12.5px; }
  table.data th { text-align:left; background:#eef2f7; padding:8px 10px; font-weight:600;
                  position:sticky; top:0; }
  table.data td { padding:6px 10px; border-top:1px solid var(--border); }
  table.data tr:nth-child(even) td { background:#f9fafb; }
  table.data td.diff { background:#fff7ed; font-weight:600; }
  table.data td.bad { background:#fef2f2; color:var(--bad); font-weight:600; }
  table.data td.ok { color:var(--ok); }
  .scroll { max-height:520px; overflow:auto; }
  footer { color:var(--muted); font-size:12px; padding:0 28px 28px 28px; }
</style>
</head>
<body>
<header>
  <p class="eyebrow">$eyebrow</p>
  <h1>$title</h1>
  <p>$subtitle</p>
  $nav
</header>
<main>
$panes
</main>
<footer>$footer</footer>
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
  if (document.querySelector('nav button')) { showPane(0); }
  else {
    document.querySelectorAll('main section').forEach(function (s) { s.classList.add('active'); });
  }
</script>
</body>
</html>
"""
)


def kpi_html(items: Sequence[Mapping[str, Any]]) -> str:
    cards = []
    for item in items:
        tone = item.get("tone") or ""
        hint = f'<div class="hint">{escape(str(item["hint"]))}</div>' if item.get("hint") else ""
        cards.append(
            f'<div class="kpi {escape(str(tone))}">'
            f'<div class="label">{escape(str(item["label"]))}</div>'
            f'<div class="value">{escape(str(item["value"]))}</div>'
            f"{hint}</div>"
        )
    return f'<div class="kpis">{"".join(cards)}</div>'


def chips_html(items: Sequence[Mapping[str, Any]]) -> str:
    chips = []
    for item in items:
        tone = item.get("tone") or ""
        value = item.get("value", "")
        chips.append(
            f'<span class="chip {escape(str(tone))}">'
            f'{escape(str(item["label"]))} <b>{escape(str(value))}</b></span>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def note_html(text: str, tone: str = "") -> str:
    return f'<div class="note {escape(tone)}">{text}</div>'


def split_html(*columns: str) -> str:
    cells = "".join(f'<div>{column}</div>' for column in columns)
    return f'<div class="split">{cells}</div>'


def table_html(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    cell_classes: Sequence[Sequence[str]] | None = None,
    scroll: bool = True,
) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cells = []
        for j, value in enumerate(row):
            klass = ""
            if cell_classes and i < len(cell_classes) and j < len(cell_classes[i]):
                klass = cell_classes[i][j]
            attr = f' class="{escape(klass)}"' if klass else ""
            cells.append(f"<td{attr}>{escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    if scroll:
        return f'<div class="card"><div class="scroll">{table}</div></div>'
    return f'<div class="card">{table}</div>'


def identity_card(
    title: str,
    filename: str,
    characteristics: Mapping[str, Any],
    differing: Iterable[str] = (),
) -> str:
    differing = set(differing)
    rows = []
    for key, value in characteristics.items():
        klass = "diff" if key in differing else ""
        rows.append(
            f"<dt>{escape(str(key))}</dt><dd class='{klass}'>{escape(str(value))}</dd>"
        )
    return (
        '<div class="card id-card">'
        f"<h3>{escape(title)}</h3>"
        f'<p class="file">{escape(filename)}</p>'
        f"<dl>{''.join(rows)}</dl>"
        "</div>"
    )


class Dashboard:
    """Page HTML à une ou plusieurs sections, écrite d'un coup à la fin."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        eyebrow: str = "csvscope",
        footer: str = "Dashboard généré avec csvscope.",
    ):
        self.title = title
        self.subtitle = subtitle
        self.eyebrow = eyebrow
        self.footer = footer
        self._tabs: list[tuple[str, list[Any]]] = []
        self._current: list[Any] | None = None

    def tab(self, name: str) -> "Dashboard":
        self._current = []
        self._tabs.append((name, self._current))
        return self

    def _target(self) -> list[Any]:
        if self._current is None:
            self.tab("Vue d'ensemble")
        assert self._current is not None
        return self._current

    def add(self, html: str) -> "Dashboard":
        self._target().append(("html", html))
        return self

    def kpis(self, items: Sequence[Mapping[str, Any]]) -> "Dashboard":
        return self.add(kpi_html(items))

    def chips(self, items: Sequence[Mapping[str, Any]]) -> "Dashboard":
        return self.add(chips_html(items))

    def note(self, text: str, tone: str = "") -> "Dashboard":
        return self.add(note_html(text, tone))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], **kwargs: Any) -> "Dashboard":
        return self.add(table_html(headers, rows, **kwargs))

    def figure(self, figure: go.Figure) -> "Dashboard":
        self._target().append(("figure", figure))
        return self

    def split(self, *columns: str) -> "Dashboard":
        return self.add(split_html(*columns))

    def write(self, path: str | Path, plotlyjs: str = "cdn") -> Path:
        if plotlyjs not in {"inline", "cdn"}:
            raise ValueError("plotlyjs doit valoir 'inline' ou 'cdn'.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._tabs:
            self.tab("Vue d'ensemble")

        multi = len(self._tabs) > 1
        tabs_html = ""
        if multi:
            buttons = [
                f'<button type="button" onclick="showPane({i})">{escape(name)}</button>'
                for i, (name, _) in enumerate(self._tabs)
            ]
            tabs_html = f"<nav>{''.join(buttons)}</nav>"

        first_figure = True
        panes = []
        for name, fragments in self._tabs:
            parts = []
            for kind, payload in fragments:
                if kind == "html":
                    parts.append(payload)
                    continue
                include: str | bool = False
                if first_figure:
                    include = True if plotlyjs == "inline" else "cdn"
                    first_figure = False
                parts.append(
                    '<div class="card">' + to_div(payload, include_plotlyjs=include) + "</div>"
                )
            body = "".join(parts) or '<div class="note">Aucune donnée à afficher.</div>'
            panes.append(f"<section>{body}</section>")

        path.write_text(
            _PAGE.substitute(
                title=escape(self.title),
                subtitle=escape(self.subtitle),
                eyebrow=escape(self.eyebrow),
                footer=escape(self.footer),
                nav=tabs_html,
                panes="\n".join(panes),
            ),
            encoding="utf-8",
        )
        return path
