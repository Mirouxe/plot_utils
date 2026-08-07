"""csvscope — comparer des lots de CSV de séries temporelles en quelques lignes.

Le point de départ est toujours le même : un dossier de CSV où chaque fichier est
une configuration, son nom porte les caractéristiques de cette configuration, et
ses colonnes portent des grandeurs calculées en fonction du temps.

    import csvscope as cs

    ds = cs.load("mes_csv", template="cas_{materiau}_{maillage}_P{puissance}")
    print(ds.overview())

    cs.save(ds.curves("temperature", color="maillage"), "temperature.html")
    ds.report("rapport.html")

Toutes les figures sont des objets Plotly : elles restent modifiables
(``fig.update_layout(...)``) avant sauvegarde.
"""

from __future__ import annotations

from .dataset import Dataset, load, load_frames
from .interactive import configure, save, show, to_div
from .metrics import METRICS, register_metric
from .plots import (
    bars,
    compare,
    curves,
    distribution,
    envelope,
    explorer,
    grid,
    heatmap,
    metrics_table,
    parallel,
    radar,
    scatter,
    small_multiples,
)
from .report import build_report, report
from .synthetic import demo_dataset, generate_series, write_campaign
from .theme import QUALITATIVE, SEQUENTIAL, install_template

__version__ = "0.1.0"

install_template()

__all__ = [
    "Dataset",
    "load",
    "load_frames",
    "demo_dataset",
    "generate_series",
    "write_campaign",
    "curves",
    "grid",
    "explorer",
    "envelope",
    "small_multiples",
    "compare",
    "bars",
    "heatmap",
    "distribution",
    "scatter",
    "metrics_table",
    "parallel",
    "radar",
    "report",
    "build_report",
    "save",
    "show",
    "to_div",
    "configure",
    "register_metric",
    "METRICS",
    "install_template",
    "QUALITATIVE",
    "SEQUENTIAL",
    "__version__",
]
