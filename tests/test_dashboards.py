from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import csvscope as cs
from csvscope.dashboards.common import resolve_config, series_scores


def test_resolve_config_accepts_index_key_and_substring(dataset):
    first = dataset.configs[0]
    assert resolve_config(dataset, 0) == first
    assert resolve_config(dataset, first) == first
    assert resolve_config(dataset, Path(dataset.file(first)).name) == first
    unique = dataset.filter(materiau="cuivre", maillage="fin", puissance=20, debit=1.5)
    assert len(unique) == 1
    stem = Path(dataset.file(unique.configs[0])).stem
    assert resolve_config(dataset, stem) == unique.configs[0]


def test_resolve_config_rejects_ambiguous_and_unknown(dataset):
    with pytest.raises(KeyError, match="ambigu"):
        resolve_config(dataset, "alu")
    with pytest.raises(KeyError, match="Aucune configuration"):
        resolve_config(dataset, "inexistant")
    with pytest.raises(IndexError):
        resolve_config(dataset, 999)


def test_inspect_writes_a_fiche_for_one_configuration(dataset, tmp_path):
    config = dataset.configs[0]
    path = dataset.inspect(config, tmp_path / "fiche.html")
    html = path.read_text(encoding="utf-8")
    assert "Fiche configuration" in html
    assert dataset.label(config) in html
    assert "Points" in html
    assert "plotly" in html.lower()
    assert "Dans la campagne" in html


def test_inspect_file_loads_a_single_csv(campaign, tmp_path):
    csv_path = next(Path(campaign).glob("*.csv"))
    path = cs.inspect_file(csv_path, tmp_path / "seul.html")
    html = path.read_text(encoding="utf-8")
    assert "Fiche configuration" in html
    assert csv_path.name in html or csv_path.stem in html


def test_inspect_requires_a_config_when_several_are_loaded(dataset, tmp_path):
    with pytest.raises(ValueError, match="précise config"):
        cs.inspect(dataset, path=tmp_path / "x.html")


def test_diff_highlights_differing_characteristics(dataset, tmp_path):
    alu = dataset.filter(materiau="alu", maillage="fin", puissance=20, debit=1.5).configs[0]
    cuivre = dataset.filter(materiau="cuivre", maillage="fin", puissance=20, debit=1.5).configs[0]
    path = dataset.diff(alu, cuivre, tmp_path / "cmp.html")
    html = path.read_text(encoding="utf-8")
    assert "Comparaison de deux configurations" in html
    assert "materiau" in html
    assert "alu" in html and "cuivre" in html
    assert "RMSE" in html
    assert "Écarts" in html


def test_diff_files_compares_two_csv(campaign, tmp_path):
    files = sorted(Path(campaign).glob("*.csv"))[:2]
    path = cs.diff_files(files[0], files[1], tmp_path / "deux.html")
    html = path.read_text(encoding="utf-8")
    assert "vs" in html
    assert files[0].name in html
    assert files[1].name in html


def test_diff_rejects_the_same_configuration(dataset, tmp_path):
    with pytest.raises(ValueError, match="identiques"):
        dataset.diff(0, 0, tmp_path / "x.html")


def test_quantity_board_and_snapshot_and_coverage(dataset, tmp_path):
    board = dataset.quantity_board("temperature", tmp_path / "g.html")
    assert "Exploration de temperature" in board.read_text(encoding="utf-8")

    instant = dataset.snapshot(at="final", path=tmp_path / "t.html")
    html = instant.read_text(encoding="utf-8")
    assert "Instantané" in html

    cov = dataset.coverage(tmp_path / "c.html")
    text = cov.read_text(encoding="utf-8")
    assert "Couverture" in text
    assert "100 %" in text  # 2×2×2×2 = 16, campagne complète


def test_outliers_flags_an_injected_extreme(dataset, tmp_path):
    target = dataset.configs[0]
    frames = {name: frame.copy() for name, frame in dataset.frames.items()}
    frames[target]["temperature"] = frames[target]["temperature"] * 50
    twisted = cs.load_frames(frames, dataset.meta, time=dataset.time, units=dataset.units)
    path = twisted.outliers(tmp_path / "out.html", z=2.5)
    html = path.read_text(encoding="utf-8")
    assert twisted.label(target) in html
    assert "Aberrantes" in html


def test_neighbors_rank_the_query_neighbours_first(dataset, tmp_path):
    query = dataset.filter(materiau="alu", maillage="fin", puissance=5, debit=0.6).configs[0]
    path = dataset.neighbors(query, tmp_path / "n.html", k=3)
    html = path.read_text(encoding="utf-8")
    assert "Voisines" in html
    assert dataset.label(query) in html


def test_quality_reports_sampling_and_nans(dataset, tmp_path):
    path = dataset.quality(tmp_path / "q.html")
    html = path.read_text(encoding="utf-8")
    assert "Qualité des CSV" in html
    assert "Points (médiane)" in html


def test_series_scores_are_zero_for_identical_series():
    y = np.linspace(0, 1, 50)
    scores = series_scores(y, y)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["corr"] == pytest.approx(1.0)


def test_cli_dashboard_inspect(dataset, campaign, tmp_path):
    from csvscope.cli import main

    sortie = tmp_path / "cli.html"
    code = main(
        [
            "dashboard",
            str(campaign),
            "--type",
            "inspect",
            "--config",
            "0",
            "--sortie",
            str(sortie),
        ]
    )
    assert code == 0
    assert sortie.exists()
    assert "Fiche configuration" in sortie.read_text(encoding="utf-8")
