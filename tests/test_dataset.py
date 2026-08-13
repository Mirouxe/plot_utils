import numpy as np
import pandas as pd
import pytest

import csvscope as cs


def test_load_reads_characteristics_units_and_time(dataset):
    assert len(dataset) == 16
    assert dataset.characteristics == ["materiau", "maillage", "puissance", "debit"]
    assert dataset.time == "temps"
    assert dataset.units["temperature"] == "°C"
    assert "temperature" in dataset.quantities
    assert dataset.time not in dataset.quantities


def test_constant_prefix_token_is_dropped(dataset):
    assert "tag1" not in dataset.meta.columns


def test_labels_follow_the_template(dataset):
    assert dataset.label(dataset.configs[0]) == "alu 5 kW"


def test_configurations_are_sorted_naturally(campaign):
    dataset = cs.load(campaign)
    # Un tri alphabétique placerait la puissance 20 avant la 5.
    group = dataset.filter(materiau="alu", maillage="fin")
    assert [group.meta.at[c, "puissance"] for c in group.configs] == [5, 5, 20, 20]

    ordered = dataset.sort("puissance")
    puissances = [ordered.meta.at[c, "puissance"] for c in ordered.configs]
    assert puissances == sorted(puissances)


def test_load_with_template(tmp_path):
    cs.write_campaign(
        tmp_path,
        materiaux=("alu",),
        maillages=("fin",),
        puissances=(20,),
        debits=(1.5,),
        style="compact",
        points=20,
    )
    dataset = cs.load(tmp_path, template="essai_{materiau}_{maillage}_P{puissance}_Q{debit}")
    assert dataset.characteristics == ["materiau", "maillage", "puissance", "debit"]
    assert dataset.meta.iloc[0]["debit"] == 1.5


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cs.load(tmp_path / "vide")


def test_unknown_time_column_raises(campaign):
    with pytest.raises(ValueError, match="Colonne de temps"):
        cs.load(campaign, time="inexistant")


def test_filter_accepts_value_list_and_predicate(dataset):
    assert len(dataset.filter(materiau="alu")) == 8
    assert len(dataset.filter(puissance=[5, 20])) == 16
    assert len(dataset.filter(debit=lambda value: value > 1.0)) == 8
    assert len(dataset.filter(materiau="alu", maillage="fin")) == 4


def test_filter_on_unknown_characteristic_raises(dataset):
    with pytest.raises(KeyError):
        dataset.filter(inconnu=1)


def test_filter_without_match_raises(dataset):
    with pytest.raises(ValueError, match="Aucune configuration"):
        dataset.filter(puissance=999)


def test_select_and_sort(dataset):
    subset = dataset.select(dataset.configs[:3])
    assert len(subset) == 3
    descending = dataset.sort("puissance", ascending=False)
    assert descending.meta["puissance"].iloc[0] == 20


def test_derive_adds_quantities(dataset):
    enriched = dataset.derive(
        ecart="temperature - 22",
        puissance_dissipee=lambda frame: frame["debit_mesure"] * frame["temperature"],
        units={"ecart": "°C"},
    )
    assert "ecart" in enriched.quantities
    assert "puissance_dissipee" in enriched.quantities
    assert enriched.units["ecart"] == "°C"
    first = enriched.frames[enriched.configs[0]]
    assert np.allclose(first["ecart"], first["temperature"] - 22)


def test_add_characteristic_from_meta(dataset):
    tagged = dataset.add_characteristic(
        regime=lambda row: "fort" if row["puissance"] >= 20 else "faible"
    )
    assert set(tagged.values("regime")) == {"faible", "fort"}
    assert len(tagged.filter(regime="fort")) == 8


def test_table_single_metric_uses_quantity_as_column(dataset):
    table = dataset.table("temperature", "max")
    assert "temperature" in table.columns
    assert len(table) == len(dataset)
    assert table["temperature"].gt(22).all()


def test_table_multiple_metrics_are_suffixed(dataset):
    table = dataset.table(["temperature", "pression"], ["max", "mean"])
    for column in ("temperature_max", "temperature_mean", "pression_max", "pression_mean"):
        assert column in table.columns
    assert (table["temperature_max"] >= table["temperature_mean"]).all()


def test_table_keeps_characteristics(dataset):
    table = dataset.table("temperature", "max")
    for characteristic in dataset.characteristics:
        assert characteristic in table.columns


def test_long_and_wide_formats(dataset):
    long = dataset.long(["temperature", "pression"])
    assert set(long.columns) >= {"config", "label", "temps", "quantity", "value", "materiau"}
    assert set(long["quantity"].unique()) == {"temperature", "pression"}

    wide = dataset.wide(["temperature"])
    assert "temperature" in wide.columns
    assert len(wide) == sum(len(frame) for frame in dataset.frames.values())


def test_resample_puts_every_config_on_the_same_grid(dataset):
    resampled = dataset.resample(50)
    grids = [frame["temps"].to_numpy() for frame in resampled.frames.values()]
    assert all(len(grid) == 50 for grid in grids)
    assert np.allclose(grids[0], grids[-1])


def test_common_grid_covers_the_shared_range(dataset):
    grid = dataset.common_grid(10)
    assert grid[0] >= 0
    assert len(grid) == 10


def test_overview_mentions_characteristics_and_quantities(dataset):
    text = dataset.overview()
    assert "materiau" in text
    assert "temperature" in text
    assert "16 configurations" in text


def test_load_frames_from_memory():
    frames = {
        "a": pd.DataFrame({"time": [0, 1, 2], "y": [1.0, 2.0, 3.0]}),
        "b": pd.DataFrame({"time": [0, 1, 2], "y": [2.0, 3.0, 4.0]}),
    }
    meta = {"a": {"cas": "a"}, "b": {"cas": "b"}}
    dataset = cs.load_frames(frames, meta, time="time")
    assert dataset.quantities == ["y"]
    assert dataset.table("y", "max")["y"].tolist() == [3.0, 4.0]
