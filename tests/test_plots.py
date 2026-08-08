import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import csvscope as cs
from csvscope import interactive


def _many_configs(count: int, points: int = 500) -> cs.Dataset:
    """Jeu volumineux en mémoire, pour tester les comportements adaptatifs."""
    time = np.linspace(0, 10, points)
    frames = {
        f"cas_{index:03d}": pd.DataFrame({"time": time, "y": np.sin(time) + index})
        for index in range(count)
    }
    meta = {name: {"indice": index} for index, name in enumerate(frames)}
    return cs.load_frames(frames, meta, time="time")


def test_curves_has_one_trace_per_configuration(dataset):
    figure = dataset.curves("temperature", color="maillage", dash="materiau")
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == len(dataset)
    assert "temperature" in figure.layout.title.text
    assert figure.layout.yaxis.title.text == "temperature [°C]"


def test_curves_legend_shows_one_entry_per_group(dataset):
    figure = dataset.curves("temperature", color="materiau")
    shown = [trace for trace in figure.data if trace.showlegend]
    assert len(shown) == len(dataset.values("materiau"))


def test_curves_accepts_another_quantity_as_abscissa(dataset):
    figure = dataset.curves("contrainte", x="temperature")
    assert figure.layout.xaxis.title.text == "temperature [°C]"


def test_curves_downsampling_limits_points(dataset):
    figure = dataset.curves("temperature", max_points=10)
    assert all(len(trace.x) <= 10 for trace in figure.data)


def test_grid_stacks_one_subplot_per_quantity(dataset):
    quantities = ["temperature", "pression", "contrainte"]
    figure = dataset.grid(quantities, color="puissance")
    assert len(figure.data) == len(dataset) * len(quantities)
    assert sum(trace.showlegend for trace in figure.data) == len(dataset.values("puissance"))


def test_explorer_builds_menus_for_quantities_and_colors(dataset):
    quantities = ["temperature", "pression"]
    figure = dataset.explorer(quantities, color="materiau")
    assert len(figure.data) == len(dataset) * len(quantities)
    assert len(figure.layout.updatemenus) == 2
    assert [b.label for b in figure.layout.updatemenus[0].buttons] == quantities
    assert sum(trace.visible is True for trace in figure.data) == len(dataset)


def test_envelope_draws_band_and_center_per_group(dataset):
    figure = dataset.envelope("temperature", by="materiau")
    assert len(figure.data) == 2 * len(dataset.values("materiau"))
    assert "P10–P90" in figure.layout.title.text
    assert "médiane" in figure.data[1].hovertemplate


def test_envelope_quantile_band_stays_within_minmax(dataset):
    quantile = dataset.envelope("temperature", band="quantiles", quantiles=(0.25, 0.75))
    minmax = dataset.envelope("temperature", band="minmax")
    assert max(quantile.data[0].y) <= max(minmax.data[0].y)
    assert min(quantile.data[0].y) >= min(minmax.data[0].y)


def test_envelope_std_band_is_accepted(dataset):
    assert dataset.envelope("temperature", by="materiau", band="std").data


def test_envelope_rejects_unknown_band(dataset):
    with pytest.raises(ValueError, match="minmax"):
        dataset.envelope("temperature", band="autre")


def test_small_multiples_uses_one_facet_per_value(dataset):
    figure = dataset.small_multiples("temperature", facet="materiau", color="puissance")
    assert len(figure.data) == len(dataset)
    assert len(figure.layout.annotations) >= len(dataset.values("materiau"))


def test_compare_draws_aggregate_and_points(dataset):
    figure = dataset.compare("temperature", metric="max", x="puissance", color="materiau")
    groups = len(dataset.values("materiau"))
    assert len(figure.data) == 2 * groups
    assert "maximum" in figure.layout.yaxis.title.text


def test_compare_without_aggregation_shows_only_points(dataset):
    figure = dataset.compare("temperature", x="puissance", aggregate=None)
    assert len(figure.data) == 1


def test_bars_can_be_limited_to_the_top_configurations(dataset):
    figure = dataset.bars("temperature", metric="max", top=5)
    assert len(figure.data[0].y) == 5


def test_heatmap_crosses_two_characteristics(dataset):
    figure = dataset.heatmap("temperature", x="puissance", y="materiau")
    assert figure.data[0].z.shape == (
        len(dataset.values("materiau")),
        len(dataset.values("puissance")),
    )


@pytest.mark.parametrize("kind", ["box", "violin", "strip", "hist", "ecdf"])
def test_distribution_supports_every_kind(dataset, kind):
    figure = dataset.distribution("temperature", by="materiau", kind=kind)
    assert figure.data


def test_distribution_rejects_unknown_kind(dataset):
    with pytest.raises(ValueError, match="kind"):
        dataset.distribution("temperature", kind="camembert")


def test_scatter_crosses_two_metrics(dataset):
    figure = dataset.scatter(
        ("temperature", "max"), ("rendement", "mean"), color="materiau", trend=True
    )
    assert len(figure.data) == len(dataset.values("materiau")) + 1
    assert "maximum de temperature" in figure.layout.xaxis.title.text


def test_scatter_accepts_the_same_quantity_on_both_axes(dataset):
    assert dataset.scatter(("temperature", "max"), ("temperature", "mean")).data


def test_pareto_finds_the_non_dominated_configurations():
    time = np.array([0.0, 1.0])
    values = {"c1": (1.0, 1.0), "c2": (2.0, 2.0), "c3": (0.0, 3.0), "c4": (3.0, 0.0)}
    frames = {
        name: pd.DataFrame({"time": time, "a": [a, a], "b": [b, b]})
        for name, (a, b) in values.items()
    }
    ds = cs.load_frames(frames, {name: {"id": name} for name in frames}, time="time")

    figure = ds.pareto(("a", "max"), ("b", "max"), sense=("min", "min"))
    front = figure.data[-1]
    assert "front de Pareto" in front.name
    assert sorted(zip(front.x, front.y)) == [(0.0, 3.0), (1.0, 1.0), (3.0, 0.0)]


def test_pareto_rejects_an_unknown_sense(dataset):
    with pytest.raises(ValueError, match="sense"):
        dataset.pareto("temperature", "rendement", sense=("min", "plus"))


def test_curves_switch_to_webgl_and_hide_the_legend_on_large_datasets():
    ds = _many_configs(150)
    figure = ds.curves("y")
    assert all(trace.type == "scattergl" for trace in figure.data)
    assert not any(trace.showlegend for trace in figure.data)
    assert figure.layout.showlegend is False


def test_curves_stay_in_svg_with_a_readable_legend_on_small_datasets(dataset):
    figure = dataset.curves("temperature", color="materiau")
    assert all(trace.type == "scatter" for trace in figure.data)
    assert any(trace.showlegend for trace in figure.data)


def test_curves_render_can_be_forced(dataset):
    figure = dataset.curves("temperature", render="webgl")
    assert all(trace.type == "scattergl" for trace in figure.data)
    with pytest.raises(ValueError, match="render"):
        dataset.curves("temperature", render="autre")


def test_bars_keep_the_top_twenty_by_default():
    ds = _many_configs(30, points=10)
    figure = ds.bars("y")
    assert len(figure.data[0].y) == 20


def test_parallel_dimensions_cover_characteristics_and_quantities(dataset):
    quantities = ["temperature", "pression"]
    figure = dataset.parallel(quantities, color="puissance")
    labels = [dimension.label for dimension in figure.data[0].dimensions]
    assert "materiau" in labels
    assert "temperature [°C]" in labels
    assert len(labels) == len(dataset.characteristics) + len(quantities)


def test_radar_groups_configurations(dataset):
    figure = dataset.radar(
        ["temperature", "pression", "contrainte"], group="materiau", normalize="minmax"
    )
    assert len(figure.data) == len(dataset.values("materiau"))


def test_radar_limits_the_number_of_traces(dataset):
    figure = dataset.radar(["temperature", "pression", "contrainte"], max_traces=4)
    assert len(figure.data) == 4


def test_radar_needs_three_quantities(dataset):
    with pytest.raises(ValueError, match="trois grandeurs"):
        dataset.radar(["temperature", "pression"])


def test_radar_rejects_unknown_normalisation(dataset):
    with pytest.raises(ValueError, match="normalize"):
        dataset.radar(["temperature", "pression", "contrainte"], normalize="autre")


def test_metrics_table_lists_every_configuration(dataset):
    figure = cs.metrics_table(dataset, ["temperature"], metrics=("max", "mean"))
    assert len(figure.data[0].cells.values[0]) == len(dataset)


def test_unknown_quantity_raises(dataset):
    with pytest.raises(KeyError, match="Grandeurs inconnues"):
        dataset.grid(["inexistante"])


def test_interactivity_configuration_is_attached(dataset):
    figure = dataset.curves("temperature")
    config = interactive.get_config(figure)
    assert config["highlight"] is True
    assert interactive.post_script(figure) is not None


def test_interactivity_can_be_disabled(dataset):
    figure = dataset.curves("temperature", interactivity=False)
    assert interactive.post_script(figure) is None


def test_save_writes_a_self_contained_page(dataset, tmp_path):
    path = cs.save(dataset.curves("temperature"), tmp_path / "figure.html")
    content = path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in content
    assert "plotly_hover" in content
    assert "filtrer" in content
    assert "réinitialiser" in content


def test_traces_carry_the_configuration_identity(dataset):
    figure = dataset.grid(["temperature", "pression"])
    configs = {trace.meta["config"] for trace in figure.data}
    assert configs == set(dataset.configs)
    assert all(trace.meta["label"] for trace in figure.data)


def test_report_builds_tabs_for_every_section(dataset, tmp_path):
    path = dataset.report(tmp_path / "rapport.html", plotlyjs="cdn")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    for tab in ("Vue d'ensemble", "Explorateur", "Comparaisons", "Récapitulatif"):
        assert tab in content
    assert content.count("<section>") >= 6


def test_report_accepts_custom_figures(dataset, tmp_path):
    figure = dataset.curves("temperature")
    path = dataset.report(
        tmp_path / "sur_mesure.html", figures={"Mes courbes": figure}, plotlyjs="cdn"
    )
    content = path.read_text(encoding="utf-8")
    assert "Mes courbes" in content
    assert content.count("<section>") == 1


def test_report_rejects_unknown_plotlyjs_mode(dataset, tmp_path):
    with pytest.raises(ValueError, match="plotlyjs"):
        dataset.report(tmp_path / "x.html", plotlyjs="autre")
