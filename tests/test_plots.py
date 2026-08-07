import plotly.graph_objects as go
import pytest

import csvscope as cs
from csvscope import interactive


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


def test_envelope_draws_band_and_mean_per_group(dataset):
    figure = dataset.envelope("temperature", by="materiau")
    assert len(figure.data) == 2 * len(dataset.values("materiau"))


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
