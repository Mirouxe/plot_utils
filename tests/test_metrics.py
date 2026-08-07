import numpy as np
import pandas as pd
import pytest

import csvscope as cs
from csvscope import metrics


@pytest.fixture
def series():
    time = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    values = pd.Series([1.0, 3.0, 5.0, 4.0, 2.0])
    return values, time


@pytest.mark.parametrize(
    "name, expected",
    [
        ("max", 5.0),
        ("min", 1.0),
        ("mean", 3.0),
        ("median", 3.0),
        ("range", 4.0),
        ("final", 2.0),
        ("initial", 1.0),
        ("t_max", 2.0),
        ("t_min", 0.0),
        ("integral", 13.5),
    ],
)
def test_builtin_metrics(series, name, expected):
    values, time = series
    assert metrics.METRICS[name](values, time) == pytest.approx(expected)


def test_rms_and_std(series):
    values, time = series
    assert metrics.METRICS["rms"](values, time) == pytest.approx(np.sqrt(11.0))
    assert metrics.METRICS["std"](values, time) == pytest.approx(values.std(ddof=1))


def test_overshoot_is_relative_to_the_final_value(series):
    values, time = series
    assert metrics.METRICS["overshoot"](values, time) == pytest.approx(1.5)


def test_slope_of_a_straight_line():
    time = pd.Series([0.0, 1.0, 2.0])
    values = pd.Series([0.0, 2.0, 4.0])
    assert metrics.METRICS["slope"](values, time) == pytest.approx(2.0)


def test_non_finite_values_are_ignored():
    time = pd.Series([0.0, 1.0, 2.0])
    values = pd.Series([1.0, np.nan, 3.0])
    assert metrics.METRICS["max"](values, time) == 3.0
    assert metrics.METRICS["mean"](values, time) == pytest.approx(2.0)


def test_empty_series_gives_nan():
    empty = pd.Series([], dtype=float)
    assert np.isnan(metrics.METRICS["max"](empty, empty))


def test_unknown_metric_raises():
    with pytest.raises(KeyError, match="Métrique inconnue"):
        metrics.resolve_metric("inexistante")


def test_callable_metric_is_accepted(dataset):
    def amplitude(values, time):
        return float(values.max() - values.min())

    table = dataset.table("temperature", amplitude)
    assert table["temperature"].gt(0).all()


def test_register_metric_is_usable_by_name(dataset):
    @metrics.register_metric("premier_quartile", "premier quartile")
    def _q1(values, time):
        return float(values.quantile(0.25))

    assert "premier_quartile" in cs.METRICS
    table = dataset.table("temperature", "premier_quartile")
    assert table["temperature"].notna().all()
    assert metrics.metric_label("premier_quartile") == "premier quartile"


def test_normalize_metrics_accepts_several_forms():
    assert list(metrics.normalize_metrics("max")) == ["max"]
    assert list(metrics.normalize_metrics(["max", "mean"])) == ["max", "mean"]
    assert list(metrics.normalize_metrics({"pic": "max"})) == ["pic"]
