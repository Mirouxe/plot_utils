import pytest

import csvscope as cs
from csvscope.synthetic import write_campaign


@pytest.fixture(scope="session")
def campaign(tmp_path_factory):
    """Petite campagne artificielle écrite une fois pour toute la session."""
    folder = tmp_path_factory.mktemp("campagne")
    write_campaign(
        folder,
        materiaux=("alu", "cuivre"),
        maillages=("moyen", "fin"),
        puissances=(5, 20),
        debits=(0.6, 1.5),
        points=60,
    )
    return folder


@pytest.fixture
def dataset(campaign):
    return cs.load(campaign, label="{materiau} {puissance} kW")
