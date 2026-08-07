import pytest

from csvscope import naming


def test_parse_value_types():
    assert naming.parse_value("12") == 12
    assert naming.parse_value("1.5") == 1.5
    assert naming.parse_value("1e6") == 1e6
    assert naming.parse_value("-3") == -3
    assert naming.parse_value("0p25") == 0.25
    assert naming.parse_value("true") is True
    assert naming.parse_value("alu") == "alu"


def test_parse_value_keeps_p_when_disabled():
    assert naming.parse_value("0p25", decimal_p=False) == "0p25"


def test_parse_auto_key_value():
    result = naming.parse_auto("cas_materiau=alu_puissance=20_debit=1p5")
    assert result == {"materiau": "alu", "puissance": 20, "debit": 1.5, "tag1": "cas"}


def test_parse_auto_glued_tokens_and_units():
    result = naming.parse_auto("essai_P35_Q1p5_alpha15deg")
    assert result["P"] == 35
    assert result["Q"] == 1.5
    assert result["alpha"] == 15
    assert result["alpha_unit"] == "deg"


def test_parse_auto_unknown_tokens_are_numbered():
    result = naming.parse_auto("run_alu_fin")
    assert result == {"tag1": "run", "tag2": "alu", "tag3": "fin"}


def test_template_to_regex_reads_fields():
    parser = naming.make_parser(template="essai_{materiau}_{maillage}_P{puissance}_Q{debit}")
    assert parser("essai_alu_fin_P20_Q1p5.csv") == {
        "materiau": "alu",
        "maillage": "fin",
        "puissance": 20,
        "debit": 1.5,
    }


def test_template_without_field_is_rejected():
    with pytest.raises(ValueError):
        naming.template_to_regex("essai_fixe")


def test_regex_parser_with_named_groups():
    parser = naming.make_parser(regex=r"^(?P<materiau>[a-z]+)-(?P<puissance>\d+)kW$")
    assert parser("cuivre-35kW.csv") == {"materiau": "cuivre", "puissance": 35}


def test_strict_mode_rejects_unmatched_name():
    parser = naming.make_parser(template="essai_{materiau}", strict=True)
    with pytest.raises(ValueError, match="ne correspond pas"):
        parser("autre_chose_encore.csv")


def test_non_strict_mode_falls_back_on_auto_detection():
    parser = naming.make_parser(template="essai_{materiau}_{maillage}")
    assert parser("cas_puissance=20.csv") == {"tag1": "cas", "puissance": 20}


def test_rename_and_casters_are_applied():
    parser = naming.make_parser(
        template="essai_{mat}_P{p}",
        rename={"mat": "materiau", "p": "puissance"},
        casters={"p": float},
    )
    assert parser("essai_alu_P20.csv") == {"materiau": "alu", "puissance": 20.0}


def test_custom_parser_is_used():
    parser = naming.make_parser(parser=lambda stem: {"nom": stem.upper()})
    assert parser("/tmp/abc.csv") == {"nom": "ABC"}


def test_several_strategies_at_once_are_rejected():
    with pytest.raises(ValueError, match="une seule stratégie"):
        naming.make_parser(template="{a}", regex="(?P<a>.+)")


def test_build_label_default_and_template():
    characteristics = {"materiau": "alu", "puissance": 20.0}
    assert naming.build_label(characteristics) == "materiau=alu · puissance=20"
    assert (
        naming.build_label(characteristics, template="{materiau} — {puissance} kW")
        == "alu — 20.0 kW"
    )
