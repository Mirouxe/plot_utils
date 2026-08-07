"""Extraction des caractéristiques d'une configuration depuis le nom du fichier.

Quatre stratégies sont disponibles, de la plus automatique à la plus explicite :

- ``auto``     : détection des motifs ``cle=valeur``, ``cle-valeur`` et ``cleValeur``
- ``template`` : gabarit lisible du style ``"run_{materiau}_P{puissance}_{maillage}"``
- ``regex``    : expression régulière à groupes nommés
- ``parser``   : fonction Python ``nom_de_fichier -> dict``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

__all__ = [
    "parse_value",
    "parse_auto",
    "template_to_regex",
    "make_parser",
    "build_label",
]

# ``cle=valeur``, ``cle:valeur`` ou ``cle-valeur`` (valeur éventuellement non numérique)
_KV_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9]*)\s*[=:]\s*(?P<value>.+)$")

# ``P25``, ``alpha15deg``, ``re1e6``, ``dt0p01`` : préfixe alphabétique puis nombre puis unité
_GLUED_RE = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9]*?)"
    r"(?P<value>[-+]?\d+(?:[.p]\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<unit>[A-Za-z%°/]*)$"
)

_NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")

_BOOLEANS = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "on": True,
    "off": False,
}


def parse_value(raw: str, decimal_p: bool = True) -> object:
    """Convertit une chaîne extraite d'un nom de fichier en type Python utile.

    ``decimal_p`` gère la convention fréquente où le point décimal est remplacé
    par un ``p`` dans les noms de fichiers (``dt0p01`` -> ``0.01``).
    """
    text = str(raw).strip()
    if not text:
        return text

    lowered = text.lower()
    if lowered in _BOOLEANS:
        return _BOOLEANS[lowered]

    candidate = text
    if decimal_p and re.fullmatch(r"[-+]?\d+p\d+", lowered):
        candidate = lowered.replace("p", ".")

    if _NUMBER_RE.match(candidate):
        as_float = float(candidate)
        if "." not in candidate and "e" not in candidate.lower():
            return int(candidate)
        return as_float

    return text


def parse_auto(
    stem: str,
    separators: str = "_ ",
    decimal_p: bool = True,
    keep_unknown: bool = True,
) -> dict[str, object]:
    """Détecte les caractéristiques sans configuration préalable.

    Les jetons non reconnus sont conservés sous les clés ``tag1``, ``tag2``, ...
    afin de rester renommables par l'utilisateur.
    """
    pattern = "[" + re.escape(separators) + "]+"
    tokens = [tok for tok in re.split(pattern, stem) if tok]

    result: dict[str, object] = {}
    unknown_index = 0

    for token in tokens:
        kv = _KV_RE.match(token)
        if kv:
            result[kv.group("key")] = parse_value(kv.group("value"), decimal_p)
            continue

        glued = _GLUED_RE.match(token)
        if glued:
            result[glued.group("key")] = parse_value(glued.group("value"), decimal_p)
            unit = glued.group("unit")
            if unit:
                result.setdefault(f"{glued.group('key')}_unit", unit)
            continue

        if keep_unknown:
            unknown_index += 1
            result[f"tag{unknown_index}"] = parse_value(token, decimal_p)

    return result


def template_to_regex(template: str) -> re.Pattern[str]:
    """Traduit un gabarit ``"run_{materiau}_P{puissance}"`` en expression régulière.

    Les champs ne peuvent pas contenir les séparateurs littéraux qui les suivent,
    ce qui rend la lecture non ambiguë sans avoir à écrire de regex.
    """
    parts: list[str] = []
    position = 0
    field_re = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
    matches = list(field_re.finditer(template))
    if not matches:
        raise ValueError(
            f"Le gabarit {template!r} ne contient aucun champ du style {{nom}}."
        )

    for index, match in enumerate(matches):
        parts.append(re.escape(template[position : match.start()]))

        following = template[match.end() :]
        next_literal = following[0] if following else ""
        is_last = index == len(matches) - 1

        if is_last and not following:
            body = r".+"
        elif next_literal:
            body = f"[^{re.escape(next_literal)}]+"
        else:
            body = r".+?"

        parts.append(f"(?P<{match.group(1)}>{body})")
        position = match.end()

    parts.append(re.escape(template[position:]))
    return re.compile("^" + "".join(parts) + "$")


def make_parser(
    template: str | None = None,
    regex: str | re.Pattern[str] | None = None,
    parser: Callable[[str], Mapping[str, object]] | None = None,
    rename: Mapping[str, str] | None = None,
    casters: Mapping[str, Callable[[str], object]] | None = None,
    separators: str = "_ ",
    decimal_p: bool = True,
    strict: bool = False,
) -> Callable[[str | Path], dict[str, object]]:
    """Construit la fonction qui transforme un chemin de fichier en dictionnaire.

    ``strict`` fait échouer la lecture d'un fichier dont le nom ne correspond pas
    au gabarit ou à la regex, au lieu de retomber sur la détection automatique.
    """
    if sum(x is not None for x in (template, regex, parser)) > 1:
        raise ValueError("Choisis une seule stratégie : template, regex ou parser.")

    compiled: re.Pattern[str] | None = None
    if template is not None:
        compiled = template_to_regex(template)
    elif regex is not None:
        compiled = re.compile(regex) if isinstance(regex, str) else regex

    def _parse(path: str | Path) -> dict[str, object]:
        stem = Path(path).stem

        if parser is not None:
            raw = dict(parser(stem))
        elif compiled is not None:
            match = compiled.search(stem)
            if match is None:
                if strict:
                    raise ValueError(
                        f"Le nom {stem!r} ne correspond pas au motif attendu "
                        f"({compiled.pattern})."
                    )
                raw = parse_auto(stem, separators, decimal_p)
            else:
                raw = {
                    key: parse_value(value, decimal_p)
                    for key, value in match.groupdict().items()
                    if value is not None
                }
        else:
            raw = parse_auto(stem, separators, decimal_p)

        if casters:
            for key, caster in casters.items():
                if key in raw:
                    raw[key] = caster(raw[key])

        if rename:
            raw = {rename.get(key, key): value for key, value in raw.items()}

        return raw

    return _parse


def build_label(
    characteristics: Mapping[str, object],
    keys: Sequence[str] | None = None,
    template: str | None = None,
    separator: str = " · ",
) -> str:
    """Construit l'étiquette lisible d'une configuration (légende, survol, titre)."""
    if template is not None:
        return template.format(**characteristics)

    selected = list(keys) if keys is not None else list(characteristics)
    chunks = []
    for key in selected:
        if key not in characteristics:
            continue
        value = characteristics[key]
        chunks.append(f"{key}={_format_scalar(value)}")
    return separator.join(chunks)


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:g}"
    return str(value)
