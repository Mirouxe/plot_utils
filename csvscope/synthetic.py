"""Génération de CSV artificiels pour essayer la librairie et servir d'exemple.

Le cas simulé est un refroidissement de module de puissance : une rampe de charge
provoque un transitoire thermique, dont découlent contrainte mécanique, rendement
et vibrations. Les caractéristiques (matériau, maillage, puissance, débit) sont
inscrites dans le nom du fichier, comme dans une vraie campagne de calculs.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["MATERIALS", "MESHES", "generate_series", "write_campaign", "demo_dataset"]

#: Conductivité relative, inertie thermique et dilatation par matériau.
MATERIALS: dict[str, dict[str, float]] = {
    "alu": {"conduction": 1.00, "inertie": 1.00, "dilatation": 1.00},
    "cuivre": {"conduction": 1.45, "inertie": 1.30, "dilatation": 0.75},
    "composite": {"conduction": 0.62, "inertie": 0.70, "dilatation": 1.45},
}

#: Biais et bruit numérique induits par la finesse du maillage.
MESHES: dict[str, dict[str, float]] = {
    "grossier": {"biais": 1.06, "bruit": 1.8},
    "moyen": {"biais": 1.02, "bruit": 0.9},
    "fin": {"biais": 1.00, "bruit": 0.35},
}

AMBIENT = 22.0


def generate_series(
    materiau: str = "alu",
    maillage: str = "fin",
    puissance: float = 20.0,
    debit: float = 1.0,
    duration: float = 120.0,
    points: int = 400,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Série temporelle d'une configuration, avec unités dans les noms de colonnes."""
    if materiau not in MATERIALS:
        raise KeyError(f"Matériau inconnu : {materiau!r} (parmi {sorted(MATERIALS)})")
    if maillage not in MESHES:
        raise KeyError(f"Maillage inconnu : {maillage!r} (parmi {sorted(MESHES)})")

    rng = np.random.default_rng(seed)
    material = MATERIALS[materiau]
    mesh = MESHES[maillage]
    time = np.linspace(0.0, duration, points)

    # Transitoire du premier ordre, plus un dépassement amorti dû à la régulation.
    gain = 6.4 / (material["conduction"] * debit**0.55)
    delta_final = gain * puissance * mesh["biais"]
    tau = 14.0 * material["inertie"] / debit**0.35
    overshoot = 0.16 * delta_final * np.exp(-time / (2.2 * tau)) * np.sin(
        2 * np.pi * time / (1.7 * tau)
    )
    ramp = 1.0 - np.exp(-time / tau)
    thermal_noise = rng.normal(0.0, 0.09 * mesh["bruit"], points).cumsum() * 0.05
    temperature = AMBIENT + delta_final * ramp + overshoot + thermal_noise

    # Perte de charge : croît avec le débit, se dégrade légèrement à chaud.
    pressure = (
        1.15
        + 0.62 * debit**1.8
        - 0.0021 * (temperature - AMBIENT)
        + rng.normal(0.0, 0.004 * mesh["bruit"], points)
    )

    # Débit mesuré : consigne, dérive lente et bruit de capteur.
    measured_flow = (
        debit
        + 0.02 * debit * np.sin(2 * np.pi * time / 37.0)
        + rng.normal(0.0, 0.006 * mesh["bruit"], points)
    )

    # Contrainte thermomécanique : gradient au début, dilatation à la fin.
    gradient = np.gradient(temperature, time)
    stress = (
        material["dilatation"]
        * (1.35 * (temperature - AMBIENT) + 26.0 * np.clip(gradient, 0, None))
        + rng.normal(0.0, 0.5 * mesh["bruit"], points)
    )

    # Rendement du module, pénalisé par la température de jonction.
    efficiency = (
        96.5
        - 0.052 * (temperature - AMBIENT)
        - 0.28 * puissance / 35.0
        + rng.normal(0.0, 0.05 * mesh["bruit"], points)
    )

    # Vibrations : excitation hydraulique amortie par la montée en température.
    vibration = (
        0.35 * debit**1.4 * (1 + 0.6 * np.exp(-time / 25.0))
        * np.abs(np.sin(2 * np.pi * time / 6.5) + 0.4 * np.sin(2 * np.pi * time / 2.3))
        + rng.normal(0.0, 0.02 * mesh["bruit"], points)
    )

    return pd.DataFrame(
        {
            "temps [s]": np.round(time, 4),
            "temperature [°C]": np.round(temperature, 3),
            "pression [bar]": np.round(pressure, 4),
            "debit_mesure [kg/s]": np.round(measured_flow, 4),
            "contrainte [MPa]": np.round(stress, 3),
            "rendement [%]": np.round(efficiency, 3),
            "vibration [mm/s]": np.round(vibration, 4),
        }
    )


def _filename(style: str, characteristics: Mapping[str, object]) -> str:
    materiau = characteristics["materiau"]
    maillage = characteristics["maillage"]
    puissance = characteristics["puissance"]
    debit = characteristics["debit"]
    flow_token = str(debit).replace(".", "p")
    if style == "kv":
        return (
            f"cas_materiau={materiau}_maillage={maillage}"
            f"_puissance={puissance}_debit={flow_token}.csv"
        )
    if style == "compact":
        return f"essai_{materiau}_{maillage}_P{puissance}_Q{flow_token}.csv"
    raise ValueError("style doit valoir 'kv' ou 'compact'.")


def write_campaign(
    folder: str | Path,
    materiaux: Sequence[str] = ("alu", "cuivre", "composite"),
    maillages: Sequence[str] = ("moyen", "fin"),
    puissances: Sequence[float] = (5, 12, 20, 35),
    debits: Sequence[float] = (0.6, 1.5),
    style: str = "kv",
    duration: float = 120.0,
    points: int = 400,
    seed: int = 12,
    clean: bool = True,
) -> list[Path]:
    """Écrit une campagne complète de CSV artificiels et retourne les chemins créés."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if clean:
        for existing in folder.glob("*.csv"):
            existing.unlink()

    written: list[Path] = []
    combinations = itertools.product(materiaux, maillages, puissances, debits)
    for index, (materiau, maillage, puissance, debit) in enumerate(combinations):
        characteristics = {
            "materiau": materiau,
            "maillage": maillage,
            "puissance": puissance,
            "debit": debit,
        }
        frame = generate_series(
            materiau=materiau,
            maillage=maillage,
            puissance=float(puissance),
            debit=float(debit),
            duration=duration,
            points=points,
            seed=seed + index,
        )
        path = folder / _filename(style, characteristics)
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def demo_dataset(folder: str | Path | None = None, **kwargs):
    """Campagne artificielle déjà chargée en :class:`csvscope.Dataset`.

    Sans dossier, les données sont créées dans un répertoire temporaire : pratique
    pour essayer la librairie en trois lignes.
    """
    from tempfile import mkdtemp

    from .dataset import load

    target = Path(folder) if folder is not None else Path(mkdtemp(prefix="csvscope_demo_"))
    write_campaign(target, **kwargs)
    return load(
        target,
        label="{materiau} · {maillage} · {puissance} kW · {debit} kg/s",
    )
