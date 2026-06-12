# plot_utils

Petit utilitaire Python pour :
- lire un fichier Excel contenant des grandeurs physiques en fonction du temps,
- créer certaines grandeurs dérivées,
- extraire les maxima,
- tracer un graphique araignée (radar) avec les valeurs brutes.

## Installation

```bash
pip install -r requirements.txt
```

## Dépendances

- pandas
- numpy
- matplotlib
- openpyxl

## Usage

```bash
python radar_plot.py mon_fichier.xlsx \
  --columns temperature pression vitesse deformation contrainte energie_proxy ratio_contrainte_deformation \
  --output radar_maxima.png
```

## Grandeurs dérivées déjà prévues

Le script crée automatiquement, si les colonnes existent :

- `energie_proxy = pression * vitesse`
- `ratio_contrainte_deformation = contrainte / deformation` (division protégée)
- `norme_vitesse = sqrt(Ux^2 + Uy^2 + Uz^2)`

## Personnalisation

Modifie la fonction `add_derived_columns()` dans `radar_plot.py` pour ajouter tes propres opérations métier.

## Remarque

Le radar est ici tracé avec les **valeurs brutes**, sans normalisation, conformément au besoin de visualisation directe des maxima.
