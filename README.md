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
- plotly

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

## Fonction réutilisable

Tu peux aussi importer directement la fonction dans ton code :

```python
from radar_plot import plot_radar_from_df

max_values = plot_radar_from_df(
    df,
    columns=["temperature", "pression", "vitesse", "contrainte"],
    output_path="figures/radar.png",
    title="Maxima des grandeurs"
)
```

## Courbes interactives multi-CSV

Le module contient aussi une fonction Plotly pour afficher une grandeur en fonction d'une autre sur tous les CSV d'un dossier, avec survol interactif et transparence des courbes.

```python
from radar_plot import plot_csv_curves_interactive

fig = plot_csv_curves_interactive(
    folder_path="mes_csv",
    x_column="temps",
    y_column="temperature",
    opacity=0.35,
    highlight_opacity=1.0,
    highlight_line_width=4,
    output_html="courbes_interactives.html",
)
```

Au survol, tu vois :
- le nom du CSV,
- la coordonnée en abscisse,
- la coordonnée en ordonnée.

Si `output_html` est fourni, le HTML généré permet aussi :
- de **mettre en surbrillance** la courbe survolée,
- de **verrouiller** une courbe par clic,
- de **déverrouiller** par un second clic.

## Analyse statistique d'un ensemble de CSV

Le module contient aussi une fonction pour :
- lire plusieurs CSV,
- calculer le max d'une grandeur par configuration,
- tracer la distribution,
- afficher moyenne, quantiles et seuils en sigma,
- récupérer le point le plus haut entre `mean + 1σ` et `mean + 2σ`.

```python
from radar_plot import analyze_max_distribution

results = analyze_max_distribution(
    csv_folder="mes_csv",
    column_name="temperature",
    pattern="*.csv",
    plot=True,
)

print(results["highest_point_between_1sigma_2sigma"])
```

## Personnalisation

Modifie la fonction `add_derived_columns()` dans `radar_plot.py` pour ajouter tes propres opérations métier.

## Remarque

Le radar est ici tracé avec les **valeurs brutes**, sans normalisation, conformément au besoin de visualisation directe des maxima.
Sous hypothèse gaussienne, la règle usuelle est environ **68% dans ±1σ** et **95% dans ±2σ** autour de la moyenne.
