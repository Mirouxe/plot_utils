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

## Comparaison de plusieurs trajectoires (`compare_trajectories.py`)

Le script `compare_trajectories.py` regroupe les outils de comparaison de trajectoires.
Chaque trajectoire correspond à un CSV du dossier dont le **nom de fichier contient le nom
de la trajectoire**. Si plusieurs fichiers contiennent ce nom, celui dont le nom (sans
extension) est exactement ce nom est retenu ; sinon une erreur liste les fichiers ambigus.

### Radar de critères scalaires

La fonction `compare_trajectories` compare plusieurs trajectoires sur des critères choisis :
- un critère est un **opérateur statistique appliqué à une grandeur** du CSV,
- un radar comparatif (une courbe par trajectoire, un axe par critère) est sauvegardé.

```python
from compare_trajectories import compare_trajectories

table = compare_trajectories(
    trajectory_names=["traj_A", "traj_B", "traj_C"],
    criteria=[
        ("max", "temperature"),   # forme tuple
        "mean(pression)",         # forme operateur(colonne)
        "integral:vitesse",       # forme operateur:colonne
        "rms(norme_vitesse)",     # les grandeurs dérivées sont disponibles
    ],
    csv_folder="mes_csv",
    time_column="temps",
    output_path="figures/radar_trajectoires.png",
)
print(table)  # lignes = trajectoires, colonnes = critères (valeurs brutes)
```

Opérateurs disponibles (`CRITERION_OPERATORS`) : `mean`, `max`, `min`, `abs_max`, `median`,
`std`, `range`, `rms`, `final` (dernière valeur) et `integral` (intégrale temporelle par la
méthode des trapèzes sur la colonne `time_column`). Tu peux en ajouter en enrichissant le
dictionnaire `CRITERION_OPERATORS` avec une fonction `(df, colonne, colonne_temps) -> float`.

Comme les critères ont des unités différentes, chaque axe du radar est par défaut normalisé
par le maximum absolu observé entre trajectoires (valeur de référence indiquée sur l'axe).
Passe `normalize=False` pour tracer les valeurs brutes.

### Séries temporelles complètes

La fonction `compare_time_series` suit la même logique, mais compare les grandeurs sur
**toute leur évolution temporelle** et non sur un critère scalaire : pour chaque grandeur
demandée, un visuel superposant les courbes de toutes les trajectoires est sauvegardé
(un fichier par grandeur dans `output_dir`).

```python
from compare_trajectories import compare_time_series

figures = compare_time_series(
    trajectory_names=["traj_A", "traj_B", "traj_C"],
    quantities=["temperature", "pression", "norme_vitesse"],
    csv_folder="mes_csv",
    time_column="temps",
    output_dir="figures/series",
)
print(figures)  # {"temperature": "figures/series/temperature.png", ...}
```

## Personnalisation

Modifie la fonction `add_derived_columns()` dans `radar_plot.py` pour ajouter tes propres opérations métier.

## Remarque

Le radar est ici tracé avec les **valeurs brutes**, sans normalisation, conformément au besoin de visualisation directe des maxima.
Sous hypothèse gaussienne, la règle usuelle est environ **68% dans ±1σ** et **95% dans ±2σ** autour de la moyenne.
