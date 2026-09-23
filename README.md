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

Comme les critères ont des unités différentes, chaque axe du radar possède **sa propre
échelle** : la géométrie est normalisée pour que toutes les grandeurs soient visibles, mais
les graduations affichées le long de chaque axe sont les **vraies valeurs** du critère
(bornes « rondes », `n_levels` niveaux, défaut 5). Quand la valeur au centre n'est pas 0,
elle est indiquée sous le nom de l'axe.

- `include_zero=True` (défaut) : le zéro est toujours dans l'échelle (centre = 0 pour les
  grandeurs positives), ce qui rend les surfaces comparables.
- `include_zero=False` : chaque échelle est resserrée sur les valeurs observées, pour mieux
  distinguer des trajectoires proches.
- `normalize=False` : toutes les grandeurs partagent un même axe radial en valeurs brutes.

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

Pour obtenir **une seule figure** regroupant toutes les grandeurs dans une grille de
sous-graphes (un sous-graphe par grandeur, légende commune), utilise `compare_time_series_grid` :

```python
from compare_trajectories import compare_time_series_grid

chemin = compare_time_series_grid(
    trajectory_names=["traj_A", "traj_B", "traj_C"],
    quantities=["temperature", "pression", "norme_vitesse"],
    csv_folder="mes_csv",
    time_column="temps",
    output_path="figures/series_grille.png",
    n_cols=2,        # nombre de colonnes de la grille
    sharex=True,     # axe temporel partagé entre sous-graphes
)
```

## Sélection de trajectoires par étude de Pareto (`selection_pareto.py`)

Le script `selection_pareto.py` trouve les `k` trajectoires réalisant le meilleur compromis
entre plusieurs critères (2 à 4). L'entrée est un dossier contenant un CSV par trajectoire
et la liste des critères à compromettre (même syntaxe que `compare_trajectories`).

```python
from selection_pareto import select_trajectories_pareto

resume, classement = select_trajectories_pareto(
    csv_folder="mes_csv",
    criteria=["max(temperature)", "-mean(pression)", "rms(norme_vitesse)"],  # "-" = à minimiser
    k=5,
    method="ideal",          # départage au sein d'un front : ideal | crowding | weighted_sum
    weights=[2, 1, 1],       # poids optionnels (ideal et weighted_sum)
    output_dir="resultats_pareto",
)
print(resume)  # tableau récapitulatif des k trajectoires retenues
```

En ligne de commande (le sens de chaque critère se donne avec `--senses`) :

```bash
python selection_pareto.py mes_csv \
  --criteria "max(temperature)" "mean(pression)" "rms(norme_vitesse)" \
  --senses max min max -k 5 --output-dir resultats_pareto
```

Une note technique détaillant la méthode (équations, choix possibles, limites) est disponible dans
`docs/methode_selection_pareto_2sigma.docx`.

Principe :
1. chaque critère est calculé pour chaque CSV du dossier (nom de trajectoire = nom du fichier) ;
2. les trajectoires sont classées par **fronts de Pareto** successifs (rang 1 = non dominées) ;
3. au sein d'un front, elles sont départagées par `method` : distance au point idéal
   (critères normalisés, défaut), distance de crowding NSGA-II (favorise la diversité) ou
   somme pondérée ;
4. les `k` premières du classement sont sélectionnées.

### Filtre des scénarios extrêmes

Pour ne retenir que les scénarios réalistes, un filtre optionnel écarte, **avant** l'étude de
Pareto, les trajectoires dont un critère dépasse un seuil de sa distribution :

```python
resume, classement = select_trajectories_pareto(
    csv_folder="mes_csv",
    criteria=["max(temperature)", "-mean(pression)"],
    k=5,
    outlier_quantile=0.95,                    # ou outlier_sigma=2.0 (moyenne + 2σ)
    outlier_side="upper",                     # "upper" (défaut), "lower" ou "both"
    outlier_criteria=["max(temperature)"],    # optionnel : critères soumis au filtre (défaut : tous)
)
```

```bash
python selection_pareto.py mes_csv --criteria "max(temperature)" "mean(pression)" \
  --senses max min -k 5 --outlier-sigma 2
```

- `outlier_quantile=0.95` : seuil = quantile empirique 95 % de chaque critère ; écarte donc
  toujours ~5 % des trajectoires par critère, quelle que soit la forme de la distribution.
- `outlier_sigma=2.0` : seuil = moyenne + 2 écarts-types ; s'adapte à la dispersion mais des
  valeurs très extrêmes gonflent l'écart-type et peuvent en masquer d'autres.
- `outlier_criteria` accepte aussi des critères qui ne sont pas des objectifs (ils sont alors
  calculés uniquement pour le filtre).

Une trajectoire est exclue dès qu'un critère filtré sort du seuil. Les exclues sont listées
avec leur motif dans `trajectoires_exclues.csv` et apparaissent en croix orange sur les nuages
de points, où les seuils sont tracés en pointillés.

Sorties écrites dans `output_dir` :
- `classement_pareto.csv` : toutes les trajectoires avec valeurs des critères, `rang_pareto`,
  `distance_ideal`, `crowding`, `score_pondere`, `ordre` et `selectionnee` ;
- `trajectoires_exclues.csv` : trajectoires écartées par le filtre (si activé), avec le motif ;
- `trajectoires_selectionnees.csv` et `tableau_selection.png` : récapitulatif des `k` retenues ;
- `pareto_paires.png` : nuages critère contre critère (front et sélection mis en évidence) ;
- `pareto_3d.png` : nuage 3D (uniquement pour 3 critères) ;
- `coordonnees_paralleles.png` : profil normalisé (1 = meilleur) de chaque trajectoire ;
- `radar_selection.png` : radar des trajectoires retenues (à partir de 3 critères).

## Personnalisation

Modifie la fonction `add_derived_columns()` dans `radar_plot.py` pour ajouter tes propres opérations métier.

## Remarque

Le radar est ici tracé avec les **valeurs brutes**, sans normalisation, conformément au besoin de visualisation directe des maxima.
Sous hypothèse gaussienne, la règle usuelle est environ **68% dans ±1σ** et **95% dans ±2σ** autour de la moyenne.
