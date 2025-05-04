# Simulation de Murmuration (Modèle de Vicsek)

Ce projet simule le comportement collectif d'un groupe d'agents (oiseaux) en utilisant un modèle inspiré du modèle de Vicsek. L'objectif est d'observer l'émergence de comportements collectifs, comme la formation de bancs (de poissons) ou de nuées (murmuration), à partir de règles locales simples.

## Fonctionnalités

*   Simulation du mouvement d'agents auto-propulsés avec une vitesse constante.
*   Mise à jour de l'orientation des agents basée sur :
    *   L'alignement avec les voisins dans un rayon `radius_influence`.
    *   Un terme de bruit aléatoire (diffusion angulaire `diffusion`).
    *   Un coefficient de couplage `couplage` contrôlant l'influence de l'alignement.
*   Mécanisme d'évitement des collisions pour empêcher les agents de se superposer (distance minimale `radius_avoid`).
*   Conditions aux bords périodiques (les agents qui sortent d'un côté réapparaissent de l'autre).
*   Calcul et stockage du paramètre d'ordre au cours du temps (avec la fonction `param_ordre`).
*   Visualisation animée de la simulation.
*   Calcul et affichage d'un diagramme de phase du paramètre d'ordre en fonction des coefficients de couplage (`couplage`) et de diffusion (`diffusion`).
*   Utilisation de fonctions vectorisées (NumPy) pour améliorer les performances (`theta_update`, `dont_touch_predator`).

## Modèle

Le modèle simule `n_preys` agents se déplaçant dans une boîte carrée de taille `2 window_width x 2 window_width` avec une vitesse constante `v`. À chaque pas de temps `time_step`, la position et l'angle de chaque agent sont mis à jour :

1.  **Mise à jour de l'angle (`theta_update`)**: L'angle de chaque agent tend à s'aligner avec l'angle moyen de ses voisins (dans un rayon `radius_influence`), avec une force contrôlée par `couplage`, tout en étant soumis à un bruit aléatoire contrôlé par `diffusion`.
2.  **Mise à jour de la position (`pos_update_predator`)**: La position est mise à jour en fonction de la vitesse `v` et du nouvel angle.
3.  **Évitement (`dont_touch_predator`)**: Si des agents sont plus proches que la distance `radius_avoid`, leurs positions sont ajustées pour les séparer.
4.  **Conditions aux bords**: Les agents sont maintenus dans la boîte via des conditions périodiques.

## Paramètres Principaux

*   `time_tot`: Nombre total de pas de temps (frames).
*   `n_preys`: Nombre d'agents (oiseaux).
*   `n_predators`: Nombre de prédateurs.
*   `radius_influence`: Rayon d'influence des proies.
*   `radius_avoid`: Rayon d'évitement des proies pour éviter la superposition des agents et améliorer le réalisme.
*   `radius_predators`: Rayon d'évitement des proies par rapport aux prédateurs.
*   `velocity_prey`: Vitesse des proies.  `velocity_predator`: Vitesse des prédateurs.
*   `time_step`: Pas de temps.
*   `window_wwidth`: Demi-largeur de la boîte de simulation (domaine de `-window_wwidth` à `window_wwidth`).
*   `diffusion`: Coefficient de diffusion angulaire (intensité du bruit).
*   `couplage`: Coefficient de couplage (force de l'alignement).
*   `weight_afraid`: Echelle de peur des proies vis à vis des prédateurs.

## Prérequis

*   Python 3.x
*   NumPy
*   Matplotlib
*   tqdm (pour la barre de progression)
*   IPython / Jupyter Notebook (pour exécuter le fichier `.ipynb`)

Vous pouvez installer les dépendances nécessaires avec pip :
```bash
pip install numpy matplotlib tqdm ipython jupyter
