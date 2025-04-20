# Simulation de Murmuration (Modèle de Vicsek)

Ce projet simule le comportement collectif d'un groupe d'agents (oiseaux) en utilisant un modèle inspiré du modèle de Vicsek. L'objectif est d'observer l'émergence de comportements collectifs, comme la formation de bancs ou de nuées (murmuration), à partir de règles locales simples.

## Fonctionnalités

*   Simulation du mouvement d'agents auto-propulsés avec une vitesse constante.
*   Mise à jour de l'orientation des agents basée sur :
    *   L'alignement avec les voisins dans un rayon `R0`.
    *   Un terme de bruit aléatoire (diffusion angulaire `Dr`).
    *   Un coefficient de couplage `C` contrôlant l'influence de l'alignement.
*   Mécanisme d'évitement des collisions pour empêcher les agents de se superposer (distance minimale `l0`).
*   Conditions aux bords périodiques (les agents qui sortent d'un côté réapparaissent de l'autre).
*   Calcul et stockage du paramètre d'ordre au cours du temps (si la fonction `param_ordre` est correctement utilisée et décommentée).
*   Visualisation animée de la simulation.
*   Calcul et affichage d'un diagramme de phase du paramètre d'ordre en fonction des coefficients de couplage (`C`) et de diffusion (`Dr`).
*   Utilisation de fonctions vectorisées (NumPy) pour améliorer les performances (`theta_update_vectorized`, `dont_touch_vectorized`).

## Modèle

Le modèle simule `N_birds` agents se déplaçant dans une boîte carrée de taille `2L x 2L` avec une vitesse constante `v`. À chaque pas de temps `dt`, la position et l'angle de chaque agent sont mis à jour :

1.  **Mise à jour de l'angle (`theta_update_vectorized`)**: L'angle de chaque agent tend à s'aligner avec l'angle moyen de ses voisins (dans un rayon `R0`), avec une force contrôlée par `C`, tout en étant soumis à un bruit aléatoire contrôlé par `Dr`.
2.  **Mise à jour de la position (`pos_update`)**: La position est mise à jour en fonction de la vitesse `v` et du nouvel angle.
3.  **Évitement (`dont_touch_vectorized`)**: Si des agents sont plus proches que la distance `l0`, leurs positions sont ajustées pour les séparer.
4.  **Conditions aux bords**: Les agents sont maintenus dans la boîte via des conditions périodiques.

## Paramètres Principaux

*   `Nt`: Nombre total de pas de temps (frames).
*   `N_birds`: Nombre d'agents (oiseaux).
*   `L`: Demi-largeur de la boîte de simulation (domaine de `-L` à `L`).
*   `v`: Vitesse constante des agents.
*   `dt`: Pas de temps de la simulation.
*   `R0`: Rayon de voisinage pour l'interaction d'alignement.
*   `C` (`Coeff_couple`): Coefficient de couplage (force de l'alignement).
*   `Dr`: Coefficient de diffusion angulaire (intensité du bruit).
*   `l0`: Distance minimale de séparation entre les agents.

## Prérequis

*   Python 3.x
*   NumPy
*   Matplotlib
*   tqdm (pour la barre de progression)
*   IPython / Jupyter Notebook (pour exécuter le fichier `.ipynb`)

Vous pouvez installer les dépendances nécessaires avec pip :
```bash
pip install numpy matplotlib tqdm ipython jupyter
