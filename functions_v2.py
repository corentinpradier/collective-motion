import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

# Definition du paramètre d'ordre
def param_ordre(theta: np.ndarray, N_birds: int) -> float:
    """
    Calcule le paramètre d'ordre à partir des angles des oiseaux.

    Args:
        theta (np.ndarray): Tableau des angles (N_birds).
        N_birds (int): Nombre d'oiseaux.

    Returns:
        float: Paramètre d'ordre.
    """
    if N_birds == 0:
        return np.array([])

    a = np.zeros((N_birds, N_birds))
    for i in range(N_birds):
        for j in range(N_birds):
            a[i, j] = 2 * (theta[i] - theta[j])
    return np.mean(np.cos([a]))

def theta_update(
    position: np.ndarray,
    angle: np.ndarray,
    N_birds: int,
    C: float,
    D_r: float,
    dt: float,
    R0: float,
) -> np.ndarray:
    """
    Mise à jour des angles des oiseaux.

    Args:
        position (np.ndarray): Tableau des positions (2, N_birds).
        angle (np.ndarray): Tableau des angles (N_birds).
        N_birds (int): Nombre d'oiseaux.
        C (float): Coefficient de couplage.
        D_r (float): Coefficient de diffusion.
        dt (float): Pas de temps.
        R0 (float): Rayon de couplage.

    Returns:
        np.ndarray: Tableau des angles mis à jour.
    """
    N_birds = position.shape[1]

    if N_birds == 0:
        return np.array([])

    diffs = position[:, None, :] - position[:, :, None]
    dist_sq = np.sum(diffs**2, axis=0)

    neighbour_mask = dist_sq < R0**2

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    s_x = neighbour_mask.astype(float) @ cos_angle
    s_y = neighbour_mask.astype(float) @ sin_angle

    avg_neighbour_angle = np.arctan2(s_y, s_x)

    noise = np.sqrt(2 * D_r * dt) * np.random.normal(0, 1, size=N_birds)

    angle_new = C * avg_neighbour_angle + (1 - C) * (angle + noise)

    return angle_new


def dont_touch(pos: np.ndarray, l0: float) -> np.ndarray:
    """
    Ajuste les positions pour empêcher les oiseaux de se superposer (version vectorisée).

    Args:
        pos (np.ndarray): Tableau des positions (2, N_birds).
        l0 (float): Distance minimale de séparation.

    Returns:
        np.ndarray: Tableau des positions ajustées.
    """
    N_birds = pos.shape[1]
    if N_birds <= 1:
        return pos

    diffs = pos[:, None, :] - pos[:, :, None]
    dist_sq = np.sum(diffs**2, axis=0)

    mask = (dist_sq < l0**2) & (dist_sq > 1e-9)

    # indices des paires trop  proches
    indices_i, indices_j = np.where(mask)

    if indices_i.size == 0:
        return pos

    relevant_diffs = diffs[:, indices_i, indices_j]
    relevant_dist_sq = dist_sq[indices_i, indices_j]
    relevant_dist = np.sqrt(relevant_dist_sq)

    # la moitié de ce qu'il faut atteindre pour atteindre l0
    correction_magnitude = (l0 - relevant_dist) / 2.0

    # Applique la direction + la magnitude
    correction_vectors = (
        relevant_diffs / relevant_dist[None, :] * correction_magnitude[None, :]
    )

    total_corrections = np.zeros_like(pos)

    np.add.at(total_corrections, (slice(None), indices_j), correction_vectors)
    np.add.at(total_corrections, (slice(None), indices_i), -correction_vectors)

    pos_corrected = pos + total_corrections

    return pos_corrected


def pos_update(
    pos: np.ndarray, angle: np.ndarray, vitesse: float, dt: float, l0: float
) -> np.ndarray:
    """
    Mise à jour des positions des oiseaux.

    Args:
        pos (np.ndarray): Tableau des positions (2, N_birds).
        angle (np.ndarray): Tableau des angles (N_birds).
        vitesse (float): Vitesse des oiseaux.
        dt (float): Pas de temps.
        l0 (float): Distance minimale de séparation.

    Returns:
        np.ndarray: Tableau des positions mis à jour.
    """
    sx = np.cos(angle)
    sy = np.sin(angle)

    new_pos = pos.copy()

    new_pos[0, :] += vitesse * dt * sx
    new_pos[1, :] += vitesse * dt * sy

    new_pos = dont_touch(new_pos, l0)

    return new_pos


def boundary_conditions(pos: np.ndarray, L: float, i: int) -> np.ndarray:
    """
    Applique les conditions aux limites de la simulation.

    Args:
        pos (np.ndarray): Tableau des positions (2, N_birds).
        L (float): Largeur de la fenêtre.
        i (int): Numéro de l'iteration.

    Returns:
        np.ndarray: Tableau des positions mis à jour.
    """
    pos[0, i + 1, :][pos[0, i + 1, :] > L] -= 2 * L
    pos[0, i + 1, :][pos[0, i + 1, :] < -L] += 2 * L
    pos[1, i + 1, :][pos[1, i + 1, :] > L] -= 2 * L
    pos[1, i + 1, :][pos[1, i + 1, :] < -L] += 2 * L

    return pos


def main(
    n_steps: int,
    n_birds: int,
    L: float,
    couplage: float,
    r0: float,
    dr: float,
    vitesse: float,
    dt: float,
    l0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fonction principale du programme.

    Args:
        n_steps (int): Nombre d'iterations.
        n_birds (int): Nombre d'oiseaux.
        L (float): Longueur de la fenetre.
        couplage (float): Couplage entre les oiseaux.
        r0 (float): Distance de couplage.
        dr (float): Coefficient de diffusion.
        vitesse (float): Vitesse des oiseaux.
        dt (float): Pas de temps.
        l0 (float): Distance minimale de séparation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - position (np.ndarray): Tableau des positions (2, n_steps, n_birds).
            - angle (np.ndarray): Tableau des angles (n_steps, n_birds).
            - tab_param_ordre (np.ndarray): Tableau des paramètres d'ordre (n_steps).
    """
    # Définitions des tableaux et conditions initiales
    position = np.zeros((2, n_steps, n_birds))
    position[0, 0, :] = np.random.uniform(-L, L, n_birds)
    position[1, 0, :] = np.random.uniform(-L, L, n_birds)
    tab_param_ordre = np.zeros((n_steps))

    angle = np.zeros((n_steps, n_birds))
    theta = np.sqrt(2 * dr * dt) * np.random.uniform(-np.pi, np.pi, n_birds)

    for i in tqdm(range(n_steps - 1)):
        theta = theta_update(position[:, i, :], theta, n_birds, couplage, dr, dt, r0)
        position[:, i + 1, :] = pos_update(position[:, i, :], theta, vitesse, dt, l0)

        # Conditions aux bords
        position = boundary_conditions(position, L, i)

        angle[i, :] = theta

        tab_param_ordre[i] = param_ordre(angle[i, :], n_birds)
    return position, angle, tab_param_ordre


def plot_run(
    Nt: int, N_birds: int, L: float, position: np.ndarray, angle: np.ndarray
) -> None:
    """
    Tracé de la simulation.

    Args:
        Nt (int): Nombre d'iterations.
        N_birds (int): Nombre d'oiseaux.
        L (float): Largeur de la fenêtre.
        position (np.ndarray): Tableau des positions (2, Nt, N_birds).
        angle (np.ndarray): Tableau des angles (Nt, N_birds).

    Returns:
        None
    """
    plt.figure()
    for i in range(Nt - 1):
        plt.clf()  # Pour effacer la figure précédente à chaque itération
        for Nb in range(N_birds - 1):
            plt.arrow(
                position[0, i, Nb],
                position[1, i, Nb],
                np.cos(angle[i, Nb]),
                np.sin(angle[i, Nb]),
                width=0.05,
                head_width=0.5,
                color="blue",
            )
            plt.title(f"Itération n°{i}")
        plt.arrow(
            position[0, i, N_birds - 1],
            position[1, i, N_birds - 1],
            np.cos(angle[i, N_birds - 1]),
            np.sin(angle[i, N_birds - 1]),
            width=0.05,
            head_width=0.5,
            color="tab:red",
        )

        plt.xlim(-L, L)
        plt.ylim(-L, L)
        plt.xlabel("temps " + str(i))
        plt.draw()  # Mettre à jour la figure
        plt.pause(0.01)  # Mettre une pause pour voir chaque image pendant 0.1 seconde
        clear_output(wait=True)  # Effacer la sortie de la cellule
    plt.close()
