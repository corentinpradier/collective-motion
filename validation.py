import numpy as np
import matplotlib.pyplot as plt

from functions_v2 import theta_update, param_ordre


def plot_msd_po(
    position: np.ndarray, Nt: int, dt: float, tab_param_ordre: np.ndarray
) -> None:
    """
    Tracé du  MSD et du paramètre d'ordre en fonction du temps.

    Args:
        position (np.ndarray): Tableau des positions (2, N_birds).
        Nt (int): Nombre de pas de temps.
        dt (float): Pas de temps.
        tab_param_ordre (np.ndarray): Tableau des paramètres d'ordre.

    Returns:
        None
    """ 
    # Calcul MSD
    msd = np.zeros(Nt)
    for n in range(Nt):
        for i in range(0, Nt - n):
            msd[n] = msd[n] + (position[0, i + n, 0] - position[0, i, 0]) ** 2
        msd[n] = msd[n] / (Nt - n)

    # Tracé du  MSD
    print(msd[1])
    t = np.arange(Nt) * dt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(t, msd)
    ax[0].set_title("Déplacement quadratique moyen (MSD) en fonction du temps")
    ax[0].set_xlabel("temps")
    ax[0].set_ylabel("Déplacement quadratique moyen")

    ax[1].plot(t[:-1], tab_param_ordre[:-1])
    ax[1].set_title("Paramètre d'ordre en fonction du temps")
    ax[1].set_xlabel("temps")
    ax[1].set_ylabel("paramètre d'ordre")

    plt.show()


def corr_diff(
    position: np.ndarray,
    angle: np.ndarray,
    N_birds: int,
    dt: float,
    R0: float,
    pas: int = 20,
    tps: int = 90,
) -> None:
    """
    Tracé du diagramme de phase en fonction du coefficient de corrélation et du coefficient de diffusion

    Args:
        position (np.ndarray): Position des particules
        angle (np.ndarray): Angle des particules
        N_birds (int): Nombre de particules
        dt (float): Pas de temps
        R0 (float): Rayon d'influence des particules
        pas (int, optional): Grille de pas*pas carreaux. Defaults to 20.
        tps (int, optional): Tracé au temps tps. Defaults to 90.
    Returns:
        None
    """
    # Tracé du coefficient de corrélation en fonction du coefficient de diffusion
    # à l'aide du paramètre d'odre
    N_c = np.linspace(0, 1, pas + 1)  # Discrétisation du coeff de corrélation
    N_Dr = np.linspace(0, 1, pas + 1)  # Discrétisation du coeff de diffusion
    phase_diagram = np.zeros((len(N_c), len(N_Dr)))

    for nc in N_c:
        for ndr in N_Dr:
            new_theta = theta_update(
                position[:, tps, :], angle[tps, :], N_birds, nc, ndr, dt, R0
            )

            p = param_ordre(new_theta, N_birds)

            phase_diagram[int(nc * pas), int(ndr * pas)] = p

    fig, ax = plt.subplots()
    # Color bar
    im = ax.imshow(phase_diagram, vmin=0, vmax=1)
    plt.colorbar(im, label="Paramètre d'ordre")

    x_label_list = ["0", "1"]
    y_label_list = ["0", "1"]
    ax.set_xticks([0, 20])
    ax.set_xticklabels(x_label_list)
    ax.set_yticks([0, 20])
    ax.set_yticklabels(y_label_list)
    plt.xlabel("Coefficient de diffusion")
    plt.ylabel("Coeffictient de corrélation")
    plt.imshow(phase_diagram, origin="lower")
    plt.legend(
        bbox_to_anchor=(0.53, 1.15),
        title=f"{N_birds} oiseaux  ;  frame n° {tps}",
        loc=9,
        ncol=2,
    )
