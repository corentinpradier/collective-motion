import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output


class CollectiveMotion:
    def __init__(
        self,
        time_tot: int,
        n_preys: int = 100,
        n_predators: int = 1,
        radius_influence: float = 5,
        radius_avoid: float = 1,
        radius_predators: float = 5,
        velocity_prey: float = 1,
        velocity_predator=1,
        time_step: float = 1,
        window_width: float = 20,
        diffusion: float = 0.8,
        couplage: float = 0.8,
        weight_afraid: float = 3.0,
    ):
        self.time_tot = time_tot
        self.n_preys = n_preys
        self.n_predators = n_predators
        self.n_tot = self.n_predators + self.n_preys
        self.radius_influence = radius_influence
        self.radius_avoid = radius_avoid
        self.radius_predators = radius_predators
        self.velocity_prey = velocity_prey
        self.velocity_predator = velocity_predator
        self.time_step = time_step
        self.window_width = window_width
        self.diffusion = diffusion
        self.couplage = couplage
        self.weight_afraid = weight_afraid
        self.is_prey, self.is_predator = self.make_mask()
        self.velocities = self._init_velocities()

        if self.n_tot == self.n_predators:
            print("Il n'y a que des prédateurs !")
        elif self.n_tot == self.n_preys:
            print("Il n'y a pas de prédateurs :)")
        else:
            print(f"Il y a {n_preys} proies et {n_predators} prédateurs.")

    def _init_position(self):
        init_position = np.zeros((2, self.time_tot, self.n_tot))
        init_position[0, 0, :] = np.random.uniform(
            -self.window_width, self.window_width, self.n_tot
        )
        init_position[1, 0, :] = np.random.uniform(
            -self.window_width, self.window_width, self.n_tot
        )
        return init_position

    def _init_angle(self):
        angle_history = np.zeros((self.time_tot, self.n_tot))
        init_angle = np.random.uniform(-np.pi, np.pi, self.n_tot)
        angle_history[0, :] = init_angle
        return angle_history, init_angle

    def _init_velocities(self):
        velocities = np.zeros(self.n_tot)
        velocities[self.is_prey] = self.velocity_prey
        velocities[self.is_predator] = self.velocity_predator
        return velocities

    def make_mask(self):
        is_prey = np.arange(self.n_tot) < self.n_preys

        if self.n_tot > self.n_preys:
            is_predator = ~is_prey
        elif self.n_tot == self.n_preys:
            is_predator = None

        return is_prey, is_predator

    def theta_update(self, position, angle):
        if self.n_preys == 0:
            if np.any(self.is_predator):
                angle_new = np.zeros(self.n_tot)
                noise = np.sqrt(2 * self.diffusion * self.time_step) * np.random.normal(
                    0, 1, self.n_tot
                )
                angle_new[self.is_predator] = (
                    angle[self.is_predator] + noise[self.is_predator]
                )
                return angle_new
            else:
                return np.array([])

        angle_new = np.zeros(self.n_tot)

        # Réduction du bruit pour les prédateurs
        noise = np.sqrt(2 * self.diffusion * self.time_step) * np.random.normal(
            0, 1, self.n_tot
        )

        # Séparer proies et prédateurs
        position_prey = position[:, self.is_prey]
        angle_prey = angle[self.is_prey]
        position_predator = position[:, self.is_predator]

        # === PROIES ===
        if self.n_preys > 0:
            fleeing_angle_component_x = np.zeros(self.n_preys)
            fleeing_angle_component_y = np.zeros(self.n_preys)
            fear_weight = np.zeros(self.n_preys)

            if self.n_predators > 0:
                diffs_pred = position_prey[:, :, None] - position_predator[:, None, :]
                dist_sq_pred = np.sum(diffs_pred**2, axis=0)
                predator_mask = dist_sq_pred < self.radius_predators**2

                for prey_idx in range(self.n_preys):
                    nearby_predators = predator_mask[prey_idx]
                    if np.any(nearby_predators):
                        vector_to_predator = -diffs_pred[
                            :, prey_idx, nearby_predators
                        ]  # ??
                        mean_flee_vector = np.mean(vector_to_predator, axis=1)
                        norm = np.linalg.norm(mean_flee_vector)
                        if norm > 0:
                            fleeing_angle_component_x[prey_idx] = (
                                mean_flee_vector[0] / norm
                            )
                            fleeing_angle_component_y[prey_idx] = (
                                mean_flee_vector[1] / norm
                            )
                        else:
                            rand_angle = np.random.uniform(-np.pi, np.pi)
                            fleeing_angle_component_x[prey_idx] = np.cos(rand_angle)
                            fleeing_angle_component_y[prey_idx] = np.sin(rand_angle)

                        mean_dist = np.mean(
                            np.sqrt(dist_sq_pred[prey_idx, nearby_predators])
                        )
                        fear_weight[prey_idx] = np.clip(
                            (self.radius_predators - mean_dist) / self.radius_predators,
                            0,
                            1,
                        )

            diffs_prey = position_prey[:, None, :] - position_prey[:, :, None]
            dist_sq_prey = np.sum(diffs_prey**2, axis=0)
            neighbour_mask_prey = (dist_sq_prey < self.radius_influence**2) & (
                dist_sq_prey > 1e-9
            )
            np.fill_diagonal(
                neighbour_mask_prey, np.inf
            )  # S'assurer qu'on n'est pas son propre voisin

            cos_angle_prey = np.cos(angle_prey)
            sin_angle_prey = np.sin(angle_prey)

            # Définition de la peur (du poids)
            weight_normal = 1.0
            weight_afraid = self.weight_afraid
            neighbour_weight = np.ones(self.n_preys) * weight_normal
            neighbour_weight[fear_weight > 0] = weight_afraid

            weighted_cos = cos_angle_prey * neighbour_weight
            weighted_sin = sin_angle_prey * neighbour_weight

            sx_prey_weighted_sum = neighbour_mask_prey @ weighted_cos
            sy_prey_weighted_sum = neighbour_mask_prey @ weighted_sin

            avg_neighbour_angle_prey_weighted = np.arctan2(
                sy_prey_weighted_sum, sx_prey_weighted_sum
            )

            align_weight = 1.0 - fear_weight

            combined_component_x = (
                align_weight * np.cos(avg_neighbour_angle_prey_weighted)
                + fear_weight * fleeing_angle_component_x
            )
            combined_component_y = (
                align_weight * np.sin(avg_neighbour_angle_prey_weighted)
                + fear_weight * fleeing_angle_component_y
            )

            target_angle_prey = np.arctan2(combined_component_y, combined_component_x)

            angle_new[self.is_prey] = self.couplage * target_angle_prey + (
                1 - self.couplage
            ) * (angle_prey + noise[self.is_prey])

        if np.any(self.is_predator):
            predator_indices = np.where(self.is_predator)[0]
            predator_noise_scale = 0.2
            noise[self.is_predator] *= predator_noise_scale

            for i, predator_idx in enumerate(predator_indices):
                predator_pos = position[:, predator_idx]
                if self.n_preys > 0:
                    vector_to_preys = position_prey - predator_pos[:, None]
                    distances = np.linalg.norm(vector_to_preys, axis=0)

                    detection_radius = self.radius_predators * 1.5
                    nearby_preys_mask = distances < detection_radius

                    if np.any(nearby_preys_mask):
                        closest_prey_idx_local = np.argmin(distances[nearby_preys_mask])

                        nearby_prey_indices_global = np.where(self.is_prey)[0][
                            nearby_preys_mask
                        ]
                        closest_prey_idx_global = nearby_prey_indices_global[
                            closest_prey_idx_local
                        ]

                        vec_to_closest = (
                            position[:, closest_prey_idx_global] - predator_pos
                        )
                        target_angle_pred = np.arctan2(
                            vec_to_closest[1], vec_to_closest[0]
                        )
                        # Mouvement vers la proie + petit bruit
                        angle_new[predator_idx] = (
                            target_angle_pred + noise[predator_idx]
                        )
                    else:
                        # Mouvement aléatoire si aucune proie n'est proche
                        angle_new[predator_idx] = (
                            angle[predator_idx] + noise[predator_idx]
                        )
                else:
                    # Mouvement aléatoire s'il n'y a pas de proies
                    angle_new[predator_idx] = angle[predator_idx] + noise[predator_idx]

        # Normaliser tous les angles finaux (optionnel mais propre)
        angle_new = np.arctan2(np.sin(angle_new), np.cos(angle_new))

        return angle_new

    def dont_touch_predator(self, position):
        if self.n_tot <= 1:
            return position

        diffs = position[:, None, :] - position[:, :, None]
        dist_sq = np.sum(diffs**2, axis=0)

        np.fill_diagonal(dist_sq, np.inf)

        total_corrections = np.zeros_like(position)  # Initialiser les corrections

        if self.n_predators == 0:
            l0 = self.radius_avoid
            l0_sq = l0**2
            mask = (dist_sq < l0_sq) & (dist_sq > 1e-9)
            indices_i, indices_j = np.where(mask)

            if indices_i.size > 0:
                relevant_diffs = diffs[:, indices_i, indices_j]
                relevant_dist = np.sqrt(dist_sq[indices_i, indices_j])

                # Correction symétrique
                correction_magnitude = (l0 - relevant_dist) / 2.0
                correction_vectors = (
                    relevant_diffs
                    / relevant_dist[None, :]
                    * correction_magnitude[None, :]
                )

                np.add.at(
                    total_corrections, (slice(None), indices_j), correction_vectors
                )
                np.add.at(
                    total_corrections, (slice(None), indices_i), -correction_vectors
                )

        else:
            # On identifie les paires trop proches (potentiellement)
            max_radius = max(self.radius_avoid**2, self.radius_predators**2)
            potential_mask = (dist_sq < max_radius) & (dist_sq > 1e-9)
            indices_i, indices_j = np.where(potential_mask)

            if indices_i.size == 0:
                return position

            relevant_diffs = diffs[:, indices_i, indices_j]
            relevant_dist_sq = dist_sq[indices_i, indices_j]
            relevant_dist = np.sqrt(relevant_dist_sq)

            i_is_prey = self.is_prey[indices_i]
            j_is_prey = self.is_prey[indices_j]
            i_is_predator = self.is_predator[indices_i]
            j_is_predator = self.is_predator[indices_j]

            mask_pp = i_is_prey & j_is_prey
            mask_PP = i_is_predator & j_is_predator
            mask_pP = (i_is_prey & j_is_predator) | (i_is_predator & j_is_prey)

            relevant_radius_sq = np.zeros_like(relevant_dist_sq)
            relevant_radius_sq[mask_pp] = self.radius_avoid**2
            relevant_radius_sq[mask_PP] = self.radius_avoid**2
            relevant_radius_sq[mask_pP] = self.radius_predators**2

            relevant_radius = np.sqrt(relevant_radius_sq)

            actual_collision_mask = relevant_dist_sq < relevant_radius_sq

            if not np.any(actual_collision_mask):
                return position

            final_indices_i = indices_i[actual_collision_mask]
            final_indices_j = indices_j[actual_collision_mask]
            final_diffs = relevant_diffs[:, actual_collision_mask]
            final_dist = relevant_dist[actual_collision_mask]
            final_radius = relevant_radius[actual_collision_mask]

            full_correction_magnitude = final_radius - final_dist

            full_correction_vectors = (
                final_diffs / final_dist[None, :] * full_correction_magnitude[None, :]
            )

            total_corrections = np.zeros_like(position)

            final_i_is_prey = self.is_prey[final_indices_i]
            final_j_is_prey = self.is_prey[final_indices_j]
            final_i_is_predator = self.is_predator[final_indices_i]
            final_j_is_predator = self.is_predator[final_indices_j]

            final_mask_pp = final_i_is_prey & final_j_is_prey
            final_mask_PP = final_i_is_predator & final_j_is_predator
            final_mask_pP_i_prey = final_i_is_prey & final_j_is_predator  # Proie est i
            final_mask_pP_j_prey = final_i_is_predator & final_j_is_prey  # Proie est j

            half_correction_vectors = full_correction_vectors / 2.0

            if np.any(final_mask_pp):
                indices_i_pp = final_indices_i[final_mask_pp]
                indices_j_pp = final_indices_j[final_mask_pp]
                vectors_pp = half_correction_vectors[:, final_mask_pp]
                np.add.at(total_corrections, (slice(None), indices_i_pp), vectors_pp)
                np.add.at(total_corrections, (slice(None), indices_j_pp), -vectors_pp)

            if np.any(final_mask_PP):
                indices_i_PP = final_indices_i[final_mask_PP]
                indices_j_PP = final_indices_j[final_mask_PP]
                vectors_PP = half_correction_vectors[:, final_mask_PP]
                np.add.at(total_corrections, (slice(None), indices_j_PP), vectors_PP)
                np.add.at(total_corrections, (slice(None), indices_i_PP), -vectors_PP)

            # c) Corrections asymétriques (Proie(i) - Prédateur(j)) - On pousse seulement i
            if np.any(final_mask_pP_i_prey):
                indices_i_pP = final_indices_i[final_mask_pP_i_prey]
                # Utiliser le vecteur de correction TOTAL
                vectors_pP_i = full_correction_vectors[:, final_mask_pP_i_prey]
                # Pousser la proie i en s'éloignant du prédateur j (direction -vector)
                np.add.at(total_corrections, (slice(None), indices_i_pP), -vectors_pP_i)

            # d) Corrections asymétriques (Prédateur(i) - Proie(j)) - On pousse seulement j
            if np.any(final_mask_pP_j_prey):
                indices_j_Pp = final_indices_j[final_mask_pP_j_prey]
                # Utiliser le vecteur de correction TOTAL
                vectors_Pp_j = full_correction_vectors[:, final_mask_pP_j_prey]
                # Pousser la proie j en s'éloignant du prédateur i (direction +vector)
                np.add.at(total_corrections, (slice(None), indices_j_Pp), vectors_Pp_j)

        position_corrected = position + total_corrections
        return position_corrected

    def pos_update_predator(self, position, angle):
        sx = np.cos(angle)
        sy = np.sin(angle)

        new_position = position.copy()

        new_position[0, :] += self.velocities * self.time_step * sx
        new_position[1, :] += self.velocities * self.time_step * sy

        new_position = self.dont_touch_predator(new_position)

        return new_position

    def main(self):
        position = self._init_position()
        angle_history, angle = self._init_angle()
        angle_history[0, :] = angle
        tab_param_ordre = np.zeros((self.time_tot)) if self.n_predators == 0 else None

        for i in tqdm(range(self.time_tot - 1)):
            current_position = position[:, i, :]
            current_angle = angle

            theta_next_potential = self.theta_update(current_position, current_angle)

            theta_next_potential = np.arctan2(
                np.sin(theta_next_potential), np.cos(theta_next_potential)
            )

            position_next_potential = self.pos_update_predator(
                current_position.copy(), theta_next_potential.copy()
            )

            position_x_next = position_next_potential[0, :]
            position_y_next = position_next_potential[1, :]

            position_x_next[position_x_next > self.window_width] -= (
                2 * self.window_width
            )
            position_x_next[position_x_next < -self.window_width] += (
                2 * self.window_width
            )
            position_y_next[position_y_next > self.window_width] -= (
                2 * self.window_width
            )
            position_y_next[position_y_next < -self.window_width] += (
                2 * self.window_width
            )
            theta_next_final = theta_next_potential

            position[0, i + 1, :] = position_x_next
            position[1, i + 1, :] = position_y_next

            angle = theta_next_final
            angle_history[i + 1, :] = angle

            if self.n_predators == 0:
                tab_param_ordre[i] = self.param_ordre(angle)

        return position, angle_history, tab_param_ordre

    ###################
    ##### Plotting ####
    ###################
    def draw_arrows(
        self,
        ax,
        x,
        y,
        angles,
        length,
        color="tab:blue",
        cmap=None,
        width=0.1,
        head_width=0.3,
        head_length=0.4,
    ):
        from matplotlib.patches import FancyArrow
        import matplotlib.cm as cm

        if cmap is not None:
            cmap = cm.get_cmap(cmap)

            vmin = np.min(angles)
            vmax = np.max(angles)

        for xi, yi, ai in zip(x, y, angles):
            dx = length * np.cos(ai)
            dy = length * np.sin(ai)

            if cmap is not None:
                cmap = cm.get_cmap(cmap)
                vmin = -np.pi
                vmax = np.pi

                color = cmap((ai - vmin) / (vmax - vmin))

            arrow = FancyArrow(
                xi,
                yi,
                dx,
                dy,
                width=width,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                color=color,
                zorder=2,
            )
            ax.add_patch(arrow)

    def plot_frame(
        self,
        position,
        angle_history,
        ax=None,
        prey_color="blue",
        predator_color="red",
        cmap=None,
        frame=0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if frame >= position.shape[1]:
            raise ValueError("Frame index out of bounds.")

        pos_i = position[:, frame, :]
        angle_i = angle_history[frame, :]

        if np.any(self.is_prey):
            self.draw_arrows(
            ax,
            pos_i[0, self.is_prey],
            pos_i[1, self.is_prey],
            angle_i[self.is_prey],
            length=3.0,  # Taille du vecteur
            width=0.5,  # Corps
            head_width=1,  # Tête plus fine
            head_length=2,
            color=prey_color,
            cmap=cmap,
        )

        # Tracer les prédateurs
        if np.any(self.is_predator):
            self.draw_arrows(
            plt.gca(),
            pos_i[0, self.is_predator],
            pos_i[1, self.is_predator],
            angle_i[self.is_predator],
            length=4.5,  # Taille du vecteur
            width=0.5,  # Corps
            head_width=2,  # Tête plus fine
            head_length=3,
            color=predator_color,
        )

        ax.set_xlim(-self.window_width - 1, self.window_width + 1)
        ax.set_ylim(-self.window_width - 1, self.window_width + 1)
        ax.set_title(f"Frame n°{frame}")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )

    def plot_run(
        self,
        position,
        angle_history,
        prey_color="blue",
        predator_color="red",
        cmap=None,
    ):
        plt.figure(figsize=(15, 15))

        for i in range(self.time_tot - 1):
            plt.clf()

            # Positions et angles à l'instant i
            pos_i = position[:, i, :]
            angle_i = angle_history[i, :]

            # Tracer les proies
            if np.any(self.is_prey):
                self.draw_arrows(
                    plt.gca(),
                    pos_i[0, self.is_prey],
                    pos_i[1, self.is_prey],
                    angle_i[self.is_prey],
                    length=3.0,  # Taille du vecteur
                    width=0.5,  # Corps
                    head_width=1,  # Tête plus fine
                    head_length=2,
                    color=prey_color,
                    cmap=cmap
                )

            # Tracer les prédateurs
            if np.any(self.is_predator):
                self.draw_arrows(
                    plt.gca(),
                    pos_i[0, self.is_predator],
                    pos_i[1, self.is_predator],
                    angle_i[self.is_predator],
                    length=4.5,  # Taille du vecteur
                    width=0.5,  # Corps
                    head_width=2,  # Tête plus fine
                    head_length=3,
                    color=predator_color,
                )

            plt.xlim(-self.window_width - 1, self.window_width + 1)
            plt.ylim(-self.window_width - 1, self.window_width + 1)
            plt.title(f"Itération n°{i}")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.draw()
            plt.pause(0.01)
            clear_output(wait=True)
        plt.close()

    ##################
    ### Validation ###
    ##################
    def param_ordre(self, angle: np.ndarray) -> float:
        """
        Calcule le paramètre d'ordre à partir des angles des oiseaux.

        Args:
            theta (np.ndarray): Tableau des angles (N_birds).

        Returns:
            float: Paramètre d'ordre.
        """
        if self.n_preys == 0:
            return np.array([])

        a = np.zeros((self.n_preys, self.n_preys))
        for i in range(self.n_preys):
            for j in range(self.n_preys):
                a[i, j] = 2 * (angle[i] - angle[j])

        return np.mean(np.cos([a]))

    def plot_msd_po(self, position: np.ndarray, tab_param_ordre: np.ndarray) -> None:
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
        msd = np.zeros(self.time_tot)
        for n in range(self.time_tot):
            for i in range(0, self.time_tot - n):
                msd[n] = msd[n] + (position[0, i + n, 0] - position[0, i, 0]) ** 2
            msd[n] = msd[n] / (self.time_tot - n)

        # Tracé du  MSD
        print(msd[1])
        t = np.arange(self.time_tot) * self.time_step

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
        self,
        position: np.ndarray,
        angle: np.ndarray,
        pas: int = 20,
        tps: int = 0,
    ) -> None:
        """
        Tracé du diagramme de phase en fonction du coefficient de corrélation et du coefficient de diffusion.

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
                # On modifie les paramètres de la simulation
                self.couplage = nc
                self.diffusion = ndr

                new_theta = self.theta_update(position[:, tps, :], angle[tps, :])

                p = self.param_ordre(new_theta)

                phase_diagram[int(nc * pas), int(ndr * pas)] = p

        fig, ax = plt.subplots()

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
            title=f"{self.n_preys} oiseaux  ;  frame n° {tps}",
            loc=9,
            ncol=2,
        )
