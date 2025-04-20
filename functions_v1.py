import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output


# Définition du nouvel angle
def theta_update(position, angle, N_birds, C, D_r, dt, R0):
    angle_new = np.zeros(N_birds)
    for i in range(N_birds):
        s = np.zeros(2)
        for j in range(N_birds):
            if (position[0,i] - position[0,j])**2 + (position[1,i] - position[1,j])**2 < R0**2:
                s[0] += np.cos(angle[j])
                s[1] += np.sin(angle[j])
        angle_new[i] = C*np.arctan2(s[1], s[0]) + (1-C)*(angle[i]+np.sqrt(2*D_r*dt)*np.random.normal(0,1))
    return angle_new


# Mise à jour de la position
def pos_update(pos, angle, v, dt):
    sx = np.cos(angle)
    sy = np.sin(angle)
    return (pos[0,:] + v*dt*sx, pos[1,:] + v*dt*sy)


def plot_run(Nt, N_birds, L, position, angle):
    plt.figure()
    for i in range(Nt-1):
        plt.clf()  # Pour effacer la figure précédente à chaque itération
        for Nb in range(N_birds-1):
            plt.arrow(position[0,i,Nb], position[1,i,Nb], np.cos(angle[i,Nb]), np.sin(angle[i, Nb]), width=0.05, head_width=0.5, color='blue')
            plt.title(f'Itération n°{i}')
        plt.arrow(position[0,i,N_birds-1], position[1,i,N_birds-1], np.cos(angle[i,N_birds-1]), np.sin(angle[i, N_birds-1]), width=0.05, head_width=0.5, color='tab:red')

        plt.xlim(-L, L)
        plt.ylim(-L, L) 
        plt.xlabel('temps '+ str(i))  
        plt.draw()  # Mettre à jour la figure
        plt.pause(0.01)  # Mettre une pause pour voir chaque image pendant 0.1 seconde
        clear_output(wait=True)  # Effacer la sortie de la cellule
    plt.close()


def main(Nt, N_birds, L, Coeff_couple, R0, Dr, v, dt):
    # Définitions des tableaux et conditions initiales
    position = np.zeros((2,Nt,N_birds))
    position[0,0,:] = np.random.uniform(-L,L, N_birds)
    position[1,0,:] = np.random.uniform(-L,L, N_birds)
    tab_param_ordre = np.zeros((Nt))
    angle = np.zeros((Nt, N_birds))
    theta =  np.sqrt(2*Dr*dt)*np.random.uniform(-np.pi, np.pi, N_birds) 

    for i in tqdm(range(Nt-1)):
            theta = theta_update(position[:,i,:], theta, N_birds, Coeff_couple, Dr, dt, R0)
            position[:,i+1,:] = pos_update(position[:,i,:], theta, v, dt)

            # Conditions aux bords
            position[0,i+1,:][position[0,i+1,:] > L] -= 2*L
            position[0,i+1,:][position[0,i+1,:] <-L] += 2*L    
            position[1,i+1,:][position[1,i+1,:] > L] -= 2*L
            position[1,i+1,:][position[1,i+1,:] <-L] += 2*L
            
            angle[i,:] = theta
            
            # tab_param_ordre[i] = param_ordre(angle[i,:])
    return position, angle, tab_param_ordre