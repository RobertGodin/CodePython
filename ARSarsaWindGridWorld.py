# -*- coding: utf-8 -*-
"""
Optimisation de politique avec méthode Monte Carlo première visite
Politique epsilon-vorace
Environnement Blackjack
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import defaultdict
import itertools
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")



import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from collections import namedtuple
import pandas as pd

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3

def afficher_V(V, titre="Fonction valeur de la politique selon méthode de Monte Carlo première visite"):
    """
    Afficher V comme surface en 3D
    
    V : dictionnaire (etat, valeur)
    """
    # Déterminer les quadrillages des axes X et Y
    min_x = min(etat[0] for etat in V.keys()) # axe des x : main du joueur
    max_x = max(etat[0] for etat in V.keys())
    min_y = min(etat[1] for etat in V.keys()) # axe des y : main de la banque
    max_y = max(etat[1] for etat in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Rassembler les valeurs de z pour tous les (x, y) : distinguer les cas avec et sans as utilisable
    Z_sans_as = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_as = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def afficher_surface(X, Y, Z, titre):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.Reds, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Total joueur')
        ax.set_ylabel('Carte visible banque')
        ax.set_zlabel('Valeur')
        ax.set_title(titre)
        ax.view_init(ax.elev, -140)
        fig.colorbar(surface)
        plt.show()

    afficher_surface(X, Y, Z_sans_as, "{} (Sans as utilisable)".format(titre))
    afficher_surface(X, Y, Z_as, "{} (Avec as utilisable)".format(titre))


env = WindyGridworldEnv()


def contruire_politique_epsilon_vorace(Q, epsilon, nb_actions) :
    """
    Creer une fonction qui calcule les probabilités d'une politique e-vorace
    
        Q: dictionnaire etat -> valeurs des actions (np.array de taille nb_actions)
        epsilon: float entre 0 et 1
        nb_actions: nombre d'actions de l'environnement
    
    Retourne une fonction qui prend un etat et retourne les probabilités d'actions e-vorace
    
    """
    def f_politique(etat):
        probabilites_actions = np.ones(nb_actions, dtype=float) * epsilon / nb_actions
        meilleure_action = np.argmax(Q[etat])
        probabilites_actions[meilleure_action] += (1.0 - epsilon)
        return probabilites_actions
    return f_politique

def politique_optimale_sarsa(env, nombre_episodes, gamma=1.0, alpha= 0.1, epsilon=0.1):
    """
    Prédire la valeur de la politique par la métode de Monte Carlo première visite

        politique: fonction pi
        env: environnement de type OpenAI gym
        nombre_episodes: nombre d'épisodes générés pour les estimations
        gamma: facteur d'escompte des récompenses futures
    
    Retour:
        V: Dictionnaire(etat,valeur)
        The etat is a tuple and the value is a float.
    """

    statistiques = EpisodeStats(episode_lengths=np.zeros(nombre_episodes),episode_rewards=np.zeros(nombre_episodes))

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    politique = contruire_politique_epsilon_vorace(Q, epsilon, env.action_space.n)
    
    for i_episode in range(nombre_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, nombre_episodes), end="")

         # Un episode est un tableau de tules (etat, action, recompense)
            
        etat = env.reset()
        probabilites_actions = politique(etat)
        action = np.random.choice(np.arange(len(probabilites_actions)), p=probabilites_actions)
        for t in itertools.count():

            etat_suivant, recompense, final, _ = env.step(action)            
            probabilites_actions_suivant = politique(etat_suivant)
            action_suivante = np.random.choice(np.arange(len(probabilites_actions_suivant)), p=probabilites_actions_suivant)
            
            # Mettre à jour les statistiques
            statistiques.episode_rewards[i_episode] += recompense
            statistiques.episode_lengths[i_episode] = t
            
            cible = recompense+gamma*Q[etat_suivant][action_suivante]
            delta = cible-Q[etat][action]
            Q[etat][action] += alpha*delta
            
            if final:
                break
            etat = etat_suivant
            action = action_suivante
                
    return Q,statistiques

Q,statistiques = politique_optimale_sarsa(env, nombre_episodes=200,gamma=1.0, alpha= 0.5, epsilon=0.1)
plot_episode_stats(statistiques)

# Calculer et afficher la valeur de V calculée à partir de Q
V = defaultdict(float)
somme=0
nb=0
for etat, actions in Q.items():
    V[etat] = np.max(actions)
    somme+=V[etat]
    nb+=1
    
# afficher_V(V, titre="500 000 épisodes")
print("Valeur moyenne:", somme/nb)


        