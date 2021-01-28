
# -*- coding: utf-8 -*-
"""
Optimisation de politique avec méthode SARSA
Politique epsilon-vorace
Environnement WindyGridWorldEnv du livre de Barto&Sutton 2018 Ex.6.5
Le code de l'environnement est tiré de 
https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/windy_gridworld.py
Le reste du code est inspiré de :
https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb

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
from collections import namedtuple
import pandas as pd

def afficher_statistiques(longueur_episode):
    """
    Tiré de https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb
    """
    figure1 = plt.figure(figsize=(10,5))
    plt.plot(longueur_episode)
    plt.xlabel("Épisode")
    plt.ylabel("Longueur épisode")
    plt.title("Longueur épisode par rapport au temps")

    figure2 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(longueur_episode), np.arange(len(longueur_episode)))
    plt.xlabel("Temps")
    plt.ylabel("Épisode")
    plt.title("Épisode par étape de temps")

def politique_optimale_sarsa(env, nombre_episodes, gamma=1.0, alpha= 0.1, epsilon=0.1):
    """
    Prédire la valeur de la politique par la métode de Monte Carlo première visite
        politique: fonction pi
        env: environnement de type OpenAI gym
        nombre_episodes: nombre d'épisodes générés pour les estimations
        gamma: facteur d'escompte des récompenses futures
    Retour:
        V: Dictionnaire(etat:tuple,valeur:float)
    """
    longueur_episode = np.zeros(nombre_episodes)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for i_episode in range(nombre_episodes):
        if i_episode % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode, nombre_episodes), end="")
            sys.stdout.flush()

        etat = env.reset()   
        # Choisir action selon politique e-vorace
        probabilites_actions = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        meilleure_action = np.argmax(Q[etat])
        probabilites_actions[meilleure_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(probabilites_actions)), p=probabilites_actions)
        for t in itertools.count():

            etat_suivant, recompense, final, _ = env.step(action)
            # Choisir action suivante selon politique e-vorace
            probabilites_actions_suivant = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
            meilleure_action_suivante = np.argmax(Q[etat_suivant])
            probabilites_actions_suivant[meilleure_action_suivante] += (1.0 - epsilon)
            action_suivante = np.random.choice(np.arange(len(probabilites_actions_suivant)), p=probabilites_actions_suivant)
            
            cible = recompense+gamma*Q[etat_suivant][action_suivante]
            delta = cible-Q[etat][action]
            Q[etat][action] += alpha*delta
            
            if final:
                longueur_episode[i_episode] = t
                break
            etat = etat_suivant
            action = action_suivante
    return Q,statistiques

env = WindyGridworldEnv()
Q,statistiques = politique_optimale_sarsa(env, nombre_episodes=200,gamma=1.0, alpha= 0.1, epsilon=0.1)
afficher_statistiques(statistiques)
