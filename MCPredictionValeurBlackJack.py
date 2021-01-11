# -*- coding: utf-8 -*-
"""
Prédiction de valeur de pi avec méthode Monte Carlo 
Environnement Blackjack
"""
import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = As, Valet,Dame et Roi = 10
cartes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def passer_carte(np_random):
    return np_random.choice(cartes)


def passer_main(np_random):
    return [passer_carte(np_random), passer_carte(np_random)]


def as_utilisable(main):  # Does this main have a usable ace?
    return 1 in main and sum(main) + 10 <= 21


def total_main(main):  # Return current main total
    if as_utilisable(main):
            return sum(main) + 10
    return sum(main)


def a_creve(main):  # Is this main a bust?
    return total_main(main) > 21


def score(main):  # What is the score of this main (0 if bust)
    return 0 if a_creve(main) else total_main(main)


def blackjack_naturel(main):  # Is this main a natural blackjack?
    return sorted(main) == [1, 10]


class BlackjackEnv(gym.Env):
    """ Environnement de Blackjack simplifié selon l'exemple 5.1 de Sutton and Barto (2020)
    Reinforcement Learning: An Introduction.
    Le code est une traduction de https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/blackjack.py
    L'interface suit les conventions de https://gym.openai.com/
   

    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,    Les figures Valet, Dame et Roi ont une valeur de 10
    Un as vaut 1 ou 11
    Par opposition à un jeu de carte réel, on fait l'hypothèse
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self._reset()        # Number of 
        self.nA = 2

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players main and return
            self.player.append(passer_carte(self.np_random))
            if a_creve(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers main, and score
            done = True
            while total_main(self.dealer) < 17:
                self.dealer.append(passer_carte(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and blackjack_naturel(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (total_main(self.player), self.dealer[0], as_utilisable(self.player))

    def _reset(self):
        self.dealer = passer_main(self.np_random)
        self.player = passer_main(self.np_random)

        # Auto-draw another card if the score is less than 12
        while total_main(self.player) < 12:
            self.player.append(passer_carte(self.np_random))

        return self._get_obs()

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 

import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])



def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

matplotlib.style.use('ggplot')
env = BlackjackEnv()

def predire_valeurpi_mc(policy, env, num_episodes, gamma=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    N = defaultdict(float) # Nombre d'observations pour chacun des états (pour calcul incrémental)
    V = defaultdict(float) # Valeur moyenne de récompense pour chacun des états
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

         # Un episode est un tableau de tules (etat, action, recompense)
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        etats_episode = [etape[0] for etape in episode]
        G=0
        for t in range(len(episode)-1,-1,-1):
            G=gamma*G+episode[t][2]
            St = episode[t][0]
            if St not in etats_episode[0:t]:
                N[St]+=1
                V[St]=V[St]+(G-V[St])/N[St]
    return V


def politique_reste_20ou21(observation):
    """
    Politique simple : le joueur reste à 20 ou 21, carte sinon
    """
    score, dealer_score, as_utilisable = observation
    return 0 if score >= 20 else 1

V_10k = predire_valeurpi_mc(politique_arret_20ou21, env, num_episodes=10000)
plot_value_function(V_10k, title="10,000 Steps")

V_500k = predire_valeurpi_mc(politique_arret_20ou21, env, num_episodes=500000)
plot_value_function(V_500k, title="500,000 Steps")


