# -*- coding: utf-8 -*-
"""
Exemple d'utilisation de CartPole dans l'environnement gym
"""

import gym
env = gym.make('CartPole-v0')
for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Épisode terminé après {} étapes".format(t+1))
            break
env.close()