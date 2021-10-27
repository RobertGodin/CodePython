# -*- coding: utf-8 -*-

"""
Simulation du temps moyen de traitement pour la recherche linéaire dans une liste
"""

def recherche_lineaire(element, liste):
    """ Recherche linéaire. 
        Retourne vrai si element est dans la liste, faux sinon
    """
    for indice in range(len(liste)):
        if liste[indice] == element:
            return True
    return False

import numpy as np
np.random.seed(22) # Pour obtenir des résultats reproductibles avec les données aléatoires
import time    
liste_temps_ecoules_rech_lin =[]
for n in range(100000,1000001,100000):
    liste_n_entiers = np.random.randint(0,10*n,size=n)
    temps_total=0
    for i in range(100):
        un_int = np.random.randint(0,10*n,size=1)
        debut = time.perf_counter()
        recherche_lineaire(un_int[0], liste_n_entiers)
        temps_ecoule = time.perf_counter()-debut
        temps_total=temps_total+temps_ecoule
    temps_moyen = temps_total/100
    liste_temps_ecoules_rech_lin.append(temps_moyen)

import matplotlib.pyplot as plt
x=np.arange(100000,1000001,100000)
plt.plot(x,liste_temps_ecoules_rech_lin, label="recherche linéaire")
plt.title("Temps moyen de la recherche")
plt.legend(loc='upper left')
plt.show()