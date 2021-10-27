# -*- coding: utf-8 -*-

"""
Simulation du temps moyen de traitement pour la recherche dans une liste
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

def recherche_lineaire_triee(element, liste_triee):
    """ Recherche linéaire améloirée. Suppose que la liste est triée.
        Retourne vrai si element est dans la liste, faux sinon
    """
    for indice in range(len(liste_triee)):
        if liste_triee[indice] == element:
            return True
        if liste_triee[indice] > element:
            return False
    return False

import numpy as np
np.random.seed(22) # Pour obtenir des résultats reproductibles avec les données aléatoires
import time    
liste_temps_ecoules_rech_lin_triee =[]
for n in range(100000,1000001,100000):
    liste_n_entiers = np.random.randint(0,10*n,size=n)
    liste_n_entiers.sort()
    temps_total=0
    for i in range(100):
        un_int = np.random.randint(0,10*n,size=1)
        debut = time.perf_counter()
        recherche_lineaire_triee(un_int[0], liste_n_entiers)
        temps_ecoule = time.perf_counter()-debut
        temps_total=temps_total+temps_ecoule
    temps_moyen = temps_total/100
    liste_temps_ecoules_rech_lin_triee.append(temps_moyen)

import matplotlib.pyplot as plt
x=np.arange(100000,1000001,100000)
plt.plot(x,liste_temps_ecoules_rech_lin_triee, label="recherche linéaire triée")


def recherche_binaire(element, liste_triee):
    """ Recherche binaire. Suppose que la liste est triée.
        Retourne vrai si element est dans la liste, faux sinon
    """
    borne_inferieure = 0
    borne_superieure = len(liste_triee) - 1
    milieu = 0
 
    while borne_inferieure <= borne_superieure:
        milieu = (borne_superieure + borne_inferieure) // 2
        if liste_triee[milieu] < element:
            borne_inferieure = milieu + 1
        elif liste_triee[milieu] > element:
            borne_superieure = milieu - 1
        else: # liste_triee[milieu] == element 
            return True
    return False

import time    
liste_temps_ecoules_rech_bin =[]
for n in range(100000,10000001,100000):
    liste_n_entiers = np.random.randint(0,10*n,size=n)
    liste_n_entiers.sort()
    temps_total=0
    for i in range(100):
        un_int = np.random.randint(0,10*n,size=1)
        debut = time.perf_counter()
        recherche_binaire(un_int[0], liste_n_entiers)
        temps_ecoule = time.perf_counter()-debut
        temps_total=temps_total+temps_ecoule
    temps_moyen = temps_total/100
    liste_temps_ecoules_rech_bin.append(temps_moyen)

x=np.arange(100000,10000001,100000)
plt.plot(x,liste_temps_ecoules_rech_bin, label="recherche binaire")
plt.title("Temps moyen de la recherche")
plt.legend(loc='upper left')
plt.show()
