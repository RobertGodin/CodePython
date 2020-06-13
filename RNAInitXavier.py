# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:19:48 2020

@author: Robert
"""

    def __init__(self,n,m,init_W=None,init_B=None):
        """ Initilalise les paramètres de la couche. W et B sont initialisés avec init_W et init_B lorsque spécifiés.
        Sinon, des valeurs aléatoires sont générés pour W une distribution normale N(0,var) où var est calibré selon
        l'approche de Xavier. B est initialisée avec des 0. 
        si les paramètres init_W et init_B ne sont pas spécifiés.
        n : int, taille du vecteur d'entrée X
        m : int, taille du vecteur de sortie Y
        init_W : np.array, shape(n,m), valeur initiale optionnelle de W
        init_B : np.array, shape(1,m), valeur initial optionnelle de B
        """
        if init_W is None :
            self.W = np.random.randn(n,m)*np.sqrt(2/(n+m)) 
        else:
            self.W = init_W
        if init_B is None :
            self.B = np.zeros(1, m)
        else:
            self.B = init_B