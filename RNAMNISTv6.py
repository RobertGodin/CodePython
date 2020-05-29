# -*- coding: utf-8 -*-
"""
Exemple de réseau de neuronne à propagation avant et
rétropropagation de l'erreur pour l'apprentissage
"""
import random
import numpy as np
np.random.seed(42) # pour reproduire les mêmes résultats
random.seed(42)
import matplotlib.pyplot as plt

def sigmoide(z):
    """The sigmoide function."""
    return 1.0/(1.0+np.exp(-z))

def derivee_sigmoide(z):
    """Derivative of the sigmoide function."""
    return sigmoide(z)*(1-sigmoide(z))

def bitmap(classe):
    """ Representer l'entier de classe par un vecteur bitmap (10,1) 
    classe : entier entre 0 et 9 qui représente la classe de l'observation"""
    e = np.zeros((10, 1))
    e[classe] = 1.0
    return e

class RNA(object):
    """ Un RNA est un réseau de neuronnes artificiel multi-couche.
    """
        
    def __init__(self, ncs):
        """ ncs[c] contient le nombre de neurones de la couche c, c = 0 ...nombre_couches-1
        la couche d'indice 0 est la couche d'entrée
        ncs[nombre_couches-1] doit correspondre au nombre de catégories des y (sortie)
        
        liste_w[c] est la matrice des poids entre la couche c et c+1
        liste_w[c][i,j] est le poids entre le neuronne i de la couche c+1 et j de la couche c
        i = 0 correspond au biais par convention
        les poids sont initialisés avec un nombre aléatoire selon une distribution N(0,1)
        """
        self.ncs = ncs
        self.nombre_couches = len(ncs)
        # Initialiser les matrices des poids w avec des valeurs aleatoires N(0,1)
        self.liste_w = [np.random.randn(x+1,y) for x, y in zip(ncs[:-1], ncs[1:])]

    def propagation_avant_w(self, activation):
        """
        Traiter une entrée par propagation avant
        
        activation: activation initiale qui correspond aux entrées (taille self.ncs[0])
        retourne l'activation de sortie après propagation avant"""
        
        for w in self.liste_w:
            activation = np.vstack((np.ones(1),sigmoide(np.dot(w.transpose(),activation))))
        return activation

    def entrainer_par_mini_lot(self,donnees_entrainement,donnees_test,nombre_epochs,taille_mini_lot,eta):
        """
        Entrainer le RNA par mini-lots
        Affiche le nombre de bons résultats des donnees_test pour chaque epoch
        
        donnees_entrainement : liste de tuples (x,y) pour l'entrainement où
            x est un tableau de taille (ncs[0],1) où n est la taille des entrées
            y est un encodage bitmap de la catégorie en tableau de taille ncs[nombre_couches-1]
        donnees_test : liste de tuples (x,y) pour les tests
            x est un tableau de taille (ncs[0],1) où n est la taille des entrées
            y un int où 0<=y< nombre de catégories
        nombre_epochs : nombre de passe d'entrainement
        taille_mini_lot : la taille de chacun des mini-lots
        eta : vitesse d'apprentissage
        """
        n_test = len(donnees_test)
        n_ent = len(donnees_entrainement)
        self.liste_eqm_ent = []
        self.liste_ok_ent = []
        self.liste_eqm_test = []
        self.liste_ok_test = []
        
        for j in range(nombre_epochs):
            random.shuffle(donnees_entrainement)
            mini_lots = [donnees_entrainement[k:k+taille_mini_lot] for k in range(0, n_ent, taille_mini_lot)]
            # Entrainer un mimi-lot à la fois
            for mini_lot in mini_lots:
                # Initialiser les gradiants totaux à 0
                liste_dJ_dw = [np.zeros(w.shape) for w in self.liste_w]
                for x, y in mini_lot:
                    dJ_dw_une_ligne = self.retropropagation_w(x, y)
                    # ajouter les gradiants d'une observation aux totaux partiels du lot
                    liste_dJ_dw = [dJ_dw+dJ_dw_1 for (dJ_dw, dJ_dw_1) in zip(liste_dJ_dw, dJ_dw_une_ligne)]
                # mettre à jour les paramètres du RNA avec les gradiants du lot    
                self.liste_w = [w-(eta/len(mini_lot))*dw  for (w, dw) in zip(self.liste_w, liste_dJ_dw)]
            
             # Calcul des métriques de performance
            eqm_ent,ok_ent = self.metriques(donnees_entrainement)
            eqm_test,ok_test = self.metriques(donnees_test)
            self.liste_eqm_ent.append(eqm_ent/n_ent)
            self.liste_ok_ent.append(ok_ent/n_ent)
            self.liste_eqm_test.append(eqm_test/n_test)
            self.liste_ok_test.append(ok_test/n_test)
            print("Epoch {0}: EQM entrainement: {1} Taux correct: {2}".format(j, eqm_ent/n_ent,ok_ent/n_ent))
            print("Epoch {0}: EQM test: {1} Taux correct: {2}".format(j, eqm_test/n_test,ok_test/n_test))

        # Affichage des graphiques d'évolution des performances par epoch
        plt.plot(np.arange(0,nombre_epochs),self.liste_eqm_ent,label='Eqm entraînement')
        plt.plot(np.arange(0,nombre_epochs),self.liste_eqm_test,label='Eqm test')
        plt.title("Erreur quadratique moyenne")
        plt.xlabel('epoch')
        plt.ylabel('erreur')
        plt.legend(loc='upper center')
        plt.show()

        plt.plot(np.arange(0,nombre_epochs),self.liste_ok_ent,label='Nb correct entraînement')
        plt.plot(np.arange(0,nombre_epochs),self.liste_ok_test,label='Nb correct test')
        plt.title("Taux de bonnes réponses")
        plt.xlabel('epoch')
        plt.ylabel('nb correct')
        plt.legend(loc='upper center')
        plt.show()
    
    def retropropagation_w(self, x, y):
        """Return a tuple ``(dJ_db, dJ_dw)`` representing the
        gradient for the cost function C_x.  ``dJ_db`` and
        ``dJ_dw`` are layer-by-layer lists of numpy arrays, similar
        to ``self.liste_biais`` and ``self.liste_w``."""

        # propagation_avant
        activation = np.vstack((np.ones(1),x)) # activation de la couche 0
        liste_activation = [np.vstack((np.ones(1),x))] # liste des activations couche par couche
        liste_z = [] # liste des z par couche
        for w in self.liste_w:
            z = np.dot(w.transpose(),activation)
            liste_z.append(z)
            activation = np.vstack((np.ones(1),sigmoide(z))) 
            liste_activation.append(activation)
        
        # retropropagation
        dJ_dw = [np.zeros(w.shape) for w in self.liste_w]
        dJ_dz = self.dJ_da_final(liste_activation[-1][1:], y)
        dJ_dw[-1] = np.dot(liste_activation[-2],dJ_dz.transpose())
        # itérer de la couche nc-2 à la couche 1
        for c in range(2, self.nombre_couches):
            z = liste_z[-c]
            dJ_dz = np.dot(self.liste_w[-c+1], dJ_dz)[1:] * derivee_sigmoide(z)
            dJ_dw[-c] = np.dot(liste_activation[-c-1], dJ_dz.transpose())
        return dJ_dw

    def metriques(self, donnees):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        erreur_quadratique = 0
        nb_correct = 0
        for (x,y) in donnees:
            resultat_propagation = self.propagation_avant_w(np.vstack((np.ones(1),x)))[1:]
            erreur_quadratique += sum((resultat_propagation-y)**2)
            classe_predite = np.argmax(resultat_propagation)
            if y[classe_predite] == 1:
                nb_correct+=1
        return (erreur_quadratique,nb_correct)

    def dJ_da_final(self, output_activations, y):
        """Dérivée de J par rapport à l'activation"""
        return (output_activations-y)

# Chargement des données de MNIST
import pickle, gzip

fichier_donnees = gzip.open(r"mnist.pkl.gz", 'rb')
donnees_ent, donnees_validation, donnees_test = pickle.load(fichier_donnees, encoding='latin1')
fichier_donnees.close()
    
x_ent = [np.reshape(x, (784, 1)) for x in donnees_ent[0]] # les entrees (pixels de l'image) sont par colonne
y_ent = [bitmap(y) for y in donnees_ent[1]] # Encodgae bitmap de l'entier (one hot encoding)
donneesxy_ent = list(zip(x_ent, y_ent))
x_test = [np.reshape(x, (784, 1)) for x in donnees_test[0]]
y_test = [bitmap(y) for y in donnees_test[1]] # Encodgae bitmap de l'entier (one hot encoding)
donneesxy_test = list(zip(x_test, y_test)) # Encodage int pour la classe dans les tests

# Classification par RNA
net = RNA([784, 30, 10])
net.entrainer_par_mini_lot(donneesxy_ent,donneesxy_test,30,5,3.0)
