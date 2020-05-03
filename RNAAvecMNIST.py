# -*- coding: utf-8 -*-
"""
Exemple de réseau de neuronne à propagation avant et
rétropropagation de l'erreur pour l'apprentissage
"""
import random
import numpy as np
np.random.seed(42) # pour reproduire les mêmes résultats
random.seed(42)

class RNA(object):
    """ Un RNA est un réseau de neuronnes artificiel multi-couche.
    """
        
    def __init__(self, ncs):
        """ nc[c] contient le nombre de neurones de la couche c, c = 0 ...nombre_couches-1
        la couche d'indice 0 est la couche d'entrée
        ncs[nombre_couches-1] doit correspondre au nombre de catégories des y (sortie)
        
        w[c] est la matrice des poids entre la couche c et c+1
        w[c][i,j] est le poids entre le neuronne i de la couche c+1 et j de la couche c
        i = 0 correspond au biais par convention
        les poids sont initialisés avec un nombre aléatoire selon une distribution N(0,1)
        """
        self.nombre_couches = len(ncs)
        self.ncs = ncs
        self.biais = [np.random.randn(y, 1) for y in ncs[1:]]
        self.liste_w = [np.random.randn(y, x) for x, y in zip(ncs[:-1], ncs[1:])]

    def propagation_avant(self, activation):
        """
        Traiter une entrée par propagation avant
        
        activation: activation initiale qui correspond aux entrées (taille self.ncs[0])
        retourne l'activation de sortie après propagation avant"""
        
        for b, w in zip(self.biais, self.liste_w):
            activation = sigmoid(np.dot(w, activation)+b)
        return activation

    def entrainer_par_mini_lot(self,donnees_entrainement,nombre_epochs,taille_mini_lot,eta,donnees_test=None):
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
        if donnees_test: n_test = len(donnees_test)
        n = len(donnees_entrainement)
        for j in range(nombre_epochs):
            random.shuffle(donnees_entrainement)
            mini_lots = [donnees_entrainement[k:k+taille_mini_lot] for k in range(0, n, taille_mini_lot)]
            for un_lot in mini_lots:
                self.traiter_un_lot(un_lot,eta)
            if donnees_test:
                print("Epoch {0}: {1} / {2}".format(j, self.nb_correct(donnees_test), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def traiter_un_lot(self, un_mini_lot, eta):
        """ Entrainer le RNA avec un mini lot """
        dJ_db = [np.zeros(b.shape) for b in self.biais]
        dJ_dw = [np.zeros(w.shape) for w in self.liste_w]
        for x, y in un_mini_lot:
            delta_dJ_db, delta_dJ_dw = self.retropropagation(x, y)
            dJ_db = [nb+dnb for nb, dnb in zip(dJ_db, delta_dJ_db)]
            dJ_dw = [nw+dnw for nw, dnw in zip(dJ_dw, delta_dJ_dw)]
        self.liste_w = [w-(eta/len(un_mini_lot))*nw
                        for w, nw in zip(self.liste_w, dJ_dw)]
        self.biais = [b-(eta/len(un_mini_lot))*nb
                       for b, nb in zip(self.biais, dJ_db)]

    def retropropagation(self, x, y):
        """Return a tuple ``(dJ_db, dJ_dw)`` representing the
        gradient for the cost function C_x.  ``dJ_db`` and
        ``dJ_dw`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biais`` and ``self.liste_w``."""
        dJ_db = [np.zeros(b.shape) for b in self.biais]
        dJ_dw = [np.zeros(w.shape) for w in self.liste_w]
        
        # propagation_avant
        activation = x
        activations = [x] # liste des activations par couche
        zs = [] # liste des z par couche
        for b, w in zip(self.biais, self.liste_w):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # retropropagation
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        dJ_db[-1] = delta
        dJ_dw[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.nombre_couches):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.liste_w[-l+1].transpose(), delta) * sp
            dJ_db[-l] = delta
            dJ_dw[-l] = np.dot(delta, activations[-l-1].transpose())
        return (dJ_db, dJ_dw)

    def nb_correct(self, donnees_test):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        test_results = [(np.argmax(self.propagation_avant(x)), y)
                        for (x, y) in donnees_test]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def bitmap(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Chargement des données de MNIST
import pickle, gzip

f = gzip.open(r"mnist.pkl.gz", 'rb')
donnees_ent, donnees_validation, donnees_test = pickle.load(f, encoding='latin1')
f.close()
    
x_ent = [np.reshape(x, (784, 1)) for x in donnees_ent[0]] # les entrees (pixels de l'image) sont par colonne
y_ent = [bitmap(y) for y in donnees_ent[1]] # Encodgae bitmap de l'entier (one hot encoding)
donneesxy_ent = list(zip(x_ent, y_ent))
x_test = [np.reshape(x, (784, 1)) for x in donnees_test[0]]
donneesxy_test = list(zip(x_test, donnees_test[1])) # Encodage int pour la classe dans les tests

# Classification par RNA
net = RNA([784, 30, 10])
net.entrainer_par_mini_lot(donneesxy_ent, 30, 10, 3.0, donnees_test=donneesxy_test)







