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
#            random.shuffle(donnees_entrainement)
            mini_lots = [donnees_entrainement[k:k+taille_mini_lot] for k in range(0, n, taille_mini_lot)]
            for un_lot in mini_lots:
                self.traiter_un_lot(un_lot,eta)
            if donnees_test:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(donnees_test), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def traiter_un_lot(self, un_mini_lot, eta):
        """Update the network's liste_w and biais by applying
        gradient descent using backpropagation to a single mini batch.
        The ``un_mini_lot`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.liste_w]
        for x, y in un_mini_lot:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.liste_w = [w-(eta/len(un_mini_lot))*nw
                        for w, nw in zip(self.liste_w, nabla_w)]
        self.biais = [b-(eta/len(un_mini_lot))*nb
                       for b, nb in zip(self.biais, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biais`` and ``self.liste_w``."""
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.liste_w]
        # propagation_avant
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biais, self.liste_w):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
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
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, donnees_test):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.propagation_avant(x)), y)
                        for (x, y) in donnees_test]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# RNA pour la collection Iris
# X: longueur et largeur de sépale, y: setosa ou non


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

# Charger les données
iris = datasets.load_iris()
iris_X = iris.data[:,:2] # les deux colonnes longueur et largeur de sépale
iris_y = iris.target
iris_y_setosa = (iris_y==0).astype(np.int) # setosa ou non

# Effectuer la régression logistique
regression_logistique = LogisticRegression()
regression_logistique.fit(iris_X,iris_y_setosa)

# Afficher la frontiere de classe
x_min, x_max = iris_X[:, 0].min() - .5, iris_X[:, 0].max() + .5
y_min, y_max = iris_X[:, 1].min() - .5, iris_X[:, 1].max() + .5

theta_0 = regression_logistique.intercept_
theta_1 = regression_logistique.coef_[0,0]
theta_2 = regression_logistique.coef_[0,1]
x1_intervalle = np.array((x_min,x_max))
x2_frontiere = -((theta_1/theta_2)*x1_intervalle)-(theta_0/theta_2)
plt.plot(x1_intervalle,x2_frontiere,'-r',label = 'Frontiere de classe')

# Afficher les données, la couleur dépend de la classe d'Iris
plt.title("Classe d'Iris (setosa ou non)")
plt.xlabel('longueur de sépale')
plt.ylabel('largeur de sépale')
plt.legend(loc='lower right')
plt.scatter(iris_X[:,0],iris_X[:,1],c=iris_y_setosa)
plt.show()

print("Métrique d'évaluation:")
print(regression_logistique.score(iris_X,iris_y_setosa))

def diviser_ent_test(donnees, proportion):
    donnees_permutes = np.random.permutation(donnees)
    taille_test = int(len(donnees) * proportion)
    return donnees_permutes[:taille_test],donnees_permutes[taille_test:]

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

donnees=[(np.reshape(iris_X[i], (2, 1)),vectorized_result(iris_y_setosa[i])) for i in range(len(iris_X))]
donnees_test=[(np.reshape(iris_X[i], (2, 1)),iris_y_setosa[i]) for i in range(len(iris_X))]

un_rna = RNA([2,5,2])
un_rna.entrainer_par_mini_lot(donnees, 50, 10, 3,donnees_test=donnees_test)

