# -*- coding: utf-8 -*-
import numpy as np
import random

def sigmoide(z):
    """La fonction d'activation sigmoide"""
    return 1.0/(1.0+np.exp(-z))

def sigmoide_derivee(z):
    """Derivative of the sigmoid function."""
    return sigmoide(z)*(1-sigmoide(z))

class RNA(object):
    
    def __init__(self, nc):
        """ nc[c] contient le nombre de neurones de la couche c, c = 0 ...nombre_couches-1
        la couche d'indice 0 est la couche d'entrée
        w[c] est la marice des poids entre la couche c et c+1
        w[c][i,j] est le poids entre le neuronne i de la couche c et j de la couche c+1
        i = 0 correspond au biais par convention
        les poids sont initialisés avec un nombre aléatoire selon une distribution N(0,1)
        """
        self.nombre_couches = len(nc)
        self.nc = nc
        np.random.seed(42)
        self.w = [np.random.randn(x+1, y) for x, y in zip(nc[:-1], nc[1:])]
        print("nc:",nc)
        print("w:",self.w)

    def propagation_avant(self, a):
        """a est un vecteur d'activation. a[0]=1 correspond au biais
        retourne l'actication finale"""
        for wc in self.w:
            a = np.vstack((np.ones(1),sigmoide(np.dot(wc.transpose(), a))))
        return a

    def SGD(self, donnees_entrainement, epochs, taille_mini_batch, eta,
            donnees_test):
        """
        donnees_entrainement : list de tuples (x,y) pour l'entrainement
        donnees_test : list de tuples (x,y) pour les tests
        """
        n = len(donnees_entrainement)
        n_test = len(donnees_test)

        for j in range(epochs):
            random.shuffle(donnees_entrainement)
            mini_batches = [
                donnees_entrainement[k:k+taille_mini_batch]
                for k in range(0, n, taille_mini_batch)]
            for mini_batch in mini_batches:
                self.mini_batch(mini_batch, eta)
            print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(donnees_test), n_test))

    def mini_batch(self, mini_batch, eta):
        """
        Traitement d'une mini_batch pour entrainer les paramètres
        par rétropropation de l'erreur
        donnees_entrainement : list de tuples (x,y) pour l'entrainement
        eta : la vitesse d'entrainement
        """
        nabla_w = [np.zeros(w.shape) for w in self.w]
        for x, y in mini_batch:
            delta_nabla_w = self.retropropagation(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.w = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.w, nabla_w)]

    def retropropagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.w``."""
        # print(x,y)
        # print(type(x))
        # print(type(y))
        nabla_w = [np.zeros(wc.shape) for wc in self.w]
        # propagation avant avec stockage des activations[0] est à 1 pour les biais
        activation = np.vstack((np.ones(1),x.reshape(x.size,1))) # activation
        activations = [np.vstack((np.ones(1),x.reshape(x.size,1)))] # liste des activations couche par couche
        zs = [] # # liste des z couche par couche

        for wc in self.w:
            z = np.dot(wc.transpose(), activation)
            zs.append(z)
            activation = np.vstack((np.ones(1),sigmoide(z))) 
            activations.append(activation)

        # rétropropagation 
        # delta est le vecteur des dérivées par rapport z de l couche c-1
        # calcul de la dérivée par rapport a z pour la couche de sortie
        delta = derivee_cout(activations[-1][1:], y) * sigmoide_derivee(zs[-1])
        nabla_w[-1] = np.dot(activations[-2],delta.transpose())

        # rétropropagation couche par couche en partant de l'avant-dernière
        for l in range(2, self.nombre_couches):
            z = zs[-l]
            sp = sigmoide_derivee(z)
            delta = np.dot(self.w[-l+1], delta)[1:] * sp
            nabla_w[-l] = np.dot(activations[-l-1],delta.transpose())
        return nabla_w
    
    def evaluate(self, donnees_test):
        """Retourne la valeur de la fonction de coût pour les donnees de test"""
        resultats = [(self.propagation_avant(np.vstack((np.ones(1),x.reshape(x.size,1)))), y)
                        for (x, y) in donnees_test]
        total_bon =0
        for (x,y) in resultats:
            if (x[1] >= 0.5 and y == 1 ) or (x[1]<5 and y == 0):
                total_bon = total_bon + 1
        return total_bon

def derivee_cout(output_activations, y):
    """ Vecteur des dérivées de la fonction de cout vs aj"""
    return (output_activations-y)


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

donnees=[(iris_X[i],iris_y_setosa[i]) for i in range(len(iris_X))]
donnees_ent,donnees_test=diviser_ent_test(donnees, 0.2) 

un_rna = RNA([2,3,1])
un_rna.SGD(donnees, 10, 10, 1,donnees)









