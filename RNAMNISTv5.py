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
        self.nombre_couches = len(ncs)
        self.ncs = ncs
        self.liste_biais = [np.random.randn(y, 1) for y in ncs[1:]]
        self.liste_w = [np.random.randn(y, x) for x, y in zip(ncs[:-1], ncs[1:])]
        
        self.liste_w_b = [np.random.randn(x+1,y) for x, y in zip(ncs[:-1], ncs[1:])]
        

    def propagation_avant(self, activation):
        """
        Traiter une entrée par propagation avant
        
        activation: activation initiale qui correspond aux entrées (taille self.ncs[0])
        retourne l'activation de sortie après propagation avant"""
        
        for b,w in zip(self.liste_biais, self.liste_w):
            activation = sigmoide(np.dot(w, activation)+b)
        return activation

    def propagation_avant_w_b(self, activation):
        """
        Traiter une entrée par propagation avant
        
        activation: activation initiale qui correspond aux entrées (taille self.ncs[0])
        retourne l'activation de sortie après propagation avant"""
        
        for w in self.liste_w_b:
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
        n = len(donnees_entrainement)
        self.liste_eqm_ent = []
        self.liste_ok_ent = []
        self.liste_eqm_test = []
        self.liste_ok_test = []
        
        for j in range(nombre_epochs):
            random.shuffle(donnees_entrainement)
            mini_lots = [donnees_entrainement[k:k+taille_mini_lot] for k in range(0, n, taille_mini_lot)]
            for mini_lot in mini_lots:
                # Entrainer avec un mimi-lot
                # Initialiser les gradiants totaux à 0
                liste_dJ_db = [np.zeros(b.shape) for b in self.liste_biais]
                liste_dJ_dw = [np.zeros(w.shape) for w in self.liste_w]
                
                liste_dJ_dw_b = [np.zeros(w.shape) for w in self.liste_w_b]
                
                for x, y in mini_lot:
                    dJ_db_une_ligne, dJ_dw_une_ligne = self.retropropagation(x, y)
                    dJ_dw_b_une_ligne = self.retropropagation_w_b(x, y)
                    
                    # ajouter les gradiants d'une observation aux totaux partiels du lot
                    liste_dJ_db = [dJ_db+dJ_db_1 for (dJ_db, dJ_db_1) in zip(liste_dJ_db, dJ_db_une_ligne)]
                    liste_dJ_dw = [dJ_dw+dJ_dw_1 for (dJ_dw, dJ_dw_1) in zip(liste_dJ_dw, dJ_dw_une_ligne)]
                    
                    liste_dJ_dw_b = [dJ_dw_b+dJ_dw_b_1 for (dJ_dw_b, dJ_dw_b_1) in zip(liste_dJ_dw_b, dJ_dw_b_une_ligne)]
                    
                # mettre à jour les paramètres du RNA avec les gradiants du lot    
                self.liste_biais = [b-(eta/len(mini_lot))*db for (b, db) in zip(self.liste_biais, liste_dJ_db)]
                self.liste_w = [w-(eta/len(mini_lot))*dw  for (w, dw) in zip(self.liste_w, liste_dJ_dw)]
                
                self.liste_w_b = [w-(eta/len(mini_lot))*dw  for (w, dw) in zip(self.liste_w_b, liste_dJ_dw_b)]
                
            print("EQM Epoch {0}: {1} / {2}".format(j, self.eqm(donnees_entrainement),n))
            # print(" Nb_correct Epoch{0}: {1} / {2}".format(j, self.nb_correct(donnees_test), n_test))
            
            print("EQM Epoch {0}: {1} / {2}".format(j, self.eqm_w_b(donnees_entrainement),n))
            eqm_ent,ok_ent = self.metriques(donnees_entrainement)
            eqm_test,ok_test = self.metriques(donnees_test)
            self.liste_eqm_ent.append(eqm_ent/n)
            self.liste_ok_ent.append(ok_ent/n)
            self.liste_eqm_test.append(eqm_test/n_test)
            self.liste_ok_test.append(ok_test/n_test)
            
        plt.plot(np.arange(0,nombre_epochs),self.liste_eqm_ent,label='Eqm entraînement')
        plt.plot(np.arange(0,nombre_epochs),self.liste_eqm_test,label='Eqm test')
        plt.title("Erreur quadratique moyenne")
        plt.xlabel('epoch')
        plt.ylabel('erreur')
        plt.legend(loc='upper center')
        plt.show()

        plt.plot(np.arange(0,nombre_epochs),self.liste_ok_ent,label='Nb correct entraînement')
        plt.plot(np.arange(0,nombre_epochs),self.liste_ok_test,label='Nb correct test')
        plt.title("Nombre de bonnes réponses")
        plt.xlabel('epoch')
        plt.ylabel('nb correct')
        plt.legend(loc='upper center')
        plt.show()
            

    def retropropagation(self, x, y):
        """Return a tuple ``(dJ_db, dJ_dw)`` representing the
        gradient for the cost function C_x.  ``dJ_db`` and
        ``dJ_dw`` are layer-by-layer lists of numpy arrays, similar
        to ``self.liste_biais`` and ``self.liste_w``."""
        dJ_db = [np.zeros(b.shape) for b in self.liste_biais]
        dJ_dw = [np.zeros(w.shape) for w in self.liste_w]

        # propagation_avant
        activation = x
        activations = [x] # liste des activations par couche
        zs = [] # liste des z par couche
        for b,w in zip(self.liste_biais, self.liste_w):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoide(z)
            activations.append(activation)
        
        # retropropagation
        dJ_dz = self.dJ_da_final(activations[-1], y)*derivee_sigmoide(zs[-1])
        dJ_db[-1] = dJ_dz
        dJ_dw[-1] = np.dot(dJ_dz, activations[-2].transpose())
        # itérer de la couche nc-2 à la couche 1
        for l in range(2, self.nombre_couches):
            z = zs[-l]
            sp = derivee_sigmoide(z)
            dJ_dz = np.dot(self.liste_w[-l+1].transpose(), dJ_dz) * sp
            dJ_db[-l] = dJ_dz
            dJ_dw[-l] = np.dot(dJ_dz, activations[-l-1].transpose())
        return (dJ_db, dJ_dw)
    
    def retropropagation_w_b(self, x, y):
        """Return a tuple ``(dJ_db, dJ_dw)`` representing the
        gradient for the cost function C_x.  ``dJ_db`` and
        ``dJ_dw`` are layer-by-layer lists of numpy arrays, similar
        to ``self.liste_biais`` and ``self.liste_w``."""
        dJ_dw_b = [np.zeros(w.shape) for w in self.liste_w_b]

        # propagation_avant
        activation = np.vstack((np.ones(1),x)) # activation
        activations = [np.vstack((np.ones(1),x))] # liste des activations couche par couche
        zs = [] # liste des z par couche
        for w in self.liste_w_b:
            z = np.dot(w.transpose(),activation)
            zs.append(z)
            activation = np.vstack((np.ones(1),sigmoide(z))) 
            activations.append(activation)
        
        # retropropagation
        dJ_dz = self.dJ_da_final(activations[-1][1:], y)*derivee_sigmoide(zs[-1])
        dJ_dw_b[-1] = np.dot(activations[-2],dJ_dz.transpose())
        # itérer de la couche nc-2 à la couche 1
        for l in range(2, self.nombre_couches):
            z = zs[-l]
            sp = derivee_sigmoide(z)
            dJ_dz = np.dot(self.liste_w_b[-l+1], dJ_dz)[1:] * sp
            dJ_dw_b[-l] = np.dot(activations[-l-1], dJ_dz.transpose())
        return dJ_dw_b


    def nb_correct(self, donnees_test):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        resultats_test = [(np.argmax(self.propagation_avant(x)), y) for (x, y) in donnees_test]
        return sum(int(x == y) for (x, y) in resultats_test)

    def eqm(self, donnees_ent):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        resultats_ent = [sum(self.propagation_avant(x)-y)**2 for (x, y) in donnees_ent]
        return sum(resultats_ent)/len(resultats_ent)
    
    def eqm_w_b(self, donnees_ent):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        
        resultats_ent = [sum(self.propagation_avant_w_b(np.vstack((np.ones(1),x)))[1:]-y)**2 for (x, y) in donnees_ent]  
        return sum(resultats_ent)/len(resultats_ent)

    def metriques(self, donnees):
        """Retourne le nombre de bons résultats
        Choisit l'indice de la classe dont l'activation est la plus grande"""
        erreur_quadratique = 0
        nb_correct = 0
        for (x,y) in donnees:
            resultat_propagation = self.propagation_avant_w_b(np.vstack((np.ones(1),x)))[1:]
            erreur_quadratique += sum((resultat_propagation-y)**2)
            classe_predite = np.argmax(resultat_propagation)
            if y[classe_predite] == 1:
                nb_correct+=1
            
        return (erreur_quadratique,nb_correct)
            
        resultats_ent = [sum(self.propagation_avant_w_b(np.vstack((np.ones(1),x)))[1:]-y)**2 for (x, y) in donnees_ent]  
        return sum(resultats_ent)/len(resultats_ent)

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








