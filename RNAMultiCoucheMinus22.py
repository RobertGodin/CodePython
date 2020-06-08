# -*- coding: utf-8 -*-
# Implémentation d'un RNA multi-couche, exemple avec le RNA Minus

import numpy as np
np.random.seed(42) # pour reproduire les mêmes résultats

class Couche:
    """ Classe abstraite qui représente une couche du RNA
        X: vecteur, entrée de la couche 
        Y: vecteur, sortie de la couche
    """
    def __init__(self):
        self.X = None
        self.Y = None

    def propager_une_couche(self, X):
        """ Calculer la sortie Y pour une valeur de X
        
        X : vecteur des variables prédictives
        Les valeurs de X et Y sont stockées pour les autres traitements.
        """
        raise NotImplementedError

    def retropropager_une_couche(self, dJ_dY, taux):
        """ Calculer les dérivées par rapport à X et les autres paramètres à partir de dJ_dY
        et mettre à jour les paramètres de la couche selon le taux spécifié.
        
        dJ_dY : np.float, la dérivée de J par rapport à la sortie Y
        taux : np.float, taux est le taux dans la descente de gradiant
        retourne la dérivée de J par rapport à X
        """
        raise NotImplementedError

# inherit from base class Layer
class CoucheDenseLineaire(Couche):
    """ Couche linéaire dense. Y=WX+B
    """
    def __init__(self, n, m, init_W = None, init_B = None):
        """ Initilalise les paramètres de la couche. W et B sont initialisés avec des valeurs aléatoires
        selon une distribution uniforme entre U(-0.5,0.5) si les paramètres init_W et init_B
        ne sont pas spécifiés.
        n : int, taille du vecteur d'entrée X
        m : int, taille du vecteur de sortie Y
        init_W : np.array, shape(n,m), valeur initiale optionnelle de W
        init_B : np.array, shape(1,m), valeur initial optionnelle de B
        """
        if init_W is None :
            self.W = np.random.rand(n,m) - 0.5
        else:
            self.W = init_W
        if init_B is None :
            self.B = np.random.rand(1, m) - 0.5
        else:
            self.B = init_B

    def propager_une_couche(self, X):
        """ Fait la propagation de X et retourne Y=WX+B. 
        """
        self.X = X
        self.Y = self.B + np.dot(self.X, self.W)
        return self.Y

    def retropropager_une_couche(self, dJ_dY, taux):
        """ Calculer les dérivées dJ_dW,dJ_dB,dJ_dX pour une couche linéaire dense et
        mettre à jour les paramètres
        """
        dJ_dW = np.dot(self.X.T, dJ_dY)
        dJ_dB = dJ_dY
        dJ_dX = np.dot(dJ_dY, self.W.T)
        # Metre à jour les paramètres W et B
        self.W -= taux * dJ_dW
        self.B -= taux * dJ_dB
        return dJ_dX
    
class CoucheActivation(Couche):
    """ Couche d'activation selon une fonction spécifiée dans le constructeur
    """
    def __init__(self, fonction_activation, derivee):
        """ Initialise la fonction_activation ainsi que la dérivée
        fonction_activation: une fonction qui prend chacune des valeurs de X et 
        retourne Y=fonction_activation(X)
        derivee: une fonction qui calcule la dérivée la fonction_activation
        """
        self.fonction_activation = fonction_activation
        self.derivee = derivee

    def propager_une_couche(self, X):
        """ Retourne Y=fonction_activation(X)
        """
        self.X = X
        self.Y = self.fonction_activation(self.X)
        return self.Y

    def retropropager_une_couche(self, dJ_dY, taux):
        """ Retourne la dérivée de la fonction d'activation par rapport l'entrée X
        Le taux n'est pas utilisé parce qu'il n'y a pas de paramètres à modifier dans ce genre de couche
        """
        return self.derivee(self.X) * dJ_dY

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def eqm(y_prediction,y):
    """ Retourne l'erreur quadratique moyenne entre la prédiction y_prediction et la valeur attendue y
    """
    return np.mean(np.power(y_true-y_pred, 2))

def derivee_eqm(y_prediction,y):
    return 2*(y_pred-y_true)/y_true.size

def erreur_quadratique(y_prediction,y):
    """ Retourne l'erreur quadratique entre la prédiction y_prediction et la valeur attendue y
    """ 
    return np.sum(np.power(y_prediction-y, 2))

def d_erreur_quadratique(y_prediction,y):
    return 2*(y_prediction-y)

class ReseauMultiCouches:
    """ Réseau mutli-couche formé par une séquence de Couches
    
    couches : liste de Couches du RNA
    cout : fonction qui calcule de cout J
    derivee_cout: dérivée de la fonction de cout
    """
    def __init__(self):
        self.couches = []
        self.cout = None
        self.derivee_cout = None

    def ajouter_couche(self, couche):
        self.couches.append(couche)

    def specifier_J(self, cout, derivee_cout):
        """ Spécifier la fonction de coût J et sa dérivée
        """
        self.cout = cout
        self.derivee_cout = derivee_cout

    def propagation_donnees_ent_X(self, donnees_ent_X,trace=False):
        """ Prédire Y pour chacune des observations dans donnees_ent_X)
        donnees_ent_X : np.array 3D des valeurs de X pour chacune des observations
            chacun des X est un np.array 2D de taille (1,n)
        """

        nb_observations = len(donnees_ent_X)
        predictions_Y = []
        for indice_observation in range(nb_observations):
            XY_propage = donnees_ent_X[indice_observation]
            if trace: 
                print("Valeur de X initiale:",XY_propage)
            for couche in self.couches:
                XY_propage = couche.propager_une_couche(XY_propage)
                if trace: 
                    print("Valeur de Y après propagation pour la couche:",XY_propage)
            predictions_Y.append(XY_propage)

        return predictions_Y

    def entrainer_descente_gradiant_stochastique(self, donnees_ent_X, donnees_ent_Y, nb_epochs, taux):
        """ Entrainer le réseau par descente de gradiant stochastique (une observation à la fois)
        donnees_ent_X : np.array 3D des valeurs de X pour chacune des observations
            chacun des X est un np.array 2D de taille (1,n)
        donnees_ent_Y : np.array 3D des valeurs de Y pour chacune des observations
            chacun des Y est un np.array 2D de taille (1,m)
        """
        
        nb_observations = len(donnees_ent_X)

        # Boucle d'entrainement principale, nb_epochs fois
        for cycle in range(nb_epochs):
            cout_total = 0
            # Descente de gradiant stochastique, une observation à la fois
            for indice_observation in range(nb_observations):
                # Propagation avant pour une observation X
                XY_propage = donnees_ent_X[indice_observation]
                for couche in self.couches:
                    XY_propage = couche.propager_une_couche(XY_propage)

                # Calcul du coût pour une observation
                cout_total += self.cout(XY_propage,donnees_ent_Y[indice_observation])

                # Rétropropagation pour une observation
                # dJ_dX_dJ_dY représente la valeur de la dérivée dJ_dX passée à dJ_dY de couche en couche
                dJ_dX_dJ_dY = self.derivee_cout(XY_propage,donnees_ent_Y[indice_observation])
                for couche in reversed(self.couches):
                    dJ_dX_dJ_dY = couche.retropropager_une_couche(dJ_dX_dJ_dY, taux)

            # Calculer et afficher le coût moyen pour une epoch
            cout_moyen = cout_total/nb_observations
            print('epoch %d/%d   cout_moyen=%f' % (cycle+1, nb_epochs, cout_moyen))

# Une seule observation pour illustrer le fonctionnement de RNA Minus           
donnees_ent_X = np.array([[[1,1]]])
donnees_ent_Y = np.array([[[1,0]]])

# Définir les paramètres initiaux de RNA Minus
B1=np.array([[0.2,0.7]])
W1=np.array([[0.5,0.1],[0.3,-0.3]])
B2=np.array([[-0.2,0.5]])
W2=np.array([[0.7,-0.1],[0,0.2]])

# Définir l'architecture du RNA Minus
un_RNA = ReseauMultiCouches()
un_RNA.ajouter_couche(CoucheDenseLineaire(2, 2, init_W=W1, init_B=B1))
un_RNA.ajouter_couche(CoucheDenseLineaire(2, 2,init_W=W2, init_B=B2))

# Tester le RNA Minus avant entrainement
predictions_Y = un_RNA.propagation_donnees_ent_X(donnees_ent_X,trace=True)
print("Prédiction initiale: ",predictions_Y)
# Entrainer le RNA Minus
un_RNA.specifier_J(erreur_quadratique, d_erreur_quadratique)
un_RNA.entrainer_descente_gradiant_stochastique(donnees_ent_X, donnees_ent_Y, nb_epochs=1, taux=0.1)

# Tester le RNA Minus
predictions_Y = un_RNA.propagation_donnees_ent_X(donnees_ent_X,trace=True)
print("Prédiction après entraînement:", predictions_Y)






