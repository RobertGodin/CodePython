# -*- coding: utf-8 -*-

import numpy as np

# Base class
class Couche:
    def __init__(self):
        self.X = None
        self.Y = None

    # computes the output Y of a layer for a given input X
    def propagation(self, X):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def retropropagation(self, Y, taux):
        raise NotImplementedError

# inherit from base class Layer
class CoucheDense(Couche):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, n, m, init_W = None, init_B = None):
        if init_W is None :
            self.W = np.random.rand(n,m) - 0.5
        else:
            self.W = init_W
        if init_B is None :   
            self.B = np.random.rand(1, m) - 0.5
        else:
            self.B = init_B

    # returns output for a given input
    def propagation(self, X):
        self.X = X
        self.Y = self.B + np.dot(self.X, self.W)
        return self.Y

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def retropropagation(self, dJ_dY, taux):
        dJ_dW = np.dot(self.X.T, dJ_dY)
        dJ_dB = dJ_dY
        dJ_dX = np.dot(dJ_dY, self.W.T)
        print("dJ_dW",dJ_dW)
        print("dJ_dB",dJ_dB)
        print("dJ_dX",dJ_dX)
        # update parameters
        self.W -= taux * dJ_dW
        self.B -= taux * dJ_dB
        return dJ_dX
    
# inherit from base class Layer
class CoucheActivation(Couche):
    def __init__(self, fonction_activation, derivee):
        self.fonction_activation = fonction_activation
        self.derivee = derivee

    # returns the activated input
    def propagation(self, X):
        self.X = X
        self.Y = self.fonction_activation(self.X)
        return self.Y

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def retropropagation(self, dJ_dY, learning_rate):
        return self.derivee(self.X) * dJ_dY

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# cout function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def erreur_quadratique(y_prediction,y):
    return np.sum(np.power(y_prediction-y, 2))

def d_erreur_quadratique(y_prediction,y):
    return 2*(y_prediction-y)

class ReseauMultiCouches:
    def __init__(self):
        self.couches = []
        self.cout = None
        self.derivee_cout = None

    # ajouter_couche layer to network
    def ajouter_couche(self, couche):
        self.couches.append(couche)

    # Spécifier la fonction de coût et sa dérivée
    def specifier_J(self, cout, derivee_cout):
        self.cout = cout
        self.derivee_cout = derivee_cout

    # predict output for given input
    def propager_par_couche(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for couche in self.couches:
                output = couche.propagation(output)
            result.append(output)

        return result

    # train the network
    def entrainer_descente_gradiant_stochastique(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for couche in self.couches:
                    output = couche.propagation(output)

                # compute cout (for display purpose only)
                err += self.cout(y_train[j], output)

                # backward propagation
                error = self.derivee_cout(output,y_train[j])
                for couche in reversed(self.couches):
                    error = couche.retropropagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

# training data
#x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
#y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
            
x_train = np.array([[[1,1]]])
y_train = np.array([[[1,0]]])

# Définir les paramètres initiaux de Minus
B1=np.array([[0.2,0.7]])
W1=np.array([[0.5,0.1],[0.3,-0.3]])
B2=np.array([[-0.2,0.5]])
W2=np.array([[0.7,-0.1],[0,0.2]])

# Définir l'architecture du réseau
net = ReseauMultiCouches()
net.ajouter_couche(CoucheDense(2, 2, init_W=W1, init_B=B1))
net.ajouter_couche(CoucheDense(2, 2,init_W=W2, init_B=B2))

# Entrainer le réseau
net.specifier_J(erreur_quadratique, d_erreur_quadratique)
net.entrainer_descente_gradiant_stochastique(x_train, y_train, epochs=10, learning_rate=0.1)

# test
out = net.propager_par_couche(x_train)
print(out)





