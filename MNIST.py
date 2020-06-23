
# Exemple avec MNIST
import matplotlib.pyplot as plt
import matplotlib as mpl
# Chargement des données de MNIST
import pickle, gzip
import numpy as np

def bitmap(classe):
    """ Representer l'entier de classe par un vecteur bitmap (10,1) 
    classe : entier ebitmap(ntre 0 et 9 qui représente la classe de l'observation"""
    e = np.zeros((1,10))
    e[0,classe] = 1.0
    return e

fichier_donnees = gzip.open(r"mnist.pkl.gz", 'rb')
donnees_ent, donnees_validation, donnees_test = pickle.load(fichier_donnees, encoding='latin1')
fichier_donnees.close()

for i in range(3):
    print("Classe de l'image",i,":",donnees_ent[1][i])
    image_applatie = donnees_ent[0][i]
    une_image = image_applatie.reshape(28, 28)
    plt.imshow(une_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    
donnees_ent_X = donnees_ent[0].reshape((50000,1,784))
donnees_ent_Y = [bitmap(y) for y in donnees_ent[1]] # Encodgae bitmap de l'entier (one hot encoding)
donnees_test_X = donnees_test[0].reshape((10000,1,784))
donnees_test_Y = [bitmap(y) for y in donnees_test[1]] # Encodgae bitmap de l'entier (one hot encoding)

for i in range(3):
    print("Classe de l'image",i,":",donnees_ent_Y[i])
    image_applatie = donnees_ent_X[i]
    une_image = image_applatie.reshape(28, 28)
    plt.imshow(une_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

un_autre_chiffre = donnees_ent[0][0].reshape(28,28)
plt.imshow(un_autre_chiffre, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()