# -*- coding: utf-8 -*-

# Chargement des données de MNIST

import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

plt.figure()
im = plt.imshow(np.reshape(donnees_ent[0][0], newshape=(28,28)),
                interpolation='none', vmin=0, vmax=1, aspect='equal');

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 27, 1));
ax.set_yticks(np.arange(0, 27, 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(0, 27, 1));
ax.set_yticklabels(np.arange(0, 27, 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, 28, 1), minor=True);
ax.set_yticks(np.arange(-.5, 28, 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



image = np.reshape(donnees_ent[0][0], newshape=(28,28))

fig, ax = plt.subplots()
fig = plt.figure(figsize=(10,10))
im = ax.imshow(image,interpolation='none', vmin=0, vmax=1, aspect='equal')

ax.set_xticks(np.arange(0, 27, 1));
ax.set_yticks(np.arange(0, 27, 1));

ax.set_xticklabels(np.arange(0, 27, 1));
ax.set_yticklabels(np.arange(0, 27, 1));


# Loop over data dimensions and create text annotations.
for i in range(28):
    for j in range(28):
        text = ax.text(j, i, image[i, j],
                       ha="center", va="center", color="w",fontsize="medium")
ax.set_title("Image")
#fig.tight_layout()
plt.show()
