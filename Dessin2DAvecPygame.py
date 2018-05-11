# -*- coding: utf-8 -*-
"""
Exemple de dessin 2D avec pygame
"""
# Importer la librairie de pygame et initialiser 
import pygame
pygame.init()

size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouverture de la fenêtre 

pygame.display.set_caption('Exemple de dessin 2D avec pygame') # Définition d'un titre dans le haut de la fenêtre

# Définition des couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)

fin = False 
horloge = pygame.time.Clock()
 
# Itérer jusqu'à ce qu'un évènement provoque la fermeture de la fenêtre
while not fin:
    for event in pygame.event.get():  # Parcourir la liste des évènements 
        if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre
            fin = True  # Fin de la boucle du jeu



    fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
    
    # Dessiner la tête (cercle) dans le rectangle englobant, ligne de 0 pixels de large correspond à remplissage
    pygame.draw.ellipse(fenetre, VERT, [100, 100, 200, 200], 0)
    pygame.draw.rect(fenetre, NOIR, [150,150,20,20], 0) # L'oeil gauche
    pygame.draw.rect(fenetre, NOIR, [230,150,20,20], 0) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [150,250],[250,250], 2) # La bouche
    pygame.draw.rect(fenetre, ROUGE, [100,300,200,200], 0) # Le corps

   # This MUST happen after all the other drawing commands.
    pygame.display.flip()
 
    # This limits the while loop to a max of 60 times per second.
    # Leave this out and we will use all CPU we can.
    horloge.tick(60)
 
# Be IDLE friendly
pygame.quit()