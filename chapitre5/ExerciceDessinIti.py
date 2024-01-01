# -*- coding: utf-8 -*-
"""
Exercice : bonhomme Iti
"""
# Importer la bibliothèque de pygame et initialiser 
import sys, pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 300
HAUTEUR_FENETRE = 300
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exercice bonhomme Iti avec pygame') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

pygame.draw.ellipse(fenetre, Color('pink'), ((133, 50), (33, 50))) # Dessiner la tête
pygame.draw.arc(fenetre, Color('black'),((140,75),(19,15)),3.1416,0,1) # Le sourire
pygame.draw.ellipse(fenetre, Color('black'), ((138,66),(8,8))) # L'oeil gauche
pygame.draw.ellipse(fenetre, Color('black'), ((154,66),(8,8))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (150,100), (150,200), 2) # Le corps
pygame.draw.line(fenetre, Color('black'), (100,100), (150,150), 2) # Bras gauche
pygame.draw.line(fenetre, Color('black'), (200,100), (150,150), 2) # Bras droit
pygame.draw.line(fenetre, Color('black'), (100,250), (150,200), 2) # Jambe gauche
pygame.draw.line(fenetre, Color('black'), (200,250), (150,200), 2) # Jambe droite

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
