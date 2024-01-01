# -*- coding: utf-8 -*-
"""
Exemple de dessin 2D avec pygame qui emploie la classe pygame.Color
"""
# Importer la librairie de pygame et initialiser 
import sys, pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin 2D avec pygame') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
    
pygame.draw.ellipse(fenetre, Color('green'), ((100,100),(200,200))) # Dessiner la tête
pygame.draw.rect(fenetre, Color('black'), ((150,150),(20,20))) # L'oeil gauche
pygame.draw.rect(fenetre, Color('black'), ((230,150),(20,20))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (150,250),(250,250),2) # La bouche
pygame.draw.rect(fenetre, Color('red'), ((100,300),(200,200))) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()