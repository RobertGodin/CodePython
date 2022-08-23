# -*- coding: utf-8 -*-
"""
Exercice : dessin de 2 bots
"""
# Importer la bibliothèque de pygame et initialiser 
import sys, pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Dessin de 2 Bots') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
# Le premier Bot    
pygame.draw.ellipse(fenetre, Color('green'), ((100, 100),(200, 200))) # Dessiner la tête
pygame.draw.rect(fenetre, Color('black'), ((150, 150),(20, 20))) # L'oeil gauche
pygame.draw.rect(fenetre, Color('black'), ((230, 150),(20, 20))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (150,250),(250,250), 2) # La bouche
pygame.draw.rect(fenetre, Color('red'), ((100, 300),(200, 200))) # Le corps

# Le deuxième Bot    
pygame.draw.ellipse(fenetre, Color('green'), ((25,50),(100,100))) # Dessiner la tête
pygame.draw.rect(fenetre, Color('black'), ((50, 75),(10, 10))) # L'oeil gauche
pygame.draw.rect(fenetre, Color('black'), ((90,75),(10,10))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (50,125),(100,125), 2) # La bouche
pygame.draw.rect(fenetre, Color('red'), ((25,150),(100,100))) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
