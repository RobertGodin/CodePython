"""
Exercice : dessin d'une balle
"""
# Importer la bibliothèque de pygame et initialiser
import sys, pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 300
HAUTEUR_FENETRE = 300
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre

pygame.display.set_caption('Exercice balle avec pygame') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

pygame.draw.ellipse(fenetre, Color('darkblue'), ((60,60),(180,180))) # La balle
pygame.draw.ellipse(fenetre, Color('white'), ((105,105),(30,30))) # Le reflet
pygame.draw.arc(fenetre, Color('white'), ((90,90),(120,120)), 1.7, 3.0, 2) # L'éclat
pygame.draw.ellipse(fenetre, Color('black'), ((60,60),(180,180)), 2) # Le contour

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
