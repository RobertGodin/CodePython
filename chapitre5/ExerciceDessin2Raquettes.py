"""
Exercice : dessin de 2 raquettes
"""
# Importer la bibliothèque de pygame et initialiser
import pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre

pygame.display.set_caption('Dessin de 2 raquettes') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
# La première raquette
pygame.draw.ellipse(fenetre, Color('red'), ((50,250),(100,100))) # L'embout gauche
pygame.draw.ellipse(fenetre, Color('red'), ((250,250),(100,100))) # L'embout droit
pygame.draw.rect(fenetre, Color('gray'), ((100,250),(200,100))) # Le corps
pygame.draw.rect(fenetre, Color('yellow'), ((125,275),(30,20))) # Le voyant gauche
pygame.draw.rect(fenetre, Color('yellow'), ((245,275),(30,20))) # Le voyant droit
pygame.draw.line(fenetre, Color('black'), (125,325),(275,325), 2) # La bande

# La deuxième raquette
pygame.draw.ellipse(fenetre, Color('red'), ((20,50),(40,40))) # L'embout gauche
pygame.draw.ellipse(fenetre, Color('red'), ((100,50),(40,40))) # L'embout droit
pygame.draw.rect(fenetre, Color('gray'), ((40,50),(80,40))) # Le corps
pygame.draw.rect(fenetre, Color('yellow'), ((50,60),(12,8))) # Le voyant gauche
pygame.draw.rect(fenetre, Color('yellow'), ((98,60),(12,8))) # Le voyant droit
pygame.draw.line(fenetre, Color('black'), (50,80),(110,80), 2) # La bande

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
en_cours = True
while en_cours:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Clic sur la fermeture de fenêtre ?
            en_cours = False
pygame.quit() # Terminer pygame
