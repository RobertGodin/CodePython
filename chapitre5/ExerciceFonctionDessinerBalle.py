"""
Exercice: fonction dessiner_balle
"""

import pygame
from pygame import Color

def dessiner_balle(fenetre,r):
    """ Dessiner une balle.

    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """


    pygame.draw.ellipse(fenetre, Color('darkblue'), r) # La balle
    pygame.draw.ellipse(fenetre, Color('white'), ((r.x+r.width/4,r.y+r.height/4),(r.width/6,r.height/6))) # Le reflet
    pygame.draw.arc(fenetre, Color('white'), ((r.x+r.width/6,r.y+r.height/6),(r.width*2/3,r.height*2/3)), 1.7, 3.0, 2) # L'éclat
    pygame.draw.ellipse(fenetre, Color('black'), r, 2) # Le contour

pygame.init() # Initialiser Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption('Exemple de dessin de la balle dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

# Dessiner deux balles en appelant la fonction à deux reprises
dessiner_balle(fenetre,pygame.Rect((100,100),(30,30)))
dessiner_balle(fenetre,pygame.Rect((25,50),(100,100)))

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
en_cours = True
while en_cours:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Clic sur la fermeture de fenêtre ?
            en_cours = False
pygame.quit() # Terminer pygame
