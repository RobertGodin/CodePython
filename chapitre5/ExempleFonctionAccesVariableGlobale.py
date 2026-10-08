"""
Exemple de fonction dessiner_raquette
"""

import sys, pygame
from pygame import Color

def dessiner_raquette(r):
    """ Dessiner une raquette.

    r : rectangle englobant de type pygame.Rect
    """

    # Dessiner la raquette relativement au rectangle englobant r
    pygame.draw.ellipse(fenetre, Color('red'), ((r.x,r.y),(r.width/3,r.height))) # L'embout gauche
    pygame.draw.ellipse(fenetre, Color('red'), ((r.x+r.width*2/3,r.y),(r.width/3,r.height))) # L'embout droit
    pygame.draw.rect(fenetre, Color('gray'), ((r.x+r.width/6,r.y),(r.width*2/3,r.height))) # Le corps
    pygame.draw.rect(fenetre, Color('yellow'), ((r.x+r.width/4,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant gauche
    pygame.draw.rect(fenetre, Color('yellow'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant droit
    pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/4),(r.x+r.width*3/4,r.y+r.height*3/4), 2) # La bande

pygame.init() # Initialiser Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption('Exemple de dessin de la raquette dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

# Dessiner deux raquettes en appelant la fonction à deux reprises
dessiner_raquette(pygame.Rect((50,250),(300,100)))
dessiner_raquette(pygame.Rect((20,50),(120,40)))

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
