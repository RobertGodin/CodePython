"""
Exemple de fonction dessiner_raquette
Nom de paramètre et d'argument différents
"""

import sys, pygame
from pygame import Color

def dessiner_raquette(f,r):
    """ Dessiner une raquette.

    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """

    # Dessiner la raquette relativement au rectangle englobant r
    pygame.draw.ellipse(f, Color('red'), ((r.x,r.y),(r.width/3,r.height))) # L'embout gauche
    pygame.draw.ellipse(f, Color('red'), ((r.x+r.width*2/3,r.y),(r.width/3,r.height))) # L'embout droit
    pygame.draw.rect(f, Color('gray'), ((r.x+r.width/6,r.y),(r.width*2/3,r.height))) # Le corps
    pygame.draw.rect(f, Color('yellow'), ((r.x+r.width/4,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant gauche
    pygame.draw.rect(f, Color('yellow'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant droit
    pygame.draw.line(f, Color('black'), (r.x+r.width/4,r.y+r.height*3/4),(r.x+r.width*3/4,r.y+r.height*3/4), 2) # La bande

pygame.init() # Initialiser Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption('Exemple de dessin de la raquette dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Dessiner deux raquettes en appelant la fonction à deux reprises
dessiner_raquette(fenetre, pygame.Rect((50,250),(300,100)))
dessiner_raquette(fenetre, pygame.Rect((20,50),(120,40)))

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
