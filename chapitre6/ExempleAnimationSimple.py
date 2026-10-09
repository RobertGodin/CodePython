"""
Exemple d'animation simple
"""
# Importer la librairie de pygame et initialiser
import pygame
from pygame import Color

def dessiner_raquette(fenetre,r):
    """ Dessiner une raquette.

    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """

    # Dessiner la raquette relativement au rectangle englobant r
    pygame.draw.ellipse(fenetre, Color('red'), ((r.x,r.y),(r.width/3,r.height))) # L'embout gauche
    pygame.draw.ellipse(fenetre, Color('red'), ((r.x+r.width*2/3,r.y),(r.width/3,r.height))) # L'embout droit
    pygame.draw.rect(fenetre, Color('gray'), ((r.x+r.width/6,r.y),(r.width*2/3,r.height))) # Le corps
    pygame.draw.rect(fenetre, Color('yellow'), ((r.x+r.width/4,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant gauche
    pygame.draw.rect(fenetre, Color('yellow'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/4),(r.width/10,r.height/5))) # Le voyant droit
    pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/4),(r.x+r.width*3/4,r.y+r.height*3/4), 2) # La bande

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption("Exemple d'animation simple") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
position_verticale = 500 # Position verticale de la raquette
VITESSE_DEPLACEMENT = 5 # En pixels par scène
TAILLE_RAQUETTE = (100,25)
# Boucle d'animation par une suite de scènes
# La raquette avance de la bordure gauche jusqu'à la droite
for position_horizontale in range(0,LARGEUR_FENETRE-TAILLE_RAQUETTE[0]+1,
                                  VITESSE_DEPLACEMENT) :
    pygame.event.pump() # Laisser le système traiter les évènements
    fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
    dessiner_raquette(fenetre,pygame.Rect((position_horizontale,position_verticale),TAILLE_RAQUETTE))
    pygame.display.flip() # Mettre à jour la fenêtre graphique
    horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame
