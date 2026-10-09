"""
Exercice de la raquette qui rebondit
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
pygame.display.set_caption("Exercice de la raquette qui rebondit") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
(x_raquette,y_raquette) = (0,500) # Position de la raquette
vitesse_deplacement = 5 # En pixels par scène
TAILLE_RAQUETTE = (100,25)

# Boucle d'animation : La raquette rebondit horizontalement
fin = False
while not fin :
    for event in pygame.event.get(): # Traiter tous les évènements
        if event.type == pygame.QUIT: # Fermeture de la fenêtre ?
            fin = True  # Fin de la boucle du jeu
    # Déplacer la raquette : Inverser la direction sur le bord est atteint
    if x_raquette+vitesse_deplacement > LARGEUR_FENETRE-TAILLE_RAQUETTE[0] or x_raquette+vitesse_deplacement < 0 :
        vitesse_deplacement = -vitesse_deplacement # Inverser la direction
    x_raquette += vitesse_deplacement

    fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
    dessiner_raquette(fenetre,pygame.Rect((x_raquette,y_raquette),TAILLE_RAQUETTE)) # Dessiner la raquette
    pygame.display.flip() # Mettre à jour la fenêtre graphique

    horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame
