"""
Exemple de détection d’évènement de souris
"""
# Importer la librairie de pygame et initialiser
import pygame
from pygame import Color

def dessiner_raquette(fenetre,r):
    """ Dessiner une raquette.

    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """
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
pygame.display.set_caption('Exemple de gestion de la souris') # Définir le titre dans le haut de la fenêtre

fin = False

# Position initiale de la raquette
x=100
y=100
# Itérer jusqu'à ce qu'un évènement provoque la fermeture de la fenêtre
while not fin:
    event = pygame.event.wait() # Chercher le prochain évènement à traiter
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    elif event.type == pygame.MOUSEBUTTONUP: # Utilisateur a cliqué dans la fenêtre ?
        x=event.pos[0] # Position x de la souris
        y=event.pos[1] # Position y de la souris
        fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
        dessiner_raquette(fenetre,pygame.Rect((x-80/2,y-20/2),(80,20))) # Dessiner la raquette à la position de la souris
        pygame.display.flip() # Mettre à jour la fenêtre graphique

pygame.quit() # Terminer pygame
