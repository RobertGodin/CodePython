"""
Exercice de la balle qui rebondit
"""
# Importer la librairie de pygame et initialiser
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

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption("Exercice de la balle qui rebondit") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
x_balle = 175 # Position de la balle sur l'axe x
y_balle = 0 # Position initiale de la balle sur l'axe y
vitesse_deplacement = 5 # En pixels par scène
TAILLE_BALLE = (50,50)

# Boucle d'animation : La balle rebondit verticalement
fin = False
while not fin :
    for event in pygame.event.get(): # Traiter tous les évènements
        if event.type == pygame.QUIT: # Fermeture de la fenêtre ?
            fin = True  # Fin de la boucle du jeu
    # Déplacer la balle : Inverser la direction sur le bord est atteint
    if y_balle+vitesse_deplacement > HAUTEUR_FENETRE-TAILLE_BALLE[1] or y_balle+vitesse_deplacement < 0 :
        vitesse_deplacement = -vitesse_deplacement # Inverser la direction
    y_balle += vitesse_deplacement

    fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
    dessiner_balle(fenetre,pygame.Rect((x_balle,y_balle),TAILLE_BALLE)) # Dessiner la balle
    pygame.display.flip() # Mettre à jour la fenêtre graphique

    horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame
