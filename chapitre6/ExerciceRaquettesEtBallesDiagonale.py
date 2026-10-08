"""
Exercice : 2 raquettes et 2 balles qui rebondissent en diagonale
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
pygame.display.set_caption("Exercice des raquettes et des balles en diagonale") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Données de la raquette1
(x_raquette1,y_raquette1) = (0,500) # Position initiale de la raquette1
(vitesse_x_raquette1,vitesse_y_raquette1) = (5,0) # En pixels par scène
TAILLE_RAQUETTE1 = (80,20)

# Données de la raquette2
(x_raquette2,y_raquette2) = (100,550)
(vitesse_x_raquette2,vitesse_y_raquette2) = (3,0) # En pixels par scène
TAILLE_RAQUETTE2 = (120,30)

# Données de la balle1
(x_balle1,y_balle1) = (200,150)
(vitesse_x_balle1,vitesse_y_balle1) = (10,2) # En pixels par scène
TAILLE_BALLE1 = (40,40)

# Données de la balle2
(x_balle2,y_balle2) = (300,300)
(vitesse_x_balle2,vitesse_y_balle2) = (5,10) # En pixels par scène
TAILLE_BALLE2 = (50,50)

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer la raquette1
        if x_raquette1+vitesse_x_raquette1 > LARGEUR_FENETRE-TAILLE_RAQUETTE1[0] or x_raquette1+vitesse_x_raquette1 < 0 :
            vitesse_x_raquette1 = -vitesse_x_raquette1 # Inverser la direction en x
        x_raquette1 = x_raquette1+vitesse_x_raquette1
        if y_raquette1+vitesse_y_raquette1 > HAUTEUR_FENETRE-TAILLE_RAQUETTE1[1] or y_raquette1+vitesse_y_raquette1 < 0 :
            vitesse_y_raquette1 = -vitesse_y_raquette1 # Inverser la direction en y
        y_raquette1 = y_raquette1+vitesse_y_raquette1

        # Déplacer la raquette2
        if x_raquette2+vitesse_x_raquette2 > LARGEUR_FENETRE-TAILLE_RAQUETTE2[0] or x_raquette2+vitesse_x_raquette2 < 0 :
            vitesse_x_raquette2 = -vitesse_x_raquette2 # Inverser la direction en x
        x_raquette2 = x_raquette2+vitesse_x_raquette2
        if y_raquette2+vitesse_y_raquette2 > HAUTEUR_FENETRE-TAILLE_RAQUETTE2[1] or y_raquette2+vitesse_y_raquette2 < 0 :
            vitesse_y_raquette2 = -vitesse_y_raquette2 # Inverser la direction en y
        y_raquette2 = y_raquette2+vitesse_y_raquette2

        # Déplacer la balle1
        if x_balle1+vitesse_x_balle1 > LARGEUR_FENETRE-TAILLE_BALLE1[0] or x_balle1+vitesse_x_balle1 < 0 :
            vitesse_x_balle1 = -vitesse_x_balle1 # Inverser la direction en x
        x_balle1 = x_balle1+vitesse_x_balle1
        if y_balle1+vitesse_y_balle1 > HAUTEUR_FENETRE-TAILLE_BALLE1[1] or y_balle1+vitesse_y_balle1 < 0 :
            vitesse_y_balle1 = -vitesse_y_balle1 # Inverser la direction en y
        y_balle1 = y_balle1+vitesse_y_balle1

        # Déplacer la balle2
        if x_balle2+vitesse_x_balle2 > LARGEUR_FENETRE-TAILLE_BALLE2[0] or x_balle2+vitesse_x_balle2 < 0 :
            vitesse_x_balle2 = -vitesse_x_balle2 # Inverser la direction en x
        x_balle2 = x_balle2+vitesse_x_balle2
        if y_balle2+vitesse_y_balle2 > HAUTEUR_FENETRE-TAILLE_BALLE2[1] or y_balle2+vitesse_y_balle2 < 0 :
            vitesse_y_balle2 = -vitesse_y_balle2 # Inverser la direction en y
        y_balle2 = y_balle2+vitesse_y_balle2


        fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
        dessiner_raquette(fenetre,pygame.Rect((x_raquette1,y_raquette1),TAILLE_RAQUETTE1)) # Dessiner la raquette1
        dessiner_raquette(fenetre,pygame.Rect((x_raquette2,y_raquette2),TAILLE_RAQUETTE2)) # Dessiner la raquette2
        dessiner_balle(fenetre,pygame.Rect((x_balle1,y_balle1),TAILLE_BALLE1)) # Dessiner la balle1
        dessiner_balle(fenetre,pygame.Rect((x_balle2,y_balle2),TAILLE_BALLE2)) # Dessiner la balle2
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame

