# -*- coding: utf-8 -*-
"""
Exemple d'animation simple
"""
# Importer la librairie de pygame et initialiser 
import pygame
from pygame import Color

def dessiner_bot(fenetre,r):
    """ Dessiner un Bot. 
    
    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """

    # Dessiner le Bot relativement au rectangle englobant r
    pygame.draw.ellipse(fenetre, Color('green'), ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
    pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
    pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
    pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
    pygame.draw.rect(fenetre, Color('red'), ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exemple d'animation simple") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
position_verticale = 100 # Position verticale du Bot
VITESSE_DEPLACEMENT = 5 # En pixels par scène
TAILLE_BOT = (50,100)
# Boucle d'animation par une suite de scènes
# Le Bot avance de la bordure gauche jusqu'à la droite
for position_horizontale in range(0,LARGEUR_FENETRE-TAILLE_BOT[0],VITESSE_DEPLACEMENT) :
    fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
    dessiner_bot(fenetre,pygame.Rect((position_horizontale,position_verticale),TAILLE_BOT))
    pygame.display.flip() # Mettre à jour la fenêtre graphique
    horloge.tick(60) # Pour animer avec 60 images par seconde
 
pygame.quit() # Terminer pygame