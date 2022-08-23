# -*- coding: utf-8 -*-
"""
Exemple de détection d’évènement de souris
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
pygame.display.set_caption('Exemple de gestion de la souris') # Définir le titre dans le haut de la fenêtre

fin = False 

# Position initiale du Bot
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
        dessiner_bot(fenetre,pygame.Rect((x-30/2,y-60/2),(30,60))) # Dessiner le Bot à la position de la souris
        pygame.display.flip() # Mettre à jour la fenêtre graphique
 
pygame.quit() # Terminer pygame