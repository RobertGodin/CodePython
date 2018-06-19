# -*- coding: utf-8 -*-
"""
Exemple de dessin 2D avec pygame
"""
# Importer la librairie de pygame et initialiser 
import pygame

def dessiner_bot(fenetre,x,y,largeur,hauteur):
    """ Dessiner un Bot. 
    
    Le Bot est inscrit dans le rectangle englobant défini par les paramètres
    (x,y,largeur et hauteur) dans une fenetre de Pygame
    
    fenetre : la surface de dessin
    x,y,largeur,hauteur : paramètres du rectangle englobant en pixels
    """
    ROUGE = (255,0,0)
    NOIR = (0,0,0)
    VERT = (0,255,0)    
    pygame.draw.ellipse(fenetre, VERT, [x,y,largeur, hauteur/2]) # Dessiner la tête
    pygame.draw.rect(fenetre, NOIR, [x+largeur/4,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil gauche
    pygame.draw.rect(fenetre, NOIR, [x+largeur*3/4-largeur/10,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [x+largeur/4,y+hauteur*3/8],[x+largeur*3/4,y+hauteur*3/8], 2) # La bouche
    pygame.draw.rect(fenetre, ROUGE, [x,y+hauteur/2,largeur,hauteur/2]) # Le corps

pygame.init() # Initialiser les modules de Pygame

size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouvrir la fenêtre 
pygame.display.set_caption('Exemple de gestion de la souris avec Pygame') # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
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

    fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
    dessiner_bot(fenetre,x-30/2,y-60/2,30,60) # Dessiner le Bot à la position de la souris

    pygame.display.flip() # Mettre à jour la fenêtre graphique
 
pygame.quit() # Terminer pygame