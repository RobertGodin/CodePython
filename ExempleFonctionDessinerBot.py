# -*- coding: utf-8 -*-
"""
Exemple de fonction dessiner_iti
"""

import pygame

def dessiner_iti(fenetre,x,y,largeur,hauteur):
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


pygame.init() # Initialiser Pygame
size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Dessiner deux Bots en appelant la fonction à deux reprises
dessiner_iti(fenetre,100,100,200,400)
dessiner_iti(fenetre,25,50,100,200)

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame