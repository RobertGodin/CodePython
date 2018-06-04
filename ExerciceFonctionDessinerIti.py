# -*- coding: utf-8 -*-
"""
Exercice: fonction dessiner_iti
"""

import pygame

def dessiner_iti(fenetre,x,y,largeur,hauteur):
    """ Dessiner un Bot. 
    
    Le Bot est inscrit dans le rectangle englobant défini par les paramètres
    (x,y,largeur et hauteur) dans une fenetre de Pygame
    
    fenetre : la surface de dessin
    x,y,largeur,hauteur : paramètres du rectangle englobant en pixels
    """
    ROSE = (255,100,100) # Couleur de la tête
    NOIR = (0,0,0) # Couleur du corps  

    # Coordonnées du milieu du rectangle englobant pour faciliter les calculs
    milieux = x + largeur/2;
    milieuy = y + hauteur/2;
    
    pygame.draw.ellipse(fenetre, ROSE, [x+largeur/3,y,largeur/3,hauteur/4]) # Dessiner la tête
    pygame.draw.arc(fenetre,NOIR,[milieux-largeur/12,y+hauteur/8,largeur/6,hauteur/14],3.1416,0,2) # Le sourire
    pygame.draw.ellipse(fenetre, NOIR, [milieux-largeur/8,y+hauteur/12,largeur/12,hauteur/24]) # L'oeil gauche
    pygame.draw.ellipse(fenetre, NOIR, [milieux+largeur/8-largeur/12,y+hauteur/12,largeur/12,hauteur/24]) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [milieux,y+hauteur/4],[milieux,y+hauteur*3/4], 2) # Le corps
    pygame.draw.line(fenetre, NOIR, [x,y+hauteur/4],[milieux,milieuy], 2) # Bras gauche
    pygame.draw.line(fenetre, NOIR, [x+largeur,y+hauteur/4],[milieux,milieuy], 2) # Bras droit
    pygame.draw.line(fenetre, NOIR, [x,y+hauteur],[milieux,y+hauteur*3/4], 2) # Jambe gauche
    pygame.draw.line(fenetre, NOIR, [x+largeur,y+hauteur],[milieux,y+hauteur*3/4], 2) # Jambe droite

pygame.init() # Initialiser Pygame
size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Iti dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre


BLANC = (255,255,255)
fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Dessiner deux Iti en appelant la fonction à deux reprises
dessiner_iti(fenetre,100,100,200,400)
dessiner_iti(fenetre,25,50,100,200)

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame