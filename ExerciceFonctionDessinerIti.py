# -*- coding: utf-8 -*-
"""
Exercice: fonction dessiner_iti
"""

import pygame

def dessiner_iti(fenetre,r):
    """ Dessiner un Iti. 
    
    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """
    ROSE = (255,100,100) # Couleur de la tête
    NOIR = (0,0,0) # Couleur du corps  

    # Coordonnées du milieu du rectangle englobant pour faciliter les calculs
    milieux = r.x + r.width/2;
    milieuy = r.y + r.height/2;
    
    pygame.draw.ellipse(fenetre, ROSE, ((r.x+r.width/3,r.y),(r.width/3,r.height/4))) # Dessiner la tête
    pygame.draw.arc(fenetre,NOIR,((milieux-r.width/12,r.y+r.height/8),(r.width/6,r.height/14)),3.1416,0,2) # Le sourire
    pygame.draw.ellipse(fenetre, NOIR, ((milieux-r.width/8,r.y+r.height/12),(r.width/12,r.height/24))) # L'oeil gauche
    pygame.draw.ellipse(fenetre, NOIR, ((milieux+r.width/8-r.width/12,r.y+r.height/12),(r.width/12,r.height/24))) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, (milieux,r.y+r.height/4),(milieux,r.y+r.height*3/4), 2) # Le corps
    pygame.draw.line(fenetre, NOIR, (r.x,r.y+r.height/4),(milieux,milieuy), 2) # Bras gauche
    pygame.draw.line(fenetre, NOIR, (r.x+r.width,r.y+r.height/4),(milieux,milieuy), 2) # Bras droit
    pygame.draw.line(fenetre, NOIR, (r.x,r.y+r.height),(milieux,r.y+r.height*3/4), 2) # Jambe gauche
    pygame.draw.line(fenetre, NOIR, (r.x+r.width,r.y+r.height),(milieux,r.y+r.height*3/4), 2) # Jambe droite

pygame.init() # Initialiser Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
pygame.display.set_caption('Exemple de dessin du Iti dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre


BLANC = (255,255,255)
fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Dessiner deux Iti en appelant la fonction à deux reprises
dessiner_iti(fenetre,pygame.Rect((100,100),(30,60)))
dessiner_iti(fenetre,pygame.Rect((25,50),(100,200)))

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame