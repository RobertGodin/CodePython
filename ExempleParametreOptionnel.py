# -*- coding: utf-8 -*-
"""
Exemple de fonction dessiner_bot avec paramètre optionnel
"""

import pygame

VERT = (0,255,0)    
ROUGE = (255,0,0)

def dessiner_bot(fenetre,x,y,largeur,hauteur, couleur_tete = VERT, couleur_corps = ROUGE):
    """ Dessiner un Bot. 
    
    Le Bot est inscrit dans le rectangle englobant défini par les paramètres
    (x,y,largeur et hauteur) dans une fenetre de Pygame
    
    fenetre : la surface de dessin
    x,y,largeur,hauteur : paramètres du rectangle englobant en pixels
    """
    NOIR = (0,0,0)
    pygame.draw.ellipse(fenetre, couleur_tete, [x,y,largeur, hauteur/2]) # Dessiner la tête
    pygame.draw.rect(fenetre, NOIR, [x+largeur/4,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil gauche
    pygame.draw.rect(fenetre, NOIR, [x+largeur*3/4-largeur/10,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [x+largeur/4,y+hauteur*3/8],[x+largeur*3/4,y+hauteur*3/8], 2) # La bouche
    pygame.draw.rect(fenetre, couleur_corps, [x,y+hauteur/2,largeur,hauteur/2]) # Le corps


pygame.init() # Initialiser Pygame
size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)


fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Variantes d'appel de fonction avec paramètres optionnels
dessiner_bot(fenetre,25,50,100,200) # Employer valeurs de défaut
BLEU = (0,0,255)
dessiner_bot(fenetre,25,350,75,150,couleur_tete = BLEU) # Spécifier une couleur pour la tete
dessiner_bot(fenetre, 250,75,100,200,couleur_corps = BLEU) # Spécifier une couleur pour le corps
dessiner_bot(fenetre, 250, 300, 75, 150, couleur_corps = BLEU, couleur_tete = ROUGE) # Spécifier les deux 

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame
