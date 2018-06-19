# -*- coding: utf-8 -*-
"""
Exemple de dessin 2D avec pygame
"""
# Importer la librairie de pygame et initialiser 
import pygame
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin 2D avec pygame') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
    
pygame.draw.ellipse(fenetre, VERT, [100, 100, 200, 200]) # Dessiner la tête
pygame.draw.rect(fenetre, NOIR, [150,150,20,20]) # L'oeil gauche
pygame.draw.rect(fenetre, NOIR, [230,150,20,20]) # L'oeil droit
pygame.draw.line(fenetre, NOIR, [150,250],[250,250], 2) # La bouche
pygame.draw.rect(fenetre, ROUGE, [100,300,200,200]) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer") # Pour retarder la fermeture de la fenêtre
pygame.quit() # Terminer pygame