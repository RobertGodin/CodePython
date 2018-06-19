# -*- coding: utf-8 -*-
"""
Exercice : bonhomme Iti
"""
# Importer la librairie de pygame et initialiser 
import pygame
pygame.init()

LARGEUR_FENETRE = 300
HAUTEUR_FENETRE = 300
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exercice bonhomme Iti avec pygame') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROSE = (255,100,100)
NOIR = (0,0,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

pygame.draw.ellipse(fenetre, ROSE, [133, 50, 33, 50]) # Dessiner la tête
pygame.draw.arc(fenetre,NOIR,[140,75,19,15],3.1416,0,1) # Le sourire
pygame.draw.ellipse(fenetre, NOIR, [138,66,8,8]) # L'oeil gauche
pygame.draw.ellipse(fenetre, NOIR, [154,66,8,8]) # L'oeil droit
pygame.draw.line(fenetre, NOIR, [150,100],[150,200], 2) # Le corps
pygame.draw.line(fenetre, NOIR, [100,100],[150,150], 2) # Bras gauche
pygame.draw.line(fenetre, NOIR, [200,100],[150,150], 2) # Bras droit
pygame.draw.line(fenetre, NOIR, [100,250],[150,200], 2) # Jambe gauche
pygame.draw.line(fenetre, NOIR, [200,250],[150,200], 2) # Jambe droite

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame