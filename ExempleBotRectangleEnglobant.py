# -*- coding: utf-8 -*-
"""
Exemple de dessin du Bot dans un rectangle englobant
"""
# Importer la bibliothèque de pygame et initialiser 
import pygame
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Constantes qui représentent le rectangle englobant
X = 100
Y = 100
LARGEUR_BOT = 200
HAUTEUR_BOT = 400

# Dessiner le Bot relativement au rectangle englobant
pygame.draw.ellipse(fenetre, VERT, ((X,Y),(LARGEUR_BOT, HAUTEUR_BOT/2))) # Dessiner la tête
pygame.draw.rect(fenetre, NOIR, ((X+LARGEUR_BOT/4,Y+HAUTEUR_BOT/8),(LARGEUR_BOT/10,HAUTEUR_BOT/20))) # L'oeil gauche
pygame.draw.rect(fenetre, NOIR, ((X+LARGEUR_BOT*3/4-LARGEUR_BOT/10,Y+HAUTEUR_BOT/8),(LARGEUR_BOT/10,HAUTEUR_BOT/20))) # L'oeil droit
pygame.draw.line(fenetre, NOIR, (X+LARGEUR_BOT/4,Y+HAUTEUR_BOT*3/8),(X+LARGEUR_BOT*3/4,Y+HAUTEUR_BOT*3/8), 2) # La bouche
pygame.draw.rect(fenetre, ROUGE, ((X,Y+HAUTEUR_BOT/2),(LARGEUR_BOT,HAUTEUR_BOT/2))) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame
