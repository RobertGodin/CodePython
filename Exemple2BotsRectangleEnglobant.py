# -*- coding: utf-8 -*-
"""
Exemple de dessin de 2 Bots dans un rectangle englobant
"""
# Importer la librairie de pygame et initialiser 
import pygame
pygame.init()

size = (400, 600) # Taille de la surface graphique
fenetre = pygame.display.set_mode(size) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

# Rectangle englobant du premier Bot
x = 100
y = 100
largeur = 200
hauteur = 400

# Dessiner le Bot relativement au rectangle englobant
pygame.draw.ellipse(fenetre, VERT, [x,y,largeur, hauteur/2]) # Dessiner la tête
pygame.draw.rect(fenetre, NOIR, [x+largeur/4,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil gauche
pygame.draw.rect(fenetre, NOIR, [x+largeur*3/4-largeur/10,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil droit
pygame.draw.line(fenetre, NOIR, [x+largeur/4,y+hauteur*3/8],[x+largeur*3/4,y+hauteur*3/8], 2) # La bouche
pygame.draw.rect(fenetre, ROUGE, [x,y+hauteur/2,largeur,hauteur/2]) # Le corps

# Rectangle englobant pour le deuxième Bot
x = 25
y = 50
largeur = 100
hauteur = 200

# Dessiner le Bot relativement au rectangle englobant
pygame.draw.ellipse(fenetre, VERT, [x,y,largeur, hauteur/2]) # Dessiner la tête
pygame.draw.rect(fenetre, NOIR, [x+largeur/4,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil gauche
pygame.draw.rect(fenetre, NOIR, [x+largeur*3/4-largeur/10,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil droit
pygame.draw.line(fenetre, NOIR, [x+largeur/4,y+hauteur*3/8],[x+largeur*3/4,y+hauteur*3/8], 2) # La bouche
pygame.draw.rect(fenetre, ROUGE, [x,y+hauteur/2,largeur,hauteur/2]) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique
input("Entrez fin de ligne pour terminer")
pygame.quit() # Terminer pygame
