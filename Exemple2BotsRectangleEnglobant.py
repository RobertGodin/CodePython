# -*- coding: utf-8 -*-
"""
Exemple de dessin de 2 Bots dans un rectangle englobant
"""
# Importer la bibliothèque de pygame et initialiser 
import sys,pygame
from pygame import Color
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

# Rectangle englobant du premier Bot
r = pygame.Rect((100,100),(200,400)) # le rectangle englobant

# Dessiner le Bot relativement au rectangle englobant r
pygame.draw.ellipse(fenetre, Color('green'), ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
pygame.draw.rect(fenetre, Color('red'), ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps

# Rectangle englobant pour le deuxième Bot
r = pygame.Rect((25,50),(100,200)) # le rectangle englobant

# Dessiner le Bot relativement au rectangle englobant r
pygame.draw.ellipse(fenetre, Color('green'), ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
pygame.draw.rect(fenetre, Color('red'), ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
