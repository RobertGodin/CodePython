# -*- coding: utf-8 -*-
"""
Exemple de dessin du Bot dans un rectangle englobant
"""
# Importer la bibliothèque de pygame et initialiser 
import sys,pygame
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 

pygame.display.set_caption('Er.xemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

r = pygame.Rect((100,100),(200,400)) # le rectangle englobant

# Dessiner le Bot relativement au rectangle englobant r
pygame.draw.ellipse(fenetre, VERT, ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
pygame.draw.rect(fenetre, NOIR, ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
pygame.draw.rect(fenetre, NOIR, ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
pygame.draw.line(fenetre, NOIR, (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
pygame.draw.rect(fenetre, ROUGE, ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
