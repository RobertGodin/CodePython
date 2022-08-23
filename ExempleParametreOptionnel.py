# -*- coding: utf-8 -*-
"""
Exemple de fonction dessiner_bot avec paramètre optionnel et valeur de défaut
"""

import sys, pygame
from pygame import Color

def dessiner_bot(fenetre,r,couleur_tete = Color('green'),couleur_corps = Color('red')):
    """ Dessiner un Bot. 
    
    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """
    
    # Dessiner le Bot relativement au rectangle englobant r
    pygame.draw.ellipse(fenetre, couleur_tete, ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
    pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
    pygame.draw.rect(fenetre, Color('black'), ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
    pygame.draw.line(fenetre, Color('black'), (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
    pygame.draw.rect(fenetre, couleur_corps, ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps
    
pygame.init() # Initialiser Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre  
pygame.display.set_caption('Exemple de dessin du Bot dans un rectangle englobant') # Définir le titre dans le haut de la fenêtre

fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin

# Variantes d'appel de fonction avec paramètres optionnels
dessiner_bot(fenetre,pygame.Rect((25,50),(100,200))) # Employer valeurs de défaut
BLEU = (0,0,255)
dessiner_bot(fenetre,pygame.Rect((25,350),(75,150)),couleur_tete = Color('blue')) # Spécifier une couleur pour la tete
dessiner_bot(fenetre, pygame.Rect((250,75),(100,200)),couleur_corps = Color('blue')) # Spécifier une couleur pour le corps
dessiner_bot(fenetre, pygame.Rect((250,300),(75,150)),couleur_corps = Color('blue'),couleur_tete = Color('red')) # Spécifier les deux 

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
