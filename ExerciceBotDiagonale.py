# -*- coding: utf-8 -*-
"""
Exercice du Bot qui rebondit en diagonale
"""
# Importer la librairie de pygame et initialiser 
import pygame

def dessiner_bot(fenetre,x,y,largeur,hauteur):
    """ Dessiner un Bot. 
    
    Le Bot est inscrit dans le rectangle englobant défini par les paramètres
    (x,y,largeur et hauteur) dans une fenetre de Pygame
    
    fenetre : la surface de dessin
    x,y,largeur,hauteur : paramètres du rectangle englobant en pixels
    """
    ROUGE = (255,0,0)
    NOIR = (0,0,0)
    VERT = (0,255,0)    
    pygame.draw.ellipse(fenetre, VERT, [x,y,largeur, hauteur/2]) # Dessiner la tête
    pygame.draw.rect(fenetre, NOIR, [x+largeur/4,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil gauche
    pygame.draw.rect(fenetre, NOIR, [x+largeur*3/4-largeur/10,y+hauteur/8,largeur/10,hauteur/20]) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [x+largeur/4,y+hauteur*3/8],[x+largeur*3/4,y+hauteur*3/8], 2) # La bouche
    pygame.draw.rect(fenetre, ROUGE, [x,y+hauteur/2,largeur,hauteur/2]) # Le corps

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exercice du Bot qui rebondit en diagonale") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
x_bot = 0 # Position initiale du Bot sur l'axe x
y_bot = 0 # Position initiale du Bot sur l'axe y
vitesse_deplacement_x = 5 # En pixels par scène
vitesse_deplacement_y = 10
LARGEUR_BOT = 50
HAUTEUR_BOT = 100

# Boucle d'animation : Le Bot se déplace en diagonale
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer le Bot : Inverser la direction si le bord est atteint
        if x_bot+vitesse_deplacement_x > LARGEUR_FENETRE-LARGEUR_BOT or x_bot+vitesse_deplacement_x < 0 :
            vitesse_deplacement_x = -vitesse_deplacement_x # Inverser la direction en x    
        x_bot = x_bot+vitesse_deplacement_x
        if y_bot+vitesse_deplacement_y > HAUTEUR_FENETRE-HAUTEUR_BOT or y_bot+vitesse_deplacement_y < 0 :
            vitesse_deplacement_y = -vitesse_deplacement_y # Inverser la direction en y    
        y_bot = y_bot+vitesse_deplacement_y
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        dessiner_bot(fenetre,x_bot,y_bot,LARGEUR_BOT,HAUTEUR_BOT) # Dessiner le Bot
        pygame.display.flip() # Mettre à jour la fenêtre graphique


        horloge.tick(60) # Pour animer avec 60 images par seconde
 
pygame.quit() # Terminer pygame