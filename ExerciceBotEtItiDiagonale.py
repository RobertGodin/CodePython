# -*- coding: utf-8 -*-
"""
Exercice : 2 Bot et 2 Iti qui rebondissent en diagonale
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

def dessiner_iti(fenetre,x,y,largeur,hauteur):
    """ Dessiner un Iti. 
    
    Le Bot est inscrit dans le rectangle englobant défini par les paramètres
    (x,y,largeur et hauteur) dans une fenetre de Pygame
    
    fenetre : la surface de dessin
    x,y,largeur,hauteur : paramètres du rectangle englobant en pixels
    """
    ROSE = (255,100,100) # Couleur de la tête
    NOIR = (0,0,0) # Couleur du corps  

    # Coordonnées du milieu du rectangle englobant pour faciliter les calculs
    milieux = x + largeur/2;
    milieuy = y + hauteur/2;
    
    pygame.draw.ellipse(fenetre, ROSE, [x+largeur/3,y,largeur/3,hauteur/4]) # Dessiner la tête
    pygame.draw.arc(fenetre,NOIR,[milieux-largeur/12,y+hauteur/8,largeur/6,hauteur/14],3.1416,0,2) # Le sourire
    pygame.draw.ellipse(fenetre, NOIR, [milieux-largeur/8,y+hauteur/12,largeur/12,hauteur/24]) # L'oeil gauche
    pygame.draw.ellipse(fenetre, NOIR, [milieux+largeur/8-largeur/12,y+hauteur/12,largeur/12,hauteur/24]) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, [milieux,y+hauteur/4],[milieux,y+hauteur*3/4], 2) # Le corps
    pygame.draw.line(fenetre, NOIR, [x,y+hauteur/4],[milieux,milieuy], 2) # Bras gauche
    pygame.draw.line(fenetre, NOIR, [x+largeur,y+hauteur/4],[milieux,milieuy], 2) # Bras droit
    pygame.draw.line(fenetre, NOIR, [x,y+hauteur],[milieux,y+hauteur*3/4], 2) # Jambe gauche
    pygame.draw.line(fenetre, NOIR, [x+largeur,y+hauteur],[milieux,y+hauteur*3/4], 2) # Jambe droite

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exercice des Bots et Itis en diagonale") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Données du Bot1
x_bot1 = 0 # Position initiale du Bot1 sur l'axe x
y_bot1 = 0 # Position initiale du Bot1 sur l'axe y
vitesse_x_bot1 = 5 # En pixels par scène
vitesse_y_bot1 = 10
LARGEUR_BOT1 = 20
HAUTEUR_BOT1 = 40

# Données du Bot2
x_bot2 = 100
y_bot2 = 100
vitesse_x_bot2 = 10
vitesse_y_bot2 = 2
LARGEUR_BOT2 = 30
HAUTEUR_BOT2 = 60

# Données du Iti1
x_iti1 = 200
y_iti1 = 150
vitesse_x_iti1 = 3
vitesse_y_iti1 = 3
LARGEUR_ITI1 = 40
HAUTEUR_ITI1 = 80

# Données du Iti2
x_iti2 = 300 # Position initiale du Bot sur l'axe x
y_iti2 = 300 # Position initiale du Bot sur l'axe y
vitesse_x_iti2 = 5 # En pixels par scène
vitesse_y_iti2 = 10
LARGEUR_ITI2 = 50
HAUTEUR_ITI2 = 100

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer le Bot1
        if x_bot1+vitesse_x_bot1 > LARGEUR_FENETRE-LARGEUR_BOT1 or x_bot1+vitesse_x_bot1 < 0 :
            vitesse_x_bot1 = -vitesse_x_bot1 # Inverser la direction en x    
        x_bot1 = x_bot1+vitesse_x_bot1
        if y_bot1+vitesse_y_bot1 > HAUTEUR_FENETRE-HAUTEUR_BOT1 or y_bot1+vitesse_y_bot1 < 0 :
            vitesse_y_bot1 = -vitesse_y_bot1 # Inverser la direction en y    
        y_bot1 = y_bot1+vitesse_y_bot1
        
        # Déplacer le Bot2
        if x_bot2+vitesse_x_bot2 > LARGEUR_FENETRE-LARGEUR_BOT2 or x_bot2+vitesse_x_bot2 < 0 :
            vitesse_x_bot2 = -vitesse_x_bot2 # Inverser la direction en x    
        x_bot2 = x_bot2+vitesse_x_bot2
        if y_bot2+vitesse_y_bot2 > HAUTEUR_FENETRE-HAUTEUR_BOT2 or y_bot2+vitesse_y_bot2 < 0 :
            vitesse_y_bot2 = -vitesse_y_bot2 # Inverser la direction en y    
        y_bot2 = y_bot2+vitesse_y_bot2
        
        # Déplacer le Iti1
        if x_iti1+vitesse_x_iti1 > LARGEUR_FENETRE-LARGEUR_ITI1 or x_iti1+vitesse_x_iti1 < 0 :
            vitesse_x_iti1 = -vitesse_x_iti1 # Inverser la direction en x    
        x_iti1 = x_iti1+vitesse_x_iti1
        if y_iti1+vitesse_y_iti1 > HAUTEUR_FENETRE-HAUTEUR_ITI1 or y_iti1+vitesse_y_iti1 < 0 :
            vitesse_y_iti1 = -vitesse_y_iti1 # Inverser la direction en y    
        y_iti1 = y_iti1+vitesse_y_iti1
        
        # Déplacer le Iti2
        if x_iti2+vitesse_x_iti2 > LARGEUR_FENETRE-LARGEUR_ITI2 or x_iti2+vitesse_x_iti2 < 0 :
            vitesse_x_iti2 = -vitesse_x_iti2 # Inverser la direction en x    
        x_iti2 = x_iti2+vitesse_x_iti2
        if y_iti2+vitesse_y_iti2 > HAUTEUR_FENETRE-HAUTEUR_ITI2 or y_iti2+vitesse_y_iti2 < 0 :
            vitesse_y_iti2 = -vitesse_y_iti2 # Inverser la direction en y    
        y_iti2 = y_iti2+vitesse_y_iti2
        
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        dessiner_bot(fenetre,x_bot1,y_bot1,LARGEUR_BOT1,HAUTEUR_BOT1) # Dessiner le Bot1
        dessiner_bot(fenetre,x_bot2,y_bot2,LARGEUR_BOT2,HAUTEUR_BOT2) # Dessiner le Bot2
        dessiner_iti(fenetre,x_iti1,y_iti1,LARGEUR_ITI1,HAUTEUR_ITI1) # Dessiner le Iti1
        dessiner_iti(fenetre,x_iti2,y_iti2,LARGEUR_ITI2,HAUTEUR_ITI2) # Dessiner le Iti2
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame