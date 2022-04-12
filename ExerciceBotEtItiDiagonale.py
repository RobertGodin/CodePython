# -*- coding: utf-8 -*-
"""
Exercice : 2 Bot et 2 Iti qui rebondissent en diagonale
"""
# Importer la librairie de pygame et initialiser 
import pygame

def dessiner_bot(fenetre,r):
    """ Dessiner un Bot. 
    
    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """
    ROUGE = (255,0,0)
    NOIR = (0,0,0)
    VERT = (0,255,0)
    # Dessiner le Bot relativement au rectangle englobant r
    pygame.draw.ellipse(fenetre, VERT, ((r.x,r.y),(r.width, r.height/2))) # Dessiner la tête
    pygame.draw.rect(fenetre, NOIR, ((r.x+r.width/4,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil gauche
    pygame.draw.rect(fenetre, NOIR, ((r.x+r.width*3/4-r.width/10,r.y+r.height/8),(r.width/10,r.height/20))) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, (r.x+r.width/4,r.y+r.height*3/8),(r.x+r.width*3/4,r.y+r.height*3/8), 2) # La bouche
    pygame.draw.rect(fenetre, ROUGE, ((r.x,r.y+r.height/2),(r.width,r.height/2))) # Le corps

def dessiner_iti(fenetre,r):
    """ Dessiner un Iti. 
    
    fenetre : la surface de dessin
    r : rectangle englobant de type pygame.Rect
    """
    ROSE = (255,100,100) # Couleur de la tête
    NOIR = (0,0,0) # Couleur du corps  

    # Coordonnées du milieu du rectangle englobant pour faciliter les calculs
    milieux = r.x + r.width/2;
    milieuy = r.y + r.height/2;
    
    pygame.draw.ellipse(fenetre, ROSE, ((r.x+r.width/3,r.y),(r.width/3,r.height/4))) # Dessiner la tête
    pygame.draw.arc(fenetre,NOIR,((milieux-r.width/12,r.y+r.height/8),(r.width/6,r.height/14)),3.1416,0,2) # Le sourire
    pygame.draw.ellipse(fenetre, NOIR, ((milieux-r.width/8,r.y+r.height/12),(r.width/12,r.height/24))) # L'oeil gauche
    pygame.draw.ellipse(fenetre, NOIR, ((milieux+r.width/8-r.width/12,r.y+r.height/12),(r.width/12,r.height/24))) # L'oeil droit
    pygame.draw.line(fenetre, NOIR, (milieux,r.y+r.height/4),(milieux,r.y+r.height*3/4), 2) # Le corps
    pygame.draw.line(fenetre, NOIR, (r.x,r.y+r.height/4),(milieux,milieuy), 2) # Bras gauche
    pygame.draw.line(fenetre, NOIR, (r.x+r.width,r.y+r.height/4),(milieux,milieuy), 2) # Bras droit
    pygame.draw.line(fenetre, NOIR, (r.x,r.y+r.height),(milieux,r.y+r.height*3/4), 2) # Jambe gauche
    pygame.draw.line(fenetre, NOIR, (r.x+r.width,r.y+r.height),(milieux,r.y+r.height*3/4), 2) # Jambe droite

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exercice des Bots et Itis en diagonale") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Données du Bot1
(x_bot1,y_bot1) = (0,0) # Position initiale du Bot1 sur l'axe x
(vitesse_x_bot1,vitesse_y_bot1) = (5,10) # En pixels par scène
TAILLE_BOT1 = (20,40)

# Données du Bot2
(x_bot2,y_bot2) = (100,100)
(vitesse_x_bot2,vitesse_y_bot2) = (10,2) # En pixels par scène
TAILLE_BOT2 = (30,60)

# Données du Iti1
(x_iti1,y_iti1) = (200,150)
(vitesse_x_iti1,vitesse_y_iti1) = (10,2) # En pixels par scène
TAILLE_ITI1 = (40,80)

# Données du Iti2
(x_iti2,y_iti2) = (300,300)
(vitesse_x_iti2,vitesse_y_iti2) = (5,10) # En pixels par scène
TAILLE_ITI2 = (50,100)

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer le Bot1
        if x_bot1+vitesse_x_bot1 > LARGEUR_FENETRE-TAILLE_BOT1[0] or x_bot1+vitesse_x_bot1 < 0 :
            vitesse_x_bot1 = -vitesse_x_bot1 # Inverser la direction en x    
        x_bot1 = x_bot1+vitesse_x_bot1
        if y_bot1+vitesse_y_bot1 > HAUTEUR_FENETRE-TAILLE_BOT1[1] or y_bot1+vitesse_y_bot1 < 0 :
            vitesse_y_bot1 = -vitesse_y_bot1 # Inverser la direction en y    
        y_bot1 = y_bot1+vitesse_y_bot1
        
        # Déplacer le Bot2
        if x_bot2+vitesse_x_bot2 > LARGEUR_FENETRE-TAILLE_BOT2[0] or x_bot2+vitesse_x_bot2 < 0 :
            vitesse_x_bot2 = -vitesse_x_bot2 # Inverser la direction en x    
        x_bot2 = x_bot2+vitesse_x_bot2
        if y_bot2+vitesse_y_bot2 > HAUTEUR_FENETRE-TAILLE_BOT2[1] or y_bot2+vitesse_y_bot2 < 0 :
            vitesse_y_bot2 = -vitesse_y_bot2 # Inverser la direction en y    
        y_bot2 = y_bot2+vitesse_y_bot2
        
        # Déplacer le Iti1
        if x_iti1+vitesse_x_iti1 > LARGEUR_FENETRE-TAILLE_ITI1[0] or x_iti1+vitesse_x_iti1 < 0 :
            vitesse_x_iti1 = -vitesse_x_iti1 # Inverser la direction en x    
        x_iti1 = x_iti1+vitesse_x_iti1
        if y_iti1+vitesse_y_iti1 > HAUTEUR_FENETRE-TAILLE_ITI1[1] or y_iti1+vitesse_y_iti1 < 0 :
            vitesse_y_iti1 = -vitesse_y_iti1 # Inverser la direction en y    
        y_iti1 = y_iti1+vitesse_y_iti1
        
        # Déplacer le Iti2
        if x_iti2+vitesse_x_iti2 > LARGEUR_FENETRE-TAILLE_ITI2[0] or x_iti2+vitesse_x_iti2 < 0 :
            vitesse_x_iti2 = -vitesse_x_iti2 # Inverser la direction en x    
        x_iti2 = x_iti2+vitesse_x_iti2
        if y_iti2+vitesse_y_iti2 > HAUTEUR_FENETRE-TAILLE_ITI2[1] or y_iti2+vitesse_y_iti2 < 0 :
            vitesse_y_iti2 = -vitesse_y_iti2 # Inverser la direction en y    
        y_iti2 = y_iti2+vitesse_y_iti2
        
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        dessiner_bot(fenetre,pygame.Rect((x_bot1,y_bot1),TAILLE_BOT1)) # Dessiner le Bot1
        dessiner_bot(fenetre,pygame.Rect((x_bot2,y_bot2),TAILLE_BOT2)) # Dessiner le Bot2
        dessiner_iti(fenetre,pygame.Rect((x_iti1,y_iti1),TAILLE_ITI1)) # Dessiner le Iti1
        dessiner_iti(fenetre,pygame.Rect((x_iti2,y_iti2),TAILLE_ITI2)) # Dessiner le Iti2
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images par seconde
 
pygame.quit() # Terminer pygame