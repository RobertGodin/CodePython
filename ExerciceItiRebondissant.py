# -*- coding: utf-8 -*-
"""
Exercice du Iti qui rebondit
"""
# Importer la librairie de pygame et initialiser 
import pygame

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
pygame.display.set_caption("Exercice du Iti qui rebondit") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
x_iti = 175 # Position du Iti sur l'axe x
y_iti = 0 # Position initiale du Iti sur l'axe y
vitesse_deplacement = 5 # En pixels par scène
LARGEUR_ITI = 50
HAUTEUR_ITI = 100

# Boucle d'animation : Le Iti rebondit verticalement
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer le Iti : Inverser la direction sur le bord est atteint
        if y_iti+vitesse_deplacement > HAUTEUR_FENETRE-HAUTEUR_ITI or y_iti+vitesse_deplacement < 0 :
            vitesse_deplacement = -vitesse_deplacement # Inverser la direction    
        y_iti = y_iti+vitesse_deplacement
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        dessiner_iti(fenetre,x_iti,y_iti,LARGEUR_ITI,HAUTEUR_ITI) # Dessiner le Bot
        pygame.display.flip() # Mettre à jour la fenêtre graphique


        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame