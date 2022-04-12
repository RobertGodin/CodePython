# -*- coding: utf-8 -*-
"""
Exercice du Bot qui rebondit
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

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exercice du Bot qui rebondit") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes
(x_bot,y_bot) = (175,0) # Position du Bot 
vitesse_deplacement = 5 # En pixels par scène
TAILLE_BOT = (50,100)

# Boucle d'animation : Le Bot rebondit verticalement
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        # Déplacer le Bot : Inverser la direction sur le bord est atteint
        if y_bot+vitesse_deplacement > HAUTEUR_FENETRE-TAILLE_BOT[1] or y_bot+vitesse_deplacement < 0 :
            vitesse_deplacement = -vitesse_deplacement # Inverser la direction    
        y_bot = y_bot+vitesse_deplacement
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        dessiner_bot(fenetre,pygame.Rect((x_bot,y_bot),TAILLE_BOT)) # Dessiner le Bot
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images par seconde
 
pygame.quit() # Terminer pygame