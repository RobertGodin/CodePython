# -*- coding: utf-8 -*-
"""
Exemple d'animation d'entités avec module Entite
"""
import Entite
import pygame

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
Entite.EntiteAnime.set_fenetre(fenetre)
pygame.display.set_caption("Exemple des Bots et Itis en diagonale avec module Entite")

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de deux BotAnime et deux ItiAnime
bot1 = Entite.BotAnime(0,0,20,40,5,10)
bot2 = Entite.BotAnime(100,200,30,60,10,2)
iti1 = Entite.ItiAnime(200,150,40,80,3,3)
iti2 = Entite.ItiAnime(300,300,50,100,5,10)

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        bot1.deplacer()
        bot2.deplacer()
        iti1.deplacer()
        iti2.deplacer()

        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        bot1.dessiner()
        bot2.dessiner()
        iti1.dessiner()
        iti2.dessiner()
        
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame