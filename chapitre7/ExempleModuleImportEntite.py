"""
Exemple d'animation d'entités avec module Entite
"""
import Entite
import pygame
from pygame import Color

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
Entite.EntiteAnimee.set_fenetre(fenetre)
pygame.display.set_caption("Exemple des raquettes et des balles animées en diagonale avec super-classe EntiteAnimee") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Placer deux RaquetteAnimee et deux BalleAnimee dans la liste des entités
liste_entite = []
liste_entite.append(Entite.RaquetteAnimee(pygame.Rect((0,500),(80,20)),[5,0]))
liste_entite.append(Entite.RaquetteAnimee(pygame.Rect((100,550),(120,30)),[3,0]))
liste_entite.append(Entite.BalleAnimee(pygame.Rect((200,150),(40,40)),[3,3]))
liste_entite.append(Entite.BalleAnimee(pygame.Rect((300,300),(50,50)),[5,10]))

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        for une_entite in liste_entite :
            une_entite.deplacer()

        fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
        for une_entite in liste_entite :
            une_entite.dessiner()
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame
