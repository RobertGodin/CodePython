# -*- coding: utf-8 -*-
"""
Exemple de jeu : programme principal
"""
import EntiteDuJeu
import pygame

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
EntiteDuJeu.EntiteAnimeeAvecSon.set_fenetre(fenetre)
pygame.display.set_caption("Exemple de jeu avec module EntiteDuJeu")

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de la liste des entités du jeu
liste_entite = []
liste_entite.append(EntiteDuJeu.BotAnime(10,100,40,80,3,3,"Son2.wav"))
liste_entite.append(EntiteDuJeu.BotAnime(200,200,50,100,0,2,"Son2.wav"))
liste_entite.append(EntiteDuJeu.ItiAnimeVolant(200,50,50,100,3,0,"Son3.wav",3))
liste_entite.append(EntiteDuJeu.EntiteAnimeeParImages(50,100,100,100,5,5,"Son4.wav",9,"coq"))

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        if event.type == pygame.MOUSEBUTTONUP : # Utilisateur a cliqué dans la fenêtre ?
            x=event.pos[0] # Position x de la souris
            y=event.pos[1] # Position y de la souris
            for une_entite in liste_entite :
                if une_entite.touche(x,y):
                    une_entite.emettre_son()
                    liste_entite.remove(une_entite)
                    
        for une_entite in liste_entite :
            une_entite.prochaine_scene()
        
        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        for une_entite in liste_entite :
            une_entite.dessiner()

        pygame.display.flip() # Mettre à jour la fenêtre graphique
        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame