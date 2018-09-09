# -*- coding: utf-8 -*-
"""
Exemple de jeu avec exceptions: programme principal
"""
import EntiteDuJeuAvecException
import pygame

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
EntiteDuJeuAvecException.EntiteAnimeeAvecSon.set_fenetre(fenetre)
pygame.display.set_caption("Exemple de jeu avec module EntiteDuJeu")

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de la liste des entités du jeu
liste_entite = []
try :
    liste_entite.append(EntiteDuJeuAvecException.BotAnime(10,100,40,80,3,3,"Son9.wav"))
    liste_entite.append(EntiteDuJeuAvecException.BotAnime(200,200,50,100,0,2,"Son2.wav"))
    liste_entite.append(EntiteDuJeuAvecException.ItiAnimeVolant(200,50,50,100,3,0,"Son3.wav",3))
    liste_entite.append(EntiteDuJeuAvecException.EntiteAnimeeParImages(50,100,100,100,5,5,"Son4.wav",9,"coq"))
except EntiteDuJeuAvecException.CoordonneesEntiteErreur as e :
    print("Les coordonnées de l'entité dépassent la taille de la fenetre",e)
except EntiteDuJeuAvecException.TailleExcessiveErreur as e :
    print("L'entité a une taille excessive par rapport à la taille de la fenetre",e)
except :
    print("Une exception a été levée lors de la création des entités du jeu")
    raise
else :

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
finally : 
    pygame.quit() # Terminer pygame