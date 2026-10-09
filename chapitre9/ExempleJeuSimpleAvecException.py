"""
Exemple de jeu avec exceptions : programme principal
"""
import EntiteDuJeuAvecException
import pygame
from pygame import Color

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
EntiteDuJeuAvecException.EntiteAnimeeAvecSon.set_fenetre(fenetre)
pygame.display.set_caption("Casse-briques avec module EntiteDuJeu")
police = pygame.font.Font(None, 28) # Police par défaut de taille 28

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création des entités du jeu
try :
    raquette = EntiteDuJeuAvecException.Raquette(
        pygame.Rect((160,540),(80,12)),[6,0],"Son3.wav")
    balle = EntiteDuJeuAvecException.Balle(
        pygame.Rect((195,400),(10,10)),[3,-4],"Son3.wav")
    liste_briques = []
    couleurs = [Color('red'), Color('orange'), Color('gold'),
                Color('green'), Color('blue')]
    for rangee in range(5):
        for colonne in range(8):
            x = colonne*50+2
            y = 60+rangee*22
            rectangle = pygame.Rect((x,y),(46,18))
            if rangee == 0 : # La rangée du haut est plus résistante
                brique = EntiteDuJeuAvecException.BriqueResistante(
                    rectangle,"Son4.wav",couleurs[rangee],3)
            else :
                brique = EntiteDuJeuAvecException.Brique(
                    rectangle,"Son2.wav",couleurs[rangee])
            liste_briques.append(brique)
except EntiteDuJeuAvecException.CoordonneesEntiteErreur as e :
    print(f"Position invalide : {e}")
except EntiteDuJeuAvecException.TailleExcessiveErreur as e :
    print(f"Taille invalide : {e}")
except Exception :
    print("Une exception a été levée lors de la création des entités du jeu")
    raise
else :
    score = 0
    vies = 3
    message = "" # Devient "Victoire !" ou "Défaite..." à la fin

    # Boucle d'animation
    fin = False
    while not fin :
        for event in pygame.event.get(): # Tous les évènements
            if event.type == pygame.QUIT: # Fermeture de la fenêtre ?
                fin = True # Fin de la boucle du jeu
        if message == "" : # La partie est en cours
            raquette.prochaine_scene()
            balle.prochaine_scene()
            if balle.touche(raquette) and balle.v[1] > 0 :
                raquette.emettre_son()
                balle.rebondir_sur_raquette(raquette)
            for une_brique in list(liste_briques) : # Une copie
                if balle.touche(une_brique) :
                    balle.rebondir_sur(une_brique)
                    if une_brique.frapper() :
                        liste_briques.remove(une_brique)
                        score += 10
                    break # Un seul rebond par scène
            if balle.perdue() :
                vies -= 1
                balle.r.topleft = (195,400) # Balle remise en jeu
                balle.v = [3,-4]
            if len(liste_briques) == 0 :
                message = "Victoire !"
            elif vies == 0 :
                message = "Défaite..."

        fenetre.fill(Color('white')) # Dessiner le fond
        for une_brique in liste_briques :
            une_brique.dessiner()
        raquette.dessiner()
        balle.dessiner()
        texte = f"Score : {score}   Vies : {vies}"
        image_texte = police.render(texte, True, Color('black'))
        fenetre.blit(image_texte, (10,570))
        if message != "" :
            image_message = police.render(message, True, Color('red'))
            fenetre.blit(image_message, (150,300))

        pygame.display.flip() # Mettre à jour la fenêtre graphique
        horloge.tick(60) # Pour animer avec 60 images par seconde
finally:
    pygame.quit() # Terminer pygame
