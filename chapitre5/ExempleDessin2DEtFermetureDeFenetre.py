"""
Exemple de dessin 2D avec pygame
"""
# Importer la librairie de pygame et initialiser
import sys, pygame
pygame.init()

LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre

pygame.display.set_caption('Exemple de dessin 2D avec pygame') # Définir le titre dans le haut de la fenêtre

# Définir les couleurs employées dans le dessin
BLANC = (255,255,255)
ROUGE = (255,0,0)
NOIR = (0,0,0)
GRIS = (128,128,128)
JAUNE = (255,255,0)

fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin

pygame.draw.ellipse(fenetre, ROUGE, ((50,250),(100,100))) # L'embout gauche
pygame.draw.ellipse(fenetre, ROUGE, ((250,250),(100,100))) # L'embout droit
pygame.draw.rect(fenetre, GRIS, ((100,250),(200,100))) # Le corps
pygame.draw.rect(fenetre, JAUNE, ((125,275),(30,20))) # Le voyant gauche
pygame.draw.rect(fenetre, JAUNE, ((245,275),(30,20))) # Le voyant droit
pygame.draw.line(fenetre, NOIR, (125,325),(275,325),2) # La bande

pygame.display.flip() # Mettre à jour la fenêtre graphique

# Traiter la fermeture de la fenêtre
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Vérifier si l'utilisateur a cliqué pour fermer la fenêtre
            pygame.quit() # Terminer pygame
            sys.exit()
