"""
Exemple d'animation d'entités : création des classes RaquetteAnimee et BalleAnimee
avec variable de classe pour la fenetre
"""
# Importer la librairie de pygame et initialiser
import pygame
from pygame import Color

class RaquetteAnimee :
    """ Un objet représente une raquette qui est animée dans une fenêtre Pygame

    La raquette est inscrite dans le rectangle englobant défini par r. Elle se déplace en diagonale selon la vitesse v.
        r : pygame.Rect       Le rectangle englobant
        v : [int,int]         Vitesse de déplacement selon les deux axes x et y
    """

    @staticmethod
    def set_fenetre(fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique

            fenetre : pygame.Surface
        """
        RaquetteAnimee.f = fenetre

    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse

    def dessiner(self):
        """ Dessiner une raquette.

        La raquette est inscrite dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(RaquetteAnimee.f, Color('red'), ((self.r.x,self.r.y),(self.r.width/3,self.r.height))) # L'embout gauche
        pygame.draw.ellipse(RaquetteAnimee.f, Color('red'), ((self.r.x+self.r.width*2/3,self.r.y),(self.r.width/3,self.r.height))) # L'embout droit
        pygame.draw.rect(RaquetteAnimee.f, Color('gray'), ((self.r.x+self.r.width/6,self.r.y),(self.r.width*2/3,self.r.height))) # Le corps
        pygame.draw.rect(RaquetteAnimee.f, Color('yellow'), ((self.r.x+self.r.width/4,self.r.y+self.r.height/4),(self.r.width/10,self.r.height/5))) # Le voyant gauche
        pygame.draw.rect(RaquetteAnimee.f, Color('yellow'), ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/4),(self.r.width/10,self.r.height/5))) # Le voyant droit
        pygame.draw.line(RaquetteAnimee.f, Color('black'), (self.r.x+self.r.width/4,self.r.y+self.r.height*3/4),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/4), 2) # La bande


    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > RaquetteAnimee.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > RaquetteAnimee.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y
        self.r.y = self.r.y+self.v[1]

class BalleAnimee :
    """ Un objet représente une raquette qui est animée dans une fenêtre Pygame

    La raquette est inscrite dans le rectangle englobant défini par r. Elle se déplace en diagonale selon vitesse.
        r : pygame.Rect       Le rectangle englobant
        vitesse : [int,int]   Vitesse de déplacement selon les deux axes x et y
        fenetre : pygame.Surface
        taille_fenetre : (int,int)
    """
    @staticmethod
    def set_fenetre(fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique

            fenetre : pygame.Surface
        """
        BalleAnimee.f = fenetre

    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse

    def dessiner(self):
        """ Dessiner une balle.

        La balle est inscrite dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BalleAnimee.f, Color('darkblue'), self.r) # La balle
        pygame.draw.ellipse(BalleAnimee.f, Color('white'), ((self.r.x+self.r.width/4,self.r.y+self.r.height/4),(self.r.width/6,self.r.height/6))) # Le reflet
        pygame.draw.arc(BalleAnimee.f, Color('white'), ((self.r.x+self.r.width/6,self.r.y+self.r.height/6),(self.r.width*2/3,self.r.height*2/3)), 1.7, 3.0, 2) # L'éclat
        pygame.draw.ellipse(BalleAnimee.f, Color('black'), self.r, 2) # Le contour

    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > BalleAnimee.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > BalleAnimee.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y
        self.r.y = self.r.y+self.v[1]

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre
RaquetteAnimee.set_fenetre(fenetre)
BalleAnimee.set_fenetre(fenetre)
pygame.display.set_caption("Exemple des raquettes et des balles en diagonale : fenetre variable d'objet") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de deux RaquetteAnimee et deux BalleAnimee
raquette1 = RaquetteAnimee(pygame.Rect((0,500),(80,20)),[5,0])
raquette2 = RaquetteAnimee(pygame.Rect((100,550),(120,30)),[3,0])
balle1 = BalleAnimee(pygame.Rect((200,150),(40,40)),[3,3])
balle2 = BalleAnimee(pygame.Rect((300,300),(50,50)),[5,10])

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        raquette1.deplacer()
        raquette2.deplacer()
        balle1.deplacer()
        balle2.deplacer()

        fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
        raquette1.dessiner()
        raquette2.dessiner()
        balle1.dessiner()
        balle2.dessiner()

        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images par seconde

pygame.quit() # Terminer pygame
