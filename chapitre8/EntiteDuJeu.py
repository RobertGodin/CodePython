"""
Module qui contient la hiérarchie des classes du jeu de casse-briques
"""
# Importer la librairie de pygame et initialiser
import pygame
from pygame import Color

class EntiteAnimeeAvecSon :
    """ Un objet représente une entité qui est animée dans une fenêtre Pygame.

    L'entité est inscrite dans le rectangle englobant r. Il se déplace en diagonale selon la vitesse v.
    Elle émet un son lorsqu'elle est frappée.
        r : pygame.Rect       Le rectangle englobant
        v : [int,int]         Vitesse de déplacement selon les deux axes x et y
        son : pygame.mixer.Sound
    """

    @classmethod
    def set_fenetre(cls, fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique

            fenetre : pygame.Surface
        """
        cls.f = fenetre

    def __init__(self,rectangle,vitesse,fichier_son):
        self.r = rectangle
        self.v = vitesse
        self.son = pygame.mixer.Sound(fichier_son)

    def prochaine_scene(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > EntiteAnimeeAvecSon.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x
        self.r.x += self.v[0]
        if self.r.y+self.v[1] > EntiteAnimeeAvecSon.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y
        self.r.y += self.v[1]

    def touche(self,autre):
        """ Vrai si les rectangles englobants se chevauchent """
        return self.r.colliderect(autre.r)

    def emettre_son(self):
        self.son.play()

class Raquette(EntiteAnimeeAvecSon):
    """ La raquette se déplace à l'horizontale selon les flèches """
    def prochaine_scene(self):
        touches = pygame.key.get_pressed() # État du clavier
        largeur = Raquette.f.get_width()
        if touches[pygame.K_LEFT] :
            self.r.x = max(0, self.r.x-self.v[0])
        if touches[pygame.K_RIGHT] :
            self.r.x = min(largeur-self.r.width, self.r.x+self.v[0])

    def dessiner(self):
        pygame.draw.rect(Raquette.f, Color('darkblue'), self.r)

class Balle(EntiteAnimeeAvecSon):
    """ La balle rebondit sur les bords, sauf sur le bord du bas """
    def prochaine_scene(self):
        if self.r.bottom+self.v[1] > Balle.f.get_height() :
            self.r.y += self.v[1] # La balle sort par le bas
        else :
            super().prochaine_scene()

    def perdue(self):
        """ Vrai si la balle est complètement sortie par le bas """
        return self.r.y > Balle.f.get_height()

    def rebondir_sur(self,autre):
        """ Rebondir sur l'entité autre. Si la zone de chevauchement
        est plus large que haute, la balle a frappé le dessus ou le
        dessous de l'entité ; sinon, elle a frappé un côté.
        """
        chevauchement = self.r.clip(autre.r)
        if chevauchement.width >= chevauchement.height :
            self.v[1] = -self.v[1]
        else :
            self.v[0] = -self.v[0]

    def rebondir_sur_raquette(self,raquette):
        """ L'angle du rebond dépend de l'endroit frappé """
        self.v[1] = -abs(self.v[1])
        self.v[0] = round((self.r.centerx-raquette.r.centerx)/8)

    def dessiner(self):
        pygame.draw.ellipse(Balle.f, Color('black'), self.r)

class Brique(EntiteAnimeeAvecSon):
    """ Une brique immobile qui est détruite au premier coup """
    def __init__(self,rectangle,fichier_son,couleur):
        super().__init__(rectangle,[0,0],fichier_son)
        self.couleur = couleur

    def frapper(self):
        """ Retourne vrai si la brique doit être détruite """
        self.emettre_son()
        return True

    def dessiner(self):
        pygame.draw.rect(Brique.f, self.couleur, self.r)

class BriqueResistante(Brique):
    """ Une brique qui doit être frappée plusieurs fois """
    def __init__(self,rectangle,fichier_son,couleur,nombre_coups):
        super().__init__(rectangle,fichier_son,couleur)
        self.coups_restants = nombre_coups

    def frapper(self):
        super().frapper()
        self.coups_restants -= 1
        return self.coups_restants == 0

    def dessiner(self):
        super().dessiner()
        # Bordure dont l'épaisseur indique les coups restants
        pygame.draw.rect(BriqueResistante.f, Color('black'), self.r,
                         self.coups_restants)
