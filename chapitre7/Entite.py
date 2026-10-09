"""
Module qui contient la hiérarchie des classes EntiteAnimee
"""

import pygame
from pygame import Color

class EntiteAnimee :
    """ Un objet représente une entité qui est animée dans une fenêtre Pygame

    L'entité est inscrite dans le rectangle englobant défini par r. Il se déplace en diagonale selon la vitesse v.
        r : pygame.Rect       Le rectangle englobant
        v : [int,int]         Vitesse de déplacement selon les deux axes x et y
    """

    @classmethod
    def set_fenetre(cls, fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique

            fenetre : pygame.Surface
        """
        cls.f = fenetre

    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse

    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > EntiteAnimee.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x
        self.r.x += self.v[0]
        if self.r.y+self.v[1] > EntiteAnimee.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y
        self.r.y += self.v[1]

class RaquetteAnimee(EntiteAnimee) :
    """ Un objet représente une raquette qui est animée dans une fenêtre Pygame
        Sous-classe de EntiteAnimee
    """

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


class BalleAnimee(EntiteAnimee) :
    """ Un objet représente une balle qui est animée dans une fenêtre Pygame
        Sous-classe de EntiteAnimee
    """

    def dessiner(self):
        """ Dessiner une balle.

        La balle est inscrite dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BalleAnimee.f, Color('darkblue'), self.r) # La balle
        pygame.draw.ellipse(BalleAnimee.f, Color('white'), ((self.r.x+self.r.width/4,self.r.y+self.r.height/4),(self.r.width/6,self.r.height/6))) # Le reflet
        pygame.draw.arc(BalleAnimee.f, Color('white'), ((self.r.x+self.r.width/6,self.r.y+self.r.height/6),(self.r.width*2/3,self.r.height*2/3)), 1.7, 3.0, 2) # L'éclat
        pygame.draw.ellipse(BalleAnimee.f, Color('black'), self.r, 2) # Le contour
