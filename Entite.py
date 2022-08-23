# -*- coding: utf-8 -*-
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

    @staticmethod
    def set_fenetre(fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique
        
            fenetre : pygame.Surface
        """
        EntiteAnimee.f = fenetre

    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse

    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > EntiteAnimee.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x    
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > EntiteAnimee.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]

class BotAnime(EntiteAnimee) :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame
        Sous-classe de EntiteAnimee
    """
        
    def dessiner(self):
        """ Dessiner un Bot. 
    
        Le Bot est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BotAnime.f, Color('green'), ((self.r.x,self.r.y),(self.r.width, self.r.height/2))) # Dessiner la tête
        pygame.draw.rect(BotAnime.f, Color('black'), ((self.r.x+self.r.width/4,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil gauche
        pygame.draw.rect(BotAnime.f, Color('black'), ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil droit
        pygame.draw.line(BotAnime.f, Color('black'), (self.r.x+self.r.width/4,self.r.y+self.r.height*3/8),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/8), 2) # La bouche
        pygame.draw.rect(BotAnime.f, Color('red'), ((self.r.x,self.r.y+self.r.height/2),(self.r.width,self.r.height/2))) # Le corps


class ItiAnime(EntiteAnimee) :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnimee
    """
        
    def dessiner(self):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """
        self.milieux = self.r.x + self.r.width/2;
        self.milieuy = self.r.y + self.r.height/2;

        pygame.draw.ellipse(ItiAnime.f, Color('pink'), ((self.r.x+self.r.width/3,self.r.y),(self.r.width/3,self.r.height/4))) # Dessiner la tête
        pygame.draw.arc(ItiAnime.f, Color('black'),((self.milieux-self.r.width/12,self.r.y+self.r.height/8),(self.r.width/6,self.r.height/14)),3.1416,0,2) # Le sourire
        pygame.draw.ellipse(ItiAnime.f, Color('black'), ((self.milieux-self.r.width/8,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil gauche
        pygame.draw.ellipse(ItiAnime.f, Color('black'), ((self.milieux+self.r.width/8-self.r.width/12,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil droit
        pygame.draw.line(ItiAnime.f, Color('black'), (self.milieux,self.r.y+self.r.height/4),(self.milieux,self.r.y+self.r.height*3/4), 2) # Le corps
        pygame.draw.line(ItiAnime.f, Color('black'), (self.r.x,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras gauche
        pygame.draw.line(ItiAnime.f, Color('black'), (self.r.x+self.r.width,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras droit
        pygame.draw.line(ItiAnime.f, Color('black'), (self.r.x,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe gauche
        pygame.draw.line(ItiAnime.f, Color('black'), (self.r.x+self.r.width,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe droite

