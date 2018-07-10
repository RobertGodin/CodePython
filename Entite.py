# -*- coding: utf-8 -*-
"""
Module qui contient la hiérarchie des classes EntiteAnime
"""
# Importer la librairie de pygame et initialiser 
import pygame

ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)    
ROSE = (255,100,100) 

class EntiteAnime :
    """ Un objet représente une entité qui est animée dans une fenêtre Pygame 
    
    L'entité est inscrite dans le rectangle englobant défini par les variables d'objet
    (x,y,largeur et hauteur). Elle se déplace en diagonale selon vitesse_x et vitesse_y. 
        x : int
        y : int
        largeur : int
        hauteur : int
        vitesse_x : int
        vitesse_y : int
    """
    @staticmethod
    def set_fenetre(f):
        """ Fixer les variables de classe fenetre, largeur_fenetre et hauteur_fenetre
        
            fenetre : pygame.Surface
            largeur_fenetre : int
            hauteur_fenetre : int
        """
        EntiteAnime.fenetre = f
        EntiteAnime.largeur_fenetre = f.get_width()
        EntiteAnime.hauteur_fenetre = f.get_height()
  
    def __init__(self,x,y,largeur,hauteur,vitesse_x,vitesse_y):
        self.x = x
        self.y = y
        self.largeur = largeur
        self.hauteur = hauteur
        self.vitesse_x = vitesse_x
        self.vitesse_y = vitesse_y
        
    def deplacer(self):
        """ Déplacer l'entité en diagonale en rebondissant sur les bords de la fenetre"""
        if self.x+self.vitesse_x > EntiteAnime.largeur_fenetre-self.largeur or self.x+self.vitesse_x < 0 :
            self.vitesse_x = -self.vitesse_x # Inverser la direction en x    
        self.x = self.x+self.vitesse_x
        if self.y+self.vitesse_y > EntiteAnime.hauteur_fenetre-self.hauteur or self.y+self.vitesse_y < 0 :
            self.vitesse_y = -self.vitesse_y # Inverser la direction en y    
        self.y = self.y+self.vitesse_y

class BotAnime(EntiteAnime) :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnime
    """

    def dessiner(self):
        """ Dessiner un Bot. 
    
        Le Bot est inscrit dans le rectangle englobant défini par les variables d'objet
        (x,y,largeur et hauteur) dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BotAnime.fenetre, VERT, [self.x,self.y,self.largeur, self.hauteur/2]) # Dessiner la tête
        pygame.draw.rect(BotAnime.fenetre, NOIR, [self.x+self.largeur/4,self.y+self.hauteur/8,self.largeur/10,self.hauteur/20]) # L'oeil gauche
        pygame.draw.rect(BotAnime.fenetre, NOIR, [self.x+self.largeur*3/4-self.largeur/10,self.y+self.hauteur/8,self.largeur/10,self.hauteur/20]) # L'oeil droit
        pygame.draw.line(BotAnime.fenetre, NOIR, [self.x+self.largeur/4,self.y+self.hauteur*3/8],[self.x+self.largeur*3/4,self.y+self.hauteur*3/8], 2) # La bouche
        pygame.draw.rect(BotAnime.fenetre, ROUGE, [self.x,self.y+self.hauteur/2,self.largeur,self.hauteur/2]) # Le corps
        
    
class ItiAnime(EntiteAnime) :
    """ Un objet représente un Iti qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnime    
    """
    
    def dessiner(self):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par les variables d'objet
        (x,y,largeur et hauteur) dans une fenetre de Pygame
        """
        self.milieux = self.x + self.largeur/2;
        self.milieuy = self.y + self.hauteur/2;

        pygame.draw.ellipse(ItiAnime.fenetre, ROSE, [self.x+self.largeur/3,self.y,self.largeur/3,self.hauteur/4]) # Dessiner la tête
        pygame.draw.arc(ItiAnime.fenetre,NOIR,[self.milieux-self.largeur/12,self.y+self.hauteur/8,self.largeur/6,self.hauteur/14],3.1416,0,2) # Le sourire
        pygame.draw.ellipse(ItiAnime.fenetre, NOIR, [self.milieux-self.largeur/8,self.y+self.hauteur/12,self.largeur/12,self.hauteur/24]) # L'oeil gauche
        pygame.draw.ellipse(ItiAnime.fenetre, NOIR, [self.milieux+self.largeur/8-self.largeur/12,self.y+self.hauteur/12,self.largeur/12,self.hauteur/24]) # L'oeil droit
        pygame.draw.line(ItiAnime.fenetre, NOIR, [self.milieux,self.y+self.hauteur/4],[self.milieux,self.y+self.hauteur*3/4], 2) # Le corps
        pygame.draw.line(ItiAnime.fenetre, NOIR, [self.x,self.y+self.hauteur/4],[self.milieux,self.milieuy], 2) # Bras gauche
        pygame.draw.line(ItiAnime.fenetre, NOIR, [self.x+self.largeur,self.y+self.hauteur/4],[self.milieux,self.milieuy], 2) # Bras droit
        pygame.draw.line(ItiAnime.fenetre, NOIR, [self.x,self.y+self.hauteur],[self.milieux,self.y+self.hauteur*3/4], 2) # Jambe gauche
        pygame.draw.line(ItiAnime.fenetre, NOIR, [self.x+self.largeur,self.y+self.hauteur],[self.milieux,self.y+self.hauteur*3/4], 2) # Jambe droite
        
