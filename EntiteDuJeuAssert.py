# -*- coding: utf-8 -*-
"""
Module qui contient la hiérarchie des classes EntiteAnimee : exemple de assert
"""
# -*- coding: utf-8 -*-
"""
Module qui contient la hiérarchie des classes EntiteAnimee : exemples d'exceptions
"""
import pygame

ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)    
ROSE = (255,100,100) 

class Erreur(Exception):
    """ Classe de base pour les exceptions de ce module
    """
    pass

class CoordonneesEntiteErreur(Erreur):
    """ Les coordonnées dépassent la fenetre d'animation
    """    
    def __init__(self,position):
        self.position = position

class TailleExcessiveErreur(Erreur):
    """ La taille de l'entité est excessive par rapport à la fenetre d'animation
    """    
    def __init__(self, taille):
        self.taille=taille

class EntiteAnimeeAvecSon :
    """ Un objet représente une entité qui est animée dans une fenêtre Pygame. 
    
    L'entité est inscrite dans le rectangle englobant r. Il se déplace en diagonale selon la vitesse v.
    Elle émet un son lorsque supprimée.
        r : pygame.Rect       Le rectangle englobant 
        v : [int,int]         Vitesse de déplacement selon les deux axes x et y
        son : pygame.mixer.Sound
    """

    @staticmethod
    def set_fenetre(fenetre):
        """ Fixer la variable de classe f qui représente la fenetre graphique
        
            fenetre : pygame.Surface
        """
        EntiteAnimeeAvecSon.f = fenetre

    def __init__(self,rectangle,vitesse,fichier_son):
        if rectangle.x < 0 or rectangle.y < 0 or rectangle.x > EntiteAnimeeAvecSon.f.get_width() or rectangle.y > EntiteAnimeeAvecSon.f.get_height() :
            raise CoordonneesEntiteErreur((rectangle.x,rectangle.y))
        elif rectangle.width > EntiteAnimeeAvecSon.f.get_width() or rectangle.height > EntiteAnimeeAvecSon.f.get_height() :
            raise TailleExcessiveErreur(rectangle.size)
        else:
            self.r = rectangle
            self.v = vitesse
            self.son = pygame.mixer.Sound(fichier_son)
        
    def prochaine_scene(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > EntiteAnimeeAvecSon.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = +self.v[0] # Inverser la direction en x (bug introduit pour l'exemple !)   
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > EntiteAnimeeAvecSon.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]
        assert self.r.x >= 0 and self.r.x <= EntiteAnimeeAvecSon.f.get_width()-self.r.width, "La position x dépasse du cadre du jeu"
        assert self.r.y >= 0 and self.r.y <= EntiteAnimeeAvecSon.f.get_height()-self.r.height, "La position y dépasse du cadre du jeu"
        
    def touche(self,x,y):
        return ((x >= self.r.x) and (x <= self.r.x + self.r.width) and (y >= self.r.y) and (y <= self.r.y + self.r.height))
    
    def emettre_son(self):
        self.son.play()

class EntiteAvecEtat(EntiteAnimeeAvecSon):
    def __init__(self,rectangle,vitesse,fichier_son,nombre_etats):
        super().__init__(rectangle,vitesse,fichier_son)
        self.nombre_etats = nombre_etats
        self.etat_courant = 0
    
    def prochaine_scene(self):
        self.etat_courant=(self.etat_courant+1)%self.nombre_etats
        super().prochaine_scene()

        
        

class BotAnime(EntiteAnimeeAvecSon) :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnimee
    """
        
    def dessiner(self):
        """ Dessiner un Bot. 
    
        Le Bot est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BotAnime.f, VERT, ((self.r.x,self.r.y),(self.r.width, self.r.height/2))) # Dessiner la tête
        pygame.draw.rect(BotAnime.f, NOIR, ((self.r.x+self.r.width/4,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil gauche
        pygame.draw.rect(BotAnime.f, NOIR, ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil droit
        pygame.draw.line(BotAnime.f, NOIR, (self.r.x+self.r.width/4,self.r.y+self.r.height*3/8),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/8), 2) # La bouche
        pygame.draw.rect(BotAnime.f, ROUGE, ((self.r.x,self.r.y+self.r.height/2),(self.r.width,self.r.height/2))) # Le corps
        
    
class ItiAnimeVolant(EntiteAvecEtat) :
    """ Un objet représente un Iti qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnimee    
    """
    
    def dessiner(self):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        L'etat courant détermine la hauteur des bras.
        """
        self.milieux = self.r.x + self.r.width/2;
        self.milieuy = self.r.y + self.r.height/2;

        pygame.draw.ellipse(ItiAnimeVolant.f, ROSE, ((self.r.x+self.r.width/3,self.r.y),(self.r.width/3,self.r.height/4))) # Dessiner la tête
        pygame.draw.arc(ItiAnimeVolant.f,NOIR,((self.milieux-self.r.width/12,self.r.y+self.r.height/8),(self.r.width/6,self.r.height/14)),3.1416,0,2) # Le sourire
        pygame.draw.ellipse(ItiAnimeVolant.f, NOIR, ((self.milieux-self.r.width/8,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil gauche
        pygame.draw.ellipse(ItiAnimeVolant.f, NOIR, ((self.milieux+self.r.width/8-self.r.width/12,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil droit
        pygame.draw.line(ItiAnimeVolant.f, NOIR, (self.milieux,self.r.y+self.r.height/4),(self.milieux,self.r.y+self.r.height*3/4), 2) # Le corps
        pygame.draw.line(ItiAnimeVolant.f, NOIR, (self.r.x,self.r.y+self.r.height/4+(self.r.height/4)*self.etat_courant),(self.milieux,self.milieuy), 2) # Bras gauche
        pygame.draw.line(ItiAnimeVolant.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height/4+(self.r.height/4)*self.etat_courant),(self.milieux,self.milieuy), 2) # Bras droit
        pygame.draw.line(ItiAnimeVolant.f, NOIR, (self.r.x,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe gauche
        pygame.draw.line(ItiAnimeVolant.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe droite


class EntiteAnimeeParImages(EntiteAvecEtat):
        def __init__(self,rectangle,vitesse,fichier_son,nombre_etats,nom_dossier):
            super().__init__(rectangle,vitesse,fichier_son,nombre_etats)
            self.image_animation = []
            for i in range(nombre_etats):
                self.image_animation.append(pygame.transform.scale(pygame.image.load(nom_dossier+"/"+nom_dossier+str(i+1)+".gif"),(self.r.width,self.r.height)))
        def dessiner(self):
            EntiteAnimeeParImages.f.blit(self.image_animation[self.etat_courant],[self.r.x,self.r.y])
