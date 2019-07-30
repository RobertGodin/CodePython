# -*- coding: utf-8 -*-
"""
Module qui contient la hiérarchie des classes EntiteAnimee : exemple de assert
"""
# Importer la librairie de pygame et initialiser 
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
    def __init__(self,x,y):
        self.x = x
        self.y = y

class TailleExcessiveErreur(Erreur):
    """ La taille de l'entité est excessive par rapport à la fenetre d'animation
    """    
    def __init__(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur

class EntiteAnimeeAvecSon :
    """ Un objet représente une entité qui est animée dans une fenêtre Pygame 
    
    L'entité est inscrite dans le rectangle englobant défini par les variables d'objet
    (x,y,largeur et hauteur). Elle se déplace en diagonale selon vitesse_x et vitesse_y.
    L'entité n'est pas affichée si visible est false. Elle émet un son lorsque supprimée.
        x : int
        y : int
        largeur : int
        hauteur : int
        vitesse_x : int
        vitesse_y : int
        son : pygame.mixer.Sound
    """
    @staticmethod
    def set_fenetre(f):
        """ Fixer les variables de classe fenetre, largeur_fenetre et hauteur_fenetre
        
            fenetre : pygame.Surface
            largeur_fenetre : int
            hauteur_fenetre : int
        """
        EntiteAnimeeAvecSon.fenetre = f
        EntiteAnimeeAvecSon.largeur_fenetre = f.get_width()
        EntiteAnimeeAvecSon.hauteur_fenetre = f.get_height()
  
    def __init__(self,x,y,largeur,hauteur,vitesse_x,vitesse_y,fichier_son):
        
        if x < 0 or y < 0 or x > EntiteAnimeeAvecSon.largeur_fenetre or y > EntiteAnimeeAvecSon.hauteur_fenetre :
            raise CoordonneesEntiteErreur(x,y)
        elif largeur > EntiteAnimeeAvecSon.largeur_fenetre or hauteur > EntiteAnimeeAvecSon.hauteur_fenetre :
            raise TailleExcessiveErreur(largeur, hauteur)
        else:
            self.x = x
            self.y = y
            self.largeur = largeur
            self.hauteur = hauteur
            self.vitesse_x = vitesse_x
            self.vitesse_y = vitesse_y
            self.son = pygame.mixer.Sound(fichier_son)
        
    def prochaine_scene(self):
        """ Déplacer l'entité en diagonale en rebondissant sur les bords de la fenetre"""
        if self.x+self.vitesse_x > EntiteAnimeeAvecSon.largeur_fenetre-self.largeur or self.x+self.vitesse_x < 0 :
            self.vitesse_x = +self.vitesse_x # Inverser la direction en x    
        self.x = self.x+self.vitesse_x
        if self.y+self.vitesse_y > EntiteAnimeeAvecSon.hauteur_fenetre-self.hauteur or self.y+self.vitesse_y < 0 :
            self.vitesse_y = -self.vitesse_y # Inverser la direction en y    
        self.y = self.y+self.vitesse_y
        assert self.x >= 0 and self.x <= EntiteAnimeeAvecSon.largeur_fenetre-self.largeur, "La position x dépasse du cadre du jeu"
        assert self.y >= 0 and self.y <= EntiteAnimeeAvecSon.hauteur_fenetre-self.hauteur, "La position y dépasse du cadre du jeu"
         
    def touche(self,x,y):
        return ((x >= self.x) and (x <= self.x + self.largeur) and (y >= self.y) and (y <= self.y + self.hauteur))
    
    def emettre_son(self):
        self.son.play()

class EntiteAvecEtat(EntiteAnimeeAvecSon):
    def __init__(self,x,y,largeur,hauteur,vitesse_x,vitesse_y,fichier_son,nombre_etats):
        super().__init__(x,y,largeur,hauteur,vitesse_x,vitesse_y,fichier_son)
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
    
        Le Bot est inscrit dans le rectangle englobant défini par les variables d'objet
        (x,y,largeur et hauteur) dans une fenetre de Pygame
        """

        pygame.draw.ellipse(BotAnime.fenetre, VERT, [self.x,self.y,self.largeur, self.hauteur/2]) # Dessiner la tête
        pygame.draw.rect(BotAnime.fenetre, NOIR, [self.x+self.largeur/4,self.y+self.hauteur/8,self.largeur/10,self.hauteur/20]) # L'oeil gauche
        pygame.draw.rect(BotAnime.fenetre, NOIR, [self.x+self.largeur*3/4-self.largeur/10,self.y+self.hauteur/8,self.largeur/10,self.hauteur/20]) # L'oeil droit
        pygame.draw.line(BotAnime.fenetre, NOIR, [self.x+self.largeur/4,self.y+self.hauteur*3/8],[self.x+self.largeur*3/4,self.y+self.hauteur*3/8], 2) # La bouche
        pygame.draw.rect(BotAnime.fenetre, ROUGE, [self.x,self.y+self.hauteur/2,self.largeur,self.hauteur/2]) # Le corps
        
    
class ItiAnimeVolant(EntiteAvecEtat) :
    """ Un objet représente un Iti qui est animé dans une fenêtre Pygame 
        Sous-classe de EntiteAnimee    
    """
    
    def dessiner(self):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par les variables d'objet
        (x,y,largeur et hauteur) dans une fenetre de Pygame
        """
        self.milieux = self.x + self.largeur/2;
        self.milieuy = self.y + self.hauteur/2;

        pygame.draw.ellipse(ItiAnimeVolant.fenetre, ROSE, [self.x+self.largeur/3,self.y,self.largeur/3,self.hauteur/4]) # Dessiner la tête
        pygame.draw.arc(ItiAnimeVolant.fenetre,NOIR,[self.milieux-self.largeur/12,self.y+self.hauteur/8,self.largeur/6,self.hauteur/14],3.1416,0,2) # Le sourire
        pygame.draw.ellipse(ItiAnimeVolant.fenetre, NOIR, [self.milieux-self.largeur/8,self.y+self.hauteur/12,self.largeur/12,self.hauteur/24]) # L'oeil gauche
        pygame.draw.ellipse(ItiAnimeVolant.fenetre, NOIR, [self.milieux+self.largeur/8-self.largeur/12,self.y+self.hauteur/12,self.largeur/12,self.hauteur/24]) # L'oeil droit
        pygame.draw.line(ItiAnimeVolant.fenetre, NOIR, [self.milieux,self.y+self.hauteur/4],[self.milieux,self.y+self.hauteur*3/4], 2) # Le corps
        pygame.draw.line(ItiAnimeVolant.fenetre, NOIR, [self.x,self.y+self.hauteur/4+(self.hauteur/4)*self.etat_courant],[self.milieux,self.milieuy], 2) # Bras gauche
        pygame.draw.line(ItiAnimeVolant.fenetre, NOIR, [self.x+self.largeur,self.y+self.hauteur/4+(self.hauteur/4)*self.etat_courant],[self.milieux,self.milieuy], 2) # Bras droit
        pygame.draw.line(ItiAnimeVolant.fenetre, NOIR, [self.x,self.y+self.hauteur],[self.milieux,self.y+self.hauteur*3/4], 2) # Jambe gauche
        pygame.draw.line(ItiAnimeVolant.fenetre, NOIR, [self.x+self.largeur,self.y+self.hauteur],[self.milieux,self.y+self.hauteur*3/4], 2) # Jambe droite

class EntiteAnimeeParImages(EntiteAvecEtat):
        def __init__(self,x,y,largeur,hauteur,vitesse_x,vitesse_y,fichier_son,nombre_etats,nom_dossier):
            super().__init__(x,y,largeur,hauteur,vitesse_x,vitesse_y,fichier_son,nombre_etats)
            self.image_animation = []
            for i in range(nombre_etats):
                self.image_animation.append(pygame.transform.scale(pygame.image.load(nom_dossier+"/"+nom_dossier+str(i+1)+".gif"),(largeur,hauteur)))
        def dessiner(self):
            self.fenetre.blit(self.image_animation[self.etat_courant],[self.x,self.y])
