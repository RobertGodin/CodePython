# -*- coding: utf-8 -*-
"""
Exemple d'animation d'entités : création des classes BotAnime et ItiAnime
"""
# Importer la librairie de pygame et initialiser 
import pygame
from pygame import Color
class BotAnime :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 
    
    Le Bot est inscrit dans le rectangle englobant défini par r. Il se déplace en diagonale selon vitesse. 
        r : pygame.Rect       Le rectangle englobant 
        v : [int,int]   Vitesse de déplacement selon les deux axes x et y
    """
        
    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse
        
    def dessiner(self,fenetre):
        """ Dessiner un Bot. 
    
        Le Bot est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(fenetre, Color('green'), ((self.r.x,self.r.y),(self.r.width, self.r.height/2))) # Dessiner la tête
        pygame.draw.rect(fenetre, Color('black'), ((self.r.x+self.r.width/4,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil gauche
        pygame.draw.rect(fenetre, Color('black'), ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil droit
        pygame.draw.line(fenetre, Color('black'), (self.r.x+self.r.width/4,self.r.y+self.r.height*3/8),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/8), 2) # La bouche
        pygame.draw.rect(fenetre, Color('red'), ((self.r.x,self.r.y+self.r.height/2),(self.r.width,self.r.height/2))) # Le corps


    def deplacer(self,largeur_fenetre,hauteur_fenetre):
        """ Déplacer le Bot en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > largeur_fenetre-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x    
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > hauteur_fenetre-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]
    
class ItiAnime :
    """ Un objet représente un Iti qui est animé dans une fenêtre Pygame 
    
    Le Iti est inscrit dans le rectangle englobant défini par rectangle. Il se déplace en diagonale selon vitesse. 
        r : pygame.Rect       Le rectangle englobant 
        v : [int,int]   Vitesse de déplacement selon les deux axes x et y
    """
        
    def __init__(self,rectangle,vitesse):
        self.r = rectangle
        self.v = vitesse
        
    def dessiner(self,fenetre):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """
        self.milieux = self.r.x + self.r.width/2;
        self.milieuy = self.r.y + self.r.height/2;

        pygame.draw.ellipse(fenetre, Color('pink'), ((self.r.x+self.r.width/3,self.r.y),(self.r.width/3,self.r.height/4))) # Dessiner la tête
        pygame.draw.arc(fenetre, Color('black'),((self.milieux-self.r.width/12,self.r.y+self.r.height/8),(self.r.width/6,self.r.height/14)),3.1416,0,2) # Le sourire
        pygame.draw.ellipse(fenetre, Color('black'), ((self.milieux-self.r.width/8,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil gauche
        pygame.draw.ellipse(fenetre, Color('black'), ((self.milieux+self.r.width/8-self.r.width/12,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil droit
        pygame.draw.line(fenetre, Color('black'), (self.milieux,self.r.y+self.r.height/4),(self.milieux,self.r.y+self.r.height*3/4), 2) # Le corps
        pygame.draw.line(fenetre, Color('black'), (self.r.x,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras gauche
        pygame.draw.line(fenetre, Color('black'), (self.r.x+self.r.width,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras droit
        pygame.draw.line(fenetre, Color('black'), (self.r.x,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe gauche
        pygame.draw.line(fenetre, Color('black'), (self.r.x+self.r.width,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe droite


    def deplacer(self,largeur_fenetre,hauteur_fenetre):
        """ Déplacer le Iti en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > largeur_fenetre-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x    
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > hauteur_fenetre-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Bot et Iti en diagonale avec classe") # Définir le titre dans le haut de la fenêtre

horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de deux BotAnime et deux ItiAnime
bot1 = BotAnime(pygame.Rect((0,0),(20,40)),[5,10])
bot2 = BotAnime(pygame.Rect((100,200),(30,60)),[0,2])
iti1 = ItiAnime(pygame.Rect((200,150),(40,80)),[3,3])
iti2 = ItiAnime(pygame.Rect((300,300),(50,100)),[5,10])

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        bot1.deplacer(LARGEUR_FENETRE,HAUTEUR_FENETRE)
        bot2.deplacer(LARGEUR_FENETRE,HAUTEUR_FENETRE)
        iti1.deplacer(LARGEUR_FENETRE,HAUTEUR_FENETRE)
        iti2.deplacer(LARGEUR_FENETRE,HAUTEUR_FENETRE)

        fenetre.fill(Color('white')) # Dessiner le fond de la surface de dessin
        bot1.dessiner(fenetre)
        bot2.dessiner(fenetre)
        iti1.dessiner(fenetre)
        iti2.dessiner(fenetre)
        
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame