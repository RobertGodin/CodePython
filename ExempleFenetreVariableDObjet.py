# -*- coding: utf-8 -*-
"""
Exemple d'animation d'entités : création des classes BotAnime et ItiAnime
avec variable d'objet pour la fenetre
"""
# Importer la librairie de pygame et initialiser 
import pygame

ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)    
ROSE = (255,100,100) 

class BotAnime :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 
    
    Le Bot est inscrit dans le rectangle englobant défini par r. Il se déplace en diagonale selon la vitesse v. 
        r : pygame.Rect       Le rectangle englobant 
        v : [int,int]   Vitesse de déplacement selon les deux axes x et y
        f : pygame.Surface
    """
        
    def __init__(self,rectangle,vitesse,fenetre):
        self.r = rectangle
        self.v = vitesse
        self.f = fenetre
        
    def dessiner(self):
        """ Dessiner un Bot. 
    
        Le Bot est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """

        pygame.draw.ellipse(self.f, VERT, ((self.r.x,self.r.y),(self.r.width, self.r.height/2))) # Dessiner la tête
        pygame.draw.rect(self.f, NOIR, ((self.r.x+self.r.width/4,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil gauche
        pygame.draw.rect(self.f, NOIR, ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil droit
        pygame.draw.line(self.f, NOIR, (self.r.x+self.r.width/4,self.r.y+self.r.height*3/8),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/8), 2) # La bouche
        pygame.draw.rect(self.f, ROUGE, ((self.r.x,self.r.y+self.r.height/2),(self.r.width,self.r.height/2))) # Le corps


    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > self.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x    
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > self.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]

class ItiAnime :
    """ Un objet représente un Bot qui est animé dans une fenêtre Pygame 

    Le Iti est inscrit dans le rectangle englobant défini par r. Il se déplace en diagonale selon la vitesse v. 
        r : pygame.Rect       Le rectangle englobant 
        v : [int,int]   Vitesse de déplacement selon les deux axes x et y
        f : pygame.Surface
    """
        
    def __init__(self,rectangle,vitesse,fenetre):
        self.r = rectangle
        self.v = vitesse
        self.f = fenetre
        
    def dessiner(self):
        """ Dessiner un Iti. 
    
        Le Iti est inscrit dans le rectangle englobant défini par la variable d'objet r dans une fenetre de Pygame
        """
        self.milieux = self.r.x + self.r.width/2;
        self.milieuy = self.r.y + self.r.height/2;

        pygame.draw.ellipse(self.f, ROSE, ((self.r.x+self.r.width/3,self.r.y),(self.r.width/3,self.r.height/4))) # Dessiner la tête
        pygame.draw.arc(self.f,NOIR,((self.milieux-self.r.width/12,self.r.y+self.r.height/8),(self.r.width/6,self.r.height/14)),3.1416,0,2) # Le sourire
        pygame.draw.ellipse(self.f, NOIR, ((self.milieux-self.r.width/8,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil gauche
        pygame.draw.ellipse(self.f, NOIR, ((self.milieux+self.r.width/8-self.r.width/12,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil droit
        pygame.draw.line(self.f, NOIR, (self.milieux,self.r.y+self.r.height/4),(self.milieux,self.r.y+self.r.height*3/4), 2) # Le corps
        pygame.draw.line(self.f, NOIR, (self.r.x,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras gauche
        pygame.draw.line(self.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras droit
        pygame.draw.line(self.f, NOIR, (self.r.x,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe gauche
        pygame.draw.line(self.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe droite

    def deplacer(self):
        """ Déplacer selon self.v en diagonale en rebondissant sur les bords de la fenetre"""
        if self.r.x+self.v[0] > self.f.get_width()-self.r.width or self.r.x+self.v[0] < 0 :
            self.v[0] = -self.v[0] # Inverser la direction en x    
        self.r.x = self.r.x+self.v[0]
        if self.r.y+self.v[1] > self.f.get_height()-self.r.height or self.r.y+self.v[1] < 0 :
            self.v[1] = -self.v[1] # Inverser la direction en y    
        self.r.y = self.r.y+self.v[1]

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
pygame.display.set_caption("Exemple des Bots et Itis en diagonale : fenetre variable d'objet") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
horloge = pygame.time.Clock() # Pour contrôler la fréquence des scènes

# Création de deux BotAnime et deux ItiAnime
bot1 = BotAnime(pygame.Rect((0,0),(20,40)),[5,10],fenetre)
bot2 = BotAnime(pygame.Rect((100,200),(30,60)),[0,2],fenetre)
iti1 = ItiAnime(pygame.Rect((200,150),(40,80)),[3,3],fenetre)
iti2 = ItiAnime(pygame.Rect((300,300),(50,100)),[5,10],fenetre)

# Boucle d'animation
fin = False
while not fin :
    event = pygame.event.poll() # Chercher le prochain évènement à traiter        
    if event.type == pygame.QUIT:  # Utilisateur a cliqué sur la fermeture de fenêtre ?
        fin = True  # Fin de la boucle du jeu
    else :
        bot1.deplacer()
        bot2.deplacer()
        iti1.deplacer()
        iti2.deplacer()

        fenetre.fill(BLANC) # Dessiner le fond de la surface de dessin
        bot1.dessiner()
        bot2.dessiner()
        iti1.dessiner()
        iti2.dessiner()
        
        pygame.display.flip() # Mettre à jour la fenêtre graphique

        horloge.tick(60) # Pour animer avec 60 images pas seconde
 
pygame.quit() # Terminer pygame