# -*- coding: utf-8 -*-
"""
Exemple d'animation d'entités : exemple de super-classe EntiteAnimee
"""
import pygame
ROUGE = (255,0,0)
NOIR = (0,0,0)
VERT = (0,255,0)    
ROSE = (255,100,100) 

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

        pygame.draw.ellipse(BotAnime.f, VERT, ((self.r.x,self.r.y),(self.r.width, self.r.height/2))) # Dessiner la tête
        pygame.draw.rect(BotAnime.f, NOIR, ((self.r.x+self.r.width/4,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil gauche
        pygame.draw.rect(BotAnime.f, NOIR, ((self.r.x+self.r.width*3/4-self.r.width/10,self.r.y+self.r.height/8),(self.r.width/10,self.r.height/20))) # L'oeil droit
        pygame.draw.line(BotAnime.f, NOIR, (self.r.x+self.r.width/4,self.r.y+self.r.height*3/8),(self.r.x+self.r.width*3/4,self.r.y+self.r.height*3/8), 2) # La bouche
        pygame.draw.rect(BotAnime.f, ROUGE, ((self.r.x,self.r.y+self.r.height/2),(self.r.width,self.r.height/2))) # Le corps


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

        pygame.draw.ellipse(ItiAnime.f, ROSE, ((self.r.x+self.r.width/3,self.r.y),(self.r.width/3,self.r.height/4))) # Dessiner la tête
        pygame.draw.arc(ItiAnime.f,NOIR,((self.milieux-self.r.width/12,self.r.y+self.r.height/8),(self.r.width/6,self.r.height/14)),3.1416,0,2) # Le sourire
        pygame.draw.ellipse(ItiAnime.f, NOIR, ((self.milieux-self.r.width/8,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil gauche
        pygame.draw.ellipse(ItiAnime.f, NOIR, ((self.milieux+self.r.width/8-self.r.width/12,self.r.y+self.r.height/12),(self.r.width/12,self.r.height/24))) # L'oeil droit
        pygame.draw.line(ItiAnime.f, NOIR, (self.milieux,self.r.y+self.r.height/4),(self.milieux,self.r.y+self.r.height*3/4), 2) # Le corps
        pygame.draw.line(ItiAnime.f, NOIR, (self.r.x,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras gauche
        pygame.draw.line(ItiAnime.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height/4),(self.milieux,self.milieuy), 2) # Bras droit
        pygame.draw.line(ItiAnime.f, NOIR, (self.r.x,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe gauche
        pygame.draw.line(ItiAnime.f, NOIR, (self.r.x+self.r.width,self.r.y+self.r.height),(self.milieux,self.r.y+self.r.height*3/4), 2) # Jambe droite

pygame.init() # Initialiser les modules de Pygame
LARGEUR_FENETRE = 400
HAUTEUR_FENETRE = 600
fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE)) # Ouvrir la fenêtre 
EntiteAnimee.set_fenetre(fenetre)
pygame.display.set_caption("Exemple des Bots et Itis animés en diagonale avec super-classe EntiteAnimee") # Définir le titre dans le haut de la fenêtre

BLANC = (255,255,255)
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