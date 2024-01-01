# -*- coding: utf-8 -*-
"Petit test avec Pygame"
# Pour le nom des couleurs : <https://www.pygame.org/docs/ref/color_list.html> 
import sys
import pygame 
from pygame import Color, display 
from pygame.locals import QUIT 
FENETRE_DIMENSION = 400, 600
def creer_decor():
    fenetre = creer_fenetre("Test A", Color('grey')) 
    dessiner_bot(fenetre, cadre(facteur=1, delta_x=0, delta_y=0))
    dessiner_bot(fenetre, cadre(facteur=0.5, delta_x=-20, delta_y=0))
def creer_fenetre(titre, couleur):
    fenetre = display.set_mode(FENETRE_DIMENSION) 
    fenetre.fill(couleur) 
    display.set_caption(titre) 
    return fenetre
def dessiner_bot(fenetre, cadrer):
    pygame.draw.ellipse(fenetre, Color('green'), cadrer(100, 100, 200, 200)) # tête 
    pygame.draw.rect(fenetre, Color('black'), cadrer(150, 150, 20, 20)) # oeil gauche 
    pygame.draw.rect(fenetre, Color('black'), cadrer(230, 150, 20, 20)) # oeil droit 
    pygame.draw.line(fenetre, Color('black'), cadrer(150, 250), cadrer(250, 250), 2) # bouche
    pygame.draw.rect(fenetre, Color('red'), cadrer(100, 300, 200, 200)) # corps
def cadre(facteur, delta_x, delta_y):
    def cadrer(x, y, a=None, b=None):
        position = facteur *x + delta_x, facteur* y + delta_y 
        return position if a is None else position + (facteur *a, facteur* b)
    return cadrer
if __name__ == '__main__':
    pygame.init() 
    creer_decor() 
    while True:
        display.update() 
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit() 
                sys.exit()
