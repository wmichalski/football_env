import pygame
import time
import random
from player import Player
from ball import Ball
import math

pygame.init()

display_width = 1280
display_height = 720

black = (0,0,0)
white = (255,255,255)
green = (119,221,119)

car_width = 73

gameDisplay = pygame.display.set_mode((display_width,display_height))
clock = pygame.time.Clock()

player = Player(int(display_width*0.75), int(display_height*0.5), 20)
ball = Ball(int(display_width*0.5), int(display_height*0.5), 12)

map_height = 600
map_width = 1000


def distance_between_two_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_player(player):
    pygame.draw.circle(gameDisplay, black, (int(player.x), int(player.y)), player.kick_radius, 1) 
    pygame.draw.circle(gameDisplay, black, (int(player.x), int(player.y)), player.radius)
    pygame.draw.circle(gameDisplay, white, (int(player.x), int(player.y)), int(player.radius*0.75))

def draw_ball(player):
    pygame.draw.circle(gameDisplay, black, (int(ball.x), int(ball.y)), ball.radius)
    pygame.draw.circle(gameDisplay, white, (int(ball.x), int(ball.y)), ball.radius-2)

def draw_map():
    # top
    pygame.draw.line(gameDisplay, black, (display_width/2 - map_width/2, display_height/2 - map_height/2), (display_width/2 - map_width/2, display_height/2 + map_height/2), 1)

def kick(player, ball):
    dist = distance_between_two_points(player.x, player.y, ball.x, ball.y)
    if dist <= player.kick_radius + ball.radius:
        vector = (player.x - ball.x, player.y - ball.y)
        ball.x_velocity -= vector[0]/dist * 25
        ball.y_velocity -= vector[1]/dist * 25

def check_collisions(player, ball):
    dist = distance_between_two_points(player.x, player.y, ball.x, ball.y)
    if dist <= player.radius + ball.radius:
        # TODO odbicie piłki od gracza, gdy gracz jest jakby ścianą
        # jeśli piłkowy wektor prędkości jest w stronę gracza, to odbijamy wektor (bo piłka inicjuje uderzenie)
        # w przeciwnym razie to gracz uderza w pilke
        ball.x_velocity
        ball.x_velocity += player.x_velocity*0.9
        ball.y_velocity += player.y_velocity*0.9
        player.x_velocity *= 0.9
        player.y_velocity *= 0.9

def check_borders(ball):
    if ball.x <= display_width/2 - map_width/2 + ball.radius:
        ball.x = display_width/2 - map_width/2 + ball.radius
        ball.x_velocity *= -1
        
def game_loop():
    gameExit = False

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            keys=pygame.key.get_pressed()

            x_change = 0
            y_change = 0

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                x_change += -0.7
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                x_change += 0.7
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                y_change += -0.7
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                y_change += 0.7
            if keys[pygame.K_SPACE]:
               kick(player, ball)
                

        player.x_velocity += x_change
        player.y_velocity += y_change

        player.apply_velocity()
        player.update_velocity()

        ball.apply_velocity()
        ball.update_velocity()

        check_collisions(player, ball)
        check_borders(ball)

        gameDisplay.fill(green)
        draw_map()
        draw_player(player)
        draw_ball(ball)

        pygame.display.update()
        clock.tick(60)



game_loop()
pygame.quit()
quit()