import pygame
import time
import random
from player import Player
from ball import Ball
import math
import numpy as np

pygame.init()

display_width = 1280
display_height = 720

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (119, 221, 119)

gameDisplay = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()

fps = 60
gamespeed = 1

player = Player(int(display_width*0.75), int(display_height*0.5), 20, fps, gamespeed)
ball = Ball(int(display_width*0.5), int(display_height*0.5), 12, fps, gamespeed)

map_height = 600
map_width = 1000

goal_height = 200

def distance_between_two_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def draw_player(player):
    pygame.draw.circle(gameDisplay, black, (int(player.x),
                                            int(player.y)), player.kick_radius, 1)
    pygame.draw.circle(gameDisplay, black, (int(
        player.x), int(player.y)), player.radius+1)
    pygame.draw.circle(gameDisplay, white, (int(player.x),
                                            int(player.y)), int(player.radius*0.75))


def draw_ball(player):
    pygame.draw.circle(gameDisplay, black,
                       (int(ball.x), int(ball.y)), ball.radius+1)
    pygame.draw.circle(gameDisplay, white,
                       (int(ball.x), int(ball.y)), ball.radius-2)


def draw_map():
    # left
    pygame.draw.line(gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                          2 - map_height/2), (display_width/2 - map_width/2, display_height/2 + map_height/2), 1)
    pygame.draw.line(gameDisplay, green, (display_width/2 - map_width/2, display_height/2 -
                                          goal_height/2), (display_width/2 - map_width/2, display_height/2 + goal_height/2), 1)
    # right
    pygame.draw.line(gameDisplay, black, (display_width/2 + map_width/2, display_height /
                                          2 - map_height/2), (display_width/2 + map_width/2, display_height/2 + map_height/2), 1)
    # top
    pygame.draw.line(gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                          2 - map_height/2), (display_width/2 + map_width/2, display_height/2 - map_height/2), 1)
    # bottom
    pygame.draw.line(gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                          2 + map_height/2), (display_width/2 + map_width/2, display_height/2 + map_height/2), 1)


def kick(player, ball):
    dist = distance_between_two_points(player.x, player.y, ball.x, ball.y)
    if dist <= player.kick_radius + ball.radius:
        vector = (player.x - ball.x, player.y - ball.y)
        ball.x_velocity -= vector[0]/dist * 20
        ball.y_velocity -= vector[1]/dist * 20

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def is_ball_close_to_walls():
    # TODO not just walls but all collisions in general, especially other players

    if ball.x <= display_width/2 - map_width/2 + 2*ball.radius:
        return True

    if ball.x >= display_width/2 + map_width/2 - 2*ball.radius:
        return True

    if ball.y >= display_height/2 + map_height/2 - 2*ball.radius:
        return True

    if ball.y <= display_height/2 - map_height/2 + 2*ball.radius:
        return True

    return False


def check_collisions(player, ball):
    dist = distance_between_two_points(player.x, player.y, ball.x, ball.y)
    if dist < player.radius + ball.radius:

        # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        ray = np.array([ball.x_velocity, ball.y_velocity])
        v = np.array([player.x - ball.x, player.y - ball.y])
        nrm = v / np.sqrt(np.sum(v**2))

        # fix position once the ball is "inside the player"
        # TODO it is based now on the position of balls. what if the ball is so fast that it appears in the other half of the player?

        # if the ball is close to the walls, it cannot move further - so we take a different approach then, so that the player cannot move onto ball
        if is_ball_close_to_walls():
            middle_x = (ball.x + player.x)/2
            middle_y = (ball.y + player.y)/2

            ball.x = middle_x-0.5*v[0]*(player.radius+ball.radius)/dist
            ball.y = middle_y-0.5*v[1]*(player.radius+ball.radius)/dist

            player.x = middle_x+0.5*v[0]*(player.radius+ball.radius)/dist
            player.y = middle_y+0.5*v[1]*(player.radius+ball.radius)/dist
        else:
            ball.x = player.x-v[0]*(player.radius+ball.radius)/dist
            ball.y = player.y-v[1]*(player.radius+ball.radius)/dist

        # if the ball hits the player, then it should bounce
        if not(ball.x_velocity == 0 and ball.y_velocity == 0):
            if angle_between(ray, nrm) < 1.57:  
                reflection = ray - (2 * (np.dot(ray, nrm)) * nrm)
                ball.x_velocity = reflection[0]
                ball.y_velocity = reflection[1]

        player_velocity = np.sqrt(player.x_velocity**2 + player.y_velocity**2)
        nrm *= player_velocity

        # TODO some stickyness parameter?
        # we want the balls' velocity to be decided not only by player's velocty, but also by theirs position
        ball.x_velocity += (player.x_velocity*7-nrm[0])*0.6*0.125
        ball.y_velocity += (player.y_velocity*7-nrm[1])*0.6*0.125
        player.x_velocity *= 0.9
        player.y_velocity *= 0.9


def check_borders_ball(ball):
    # left border
    if ball.x <= display_width/2 - map_width/2 + ball.radius:
        if ball.y > display_height/2 - goal_height/2 and ball.y < display_height/2 + goal_height/2:
            # if tha ball is fully inside of the goal:
            if ball.x <= display_width/2 - map_width/2 - ball.radius:
                ball.in_goal = True
        else:
            ball.x = display_width/2 - map_width/2 + ball.radius
            ball.x_velocity *= -1

    # right border
    if ball.x >= display_width/2 + map_width/2 - ball.radius:
        ball.x = display_width/2 + map_width/2 - ball.radius
        ball.x_velocity *= -1

    # bottom border
    if ball.y >= display_height/2 + map_height/2 - ball.radius:
        ball.y = display_height/2 + map_height/2 - ball.radius
        ball.y_velocity *= -1

    # top border
    if ball.y <= display_height/2 - map_height/2 + ball.radius:
        ball.y = display_height/2 - map_height/2 + ball.radius
        ball.y_velocity *= -1

def check_borders_player(player):
    # left border
    if player.x <= 0:
        player.x = 0
        player.x_velocity = 0

    # right border
    if player.x >= display_width:
        player.x = display_width
        player.x_velocity = 0

    # bottom border
    if player.y >= display_height:
        player.y = display_height
        player.y_velocity = 0

    # top border
    if player.y <= 0:
        player.y = 0
        player.y_velocity = 0


def game_loop():
    gameExit = False

    while not gameExit:
        if ball.in_goal:
            ball.reset(display_width/2, display_height/2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            keys = pygame.key.get_pressed()

            x_change = 0
            y_change = 0

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                x_change += -0.7 * 60 / fps* gamespeed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                x_change += 0.7 * 60 / fps * gamespeed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                y_change += -0.7 * 60 / fps * gamespeed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                y_change += 0.7 * 60 / fps * gamespeed
            if keys[pygame.K_SPACE]:
                kick(player, ball)

        player.x_velocity += x_change
        player.y_velocity += y_change

        player.update_velocity()
        player.apply_velocity()

        ball.update_velocity()
        ball.apply_velocity()

        check_collisions(player, ball)
        check_borders_ball(ball)
        check_borders_player(player)

        gameDisplay.fill(green)
        draw_map()
        draw_player(player)
        draw_ball(ball)

        pygame.display.update()
        print(ball.x_velocity, ball.y_velocity)
        clock.tick(fps)


game_loop()
pygame.quit()
quit()
