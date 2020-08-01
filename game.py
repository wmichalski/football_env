import pygame
import time
import random
from player import Player
from ball import Ball
import math
import numpy as np
from statistics import mean

fps = 6000
gamespeed = 1
max_frames = 300

map_height = 200  
map_width = 333  
display_width = 400  
display_height = 300  
goal_height = 175

# map_height = 600
# map_width = 1000
# display_width = 1280
# display_height = 720
# goal_height = 200

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (119, 221, 119)


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


class Game():
    def __init__(self):
        pygame.init()

        self.gameDisplay = pygame.display.set_mode(
           (display_width, display_height))
        self.clock = pygame.time.Clock()

        self.player = Player(int(display_width*0.75),
                             int(display_height*0.75), 20, fps, gamespeed)
        self.ball = Ball(int(display_width*0.5),
                         int(display_height*0.5), 12, fps, gamespeed)

        self.memory = [] # game history will be kept here to use for learning

    def reset_game(self):
        self.player.reset(int(display_width*0.75), int(display_height*0.75))
        self.ball.reset(int(display_width*0.5), int(display_height*0.5))
        self.memory = []

    def distance_between_two_points(self, player, ball):
        return math.sqrt(np.sum((ball.coords-player.coords)**2))

    def draw_player(self, player):
        pygame.draw.circle(self.gameDisplay, black, (int(player.coords[0]),
                                                int(player.coords[1])), player.kick_radius, 1)
        pygame.draw.circle(self.gameDisplay, black, (int(
            player.coords[0]), int(player.coords[1])), player.radius+1)
        pygame.draw.circle(self.gameDisplay, white, (int(player.coords[0]),
                                                int(player.coords[1])), int(player.radius*0.75))

    def draw_ball(self, player):
        pygame.draw.circle(self.gameDisplay, black,
                           (int(self.ball.coords[0]), int(self.ball.coords[1])), self.ball.radius+1)
        pygame.draw.circle(self.gameDisplay, white,
                           (int(self.ball.coords[0]), int(self.ball.coords[1])), self.ball.radius-2)

    def draw_map(self):
        # left
        pygame.draw.line(self.gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                              2 - map_height/2), (display_width/2 - map_width/2, display_height/2 + map_height/2), 1)
        pygame.draw.line(self.gameDisplay, green, (display_width/2 - map_width/2, display_height/2 -
                                              goal_height/2), (display_width/2 - map_width/2, display_height/2 + goal_height/2), 1)
        # right
        pygame.draw.line(self.gameDisplay, black, (display_width/2 + map_width/2, display_height /
                                              2 - map_height/2), (display_width/2 + map_width/2, display_height/2 + map_height/2), 1)
        # top
        pygame.draw.line(self.gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                              2 - map_height/2), (display_width/2 + map_width/2, display_height/2 - map_height/2), 1)
        # bottom
        pygame.draw.line(self.gameDisplay, black, (display_width/2 - map_width/2, display_height /
                                              2 + map_height/2), (display_width/2 + map_width/2, display_height/2 + map_height/2), 1)

    def kick(self, player, ball):
        dist = self.distance_between_two_points(player, ball)
        if dist <= player.kick_radius + ball.radius:
            vector = player.coords - ball.coords
            ball.velocity = ball.velocity - vector/dist * 20

    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    def is_ball_close_to_walls(self, ball):
        # TODO not just walls but all collisions in general, especially other players

        if ball.coords[0] <= display_width/2 - map_width/2 + 2*ball.radius:
            return True

        if ball.coords[0]  >= display_width/2 + map_width/2 - 2*ball.radius:
            return True

        if ball.coords[1]  >= display_height/2 + map_height/2 - 2*ball.radius:
            return True

        if ball.coords[1] <= display_height/2 - map_height/2 + 2*ball.radius:
            return True

        return False

    def check_collisions(self, player, ball):
        dist = self.distance_between_two_points(player, ball)
        if dist < player.radius + ball.radius:

            # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
            ray = ball.velocity
            v = player.coords - ball.coords
            nrm = v / np.sqrt(np.sum(v**2))

            # fix position once the ball is "inside the player"
            # TODO it is based now on the position of balls. what if the ball is so fast that it appears in the other half of the player?

            # if the ball is close to the walls, it cannot move further - so we take a different approach then, so that the player cannot move onto ball
            if self.is_ball_close_to_walls(self.ball):
                middle = (ball.coords + player.coords)/2

                ball.coords = middle-0.5*v*(player.radius+ball.radius)/dist

                player.coords = middle+0.5*v*(player.radius+ball.radius)/dist
            else:
                ball.coords = player.coords-v*(player.radius+ball.radius)/dist

            # if the ball hits the player, then it should bounce
            if not(ball.velocity[0] == 0 and ball.velocity[0] == 0):
                if angle_between(ray, nrm) < 1.57:
                    reflection = ray - (2 * (np.dot(ray, nrm)) * nrm)
                    ball.velocity = reflection

            player_velocity = np.sqrt(np.sum(player.velocity**2))
            nrm *= player_velocity

            # TODO some stickyness parameter?
            # we want the balls' velocity to be decided not only by player's velocty, but also by theirs position
            ball.velocity = ball.velocity + (player.velocity*7-nrm[0])*0.6*0.125
            player.velocity = player.velocity * 0.9

    def check_borders_ball(self, ball):
        # left border
        if ball.coords[0] <= display_width/2 - map_width/2 + ball.radius:
            if ball.coords[1] > display_height/2 - goal_height/2 and ball.coords[1] < display_height/2 + goal_height/2:
                # if tha ball is fully inside of the goal:
                if ball.coords[0] <= display_width/2 - map_width/2 - ball.radius:
                    ball.in_goal = True
            else:
                ball.coords[0] = display_width/2 - map_width/2 + ball.radius
                ball.velocity[0] *= -1

        # right border
        if ball.coords[0] >= display_width/2 + map_width/2 - ball.radius:
            ball.coords[0] = display_width/2 + map_width/2 - ball.radius
            ball.velocity[0] *= -1

        # bottom border
        if ball.coords[1] >= display_height/2 + map_height/2 - ball.radius:
            ball.coords[1] = display_height/2 + map_height/2 - ball.radius
            ball.velocity[1] *= -1

        # top border
        if ball.coords[1] <= display_height/2 - map_height/2 + ball.radius:
            ball.coords[1] = display_height/2 - map_height/2 + ball.radius
            ball.velocity[1] *= -1

    def check_borders_player(self, player):
        # left border
        if player.coords[0] <= 0:
            player.coords[0] = 0
            player.velocity[0] = 0

        # right border
        if player.coords[0] >= display_width:
            player.coords[0] = display_width
            player.velocity[0] = 0

        # bottom border
        if player.coords[1] >= display_height:
            player.coords[1] = display_height
            player.velocity[1] = 0

        # top border
        if player.coords[1] <= 0:
            player.coords[1] = 0
            player.velocity[1] = 0

    def get_random_move(self):
        x_change = 0
        y_change = 0
        kick = 0

        if random.randint(0, 1) == 0:
            # move sideways
            if random.randint(0, 1) == 0:
                x_change += -0.7 * gamespeed
            else:
                x_change -= -0.7 * gamespeed

        if random.randint(0, 1) == 0:
            # move up/down
            if random.randint(0, 1) == 0:
                y_change += -0.7 * gamespeed
            else:
                y_change -= -0.7 * gamespeed

        if random.randint(0, 2) == 0:
            kick = 1

        return np.array([x_change, y_change]), kick

    def is_player_outside(self, player):
        # left border
        if player.coords[0] <= display_width/2 - map_width/2 + player.radius:
            return True

        # right border
        if player.coords[0] >= display_width/2 + map_width/2 - player.radius:
            return True

        # bottom border
        if player.coords[1] >= display_height/2 + map_height/2 - player.radius:
            return True

        # top border
        if player.coords[1] <= display_height/2 - map_height/2 + player.radius:
            return True

        return False

    def game_loop(self, TrainNet=None, TargetNet=None, epsilon=None, copy_step=None):
        gameExit = False
        done = False
        losses = list()
        rewards = 0
        iter = 0

        while not (gameExit or done):
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pygame.quit()
            #         quit()

            # keys = pygame.key.get_pressed()

            # x_change = 0
            # y_change = 0

            # if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            #     x_change += -0.7 * gamespeed
            # if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            #     x_change += 0.7 * gamespeed
            # if keys[pygame.K_UP] or keys[pygame.K_w]:
            #     y_change += -0.7 * gamespeed
            # if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            #     y_change += 0.7 * gamespeed
            # if keys[pygame.K_SPACE]:
            #     self.kick(self.player, self.ball)

            # vel_change = np.array([x_change, y_change])

            # TODO somehow find a solution to keep intuitive controls while increasing gamespeed
            # gamespeed also PROBABLY should be reduced by slow_param**gamespeed?

            observations = self.get_game_state()
            action = TrainNet.get_action(observations, epsilon)
            prev_observations = observations
            vel_change, isKick = self.make_action(action)

            # applying physics
            if(isKick):
                self.kick(self.player, self.ball)

            self.player.velocity = self.player.velocity + vel_change

            self.player.update_velocity()
            self.player.apply_velocity()

            self.ball.update_velocity()
            self.ball.apply_velocity()

            self.check_collisions(self.player, self.ball)
            self.check_borders_ball(self.ball)
            self.check_borders_player(self.player)
            # done applying physics

            reward = -3

            if self.is_player_outside(self.player):
                reward += 4

            # if self.ball.in_goal:
            #     done = True
            #     reward = 1000

            if iter == int(300):
                done = True

            rewards += reward

            observations = self.get_game_state()
            exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
            TrainNet.add_experience(exp)
            loss = TrainNet.train(TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            iter += 1
            if iter % copy_step == 0:
                TargetNet.copy_weights(TrainNet)


            self.gameDisplay.fill(green)
            self.draw_map()
            self.draw_player(self.player)
            self.draw_ball(self.ball)

            pygame.display.update()

            #print(self.ball.x_velocity, self.ball.y_velocity)
            self.clock.tick(fps)

        self.reset_game()
        return rewards, mean(losses)

    def make_action(self, action):
        x_change = 0
        y_change = 0
        kick = 0

        # 18 actions

        if action >= 9:
            kick = 1

        action = action - 9

        if action == 1:
            y_change += 0.7 * gamespeed
        if action == 2:
            y_change += 0.7 * gamespeed
            x_change += 0.7 * gamespeed
        if action == 3:
            x_change += 0.7 * gamespeed
        if action == 4:
            y_change -= 0.7 * gamespeed
            x_change += 0.7 * gamespeed
        if action == 5:
            y_change -= 0.7 * gamespeed
        if action == 6:
            y_change -= 0.7 * gamespeed
            x_change -= 0.7 * gamespeed
        if action == 7:
            x_change -= 0.7 * gamespeed
        if action == 8:
            y_change += 0.7 * gamespeed
            x_change -= 0.7 * gamespeed

        return np.array([x_change, y_change]), kick

    def get_game_state(self):
        data = np.concatenate((self.player.coords/map_width, self.player.velocity/25, self.ball.coords/map_width, self.ball.velocity/25))
        data.reshape((-1, 1))
        return data
