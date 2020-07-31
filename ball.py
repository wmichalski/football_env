import math
import numpy as np

class Ball:
    def __init__(self, x, y, radius, fps, gamespeed):
        self.coords = np.array([x, y])
        self.radius = radius
        # avoiding a 0 in velocity vector to simplify math
        self.velocity = np.array([0.001, 0.001])
        self.in_goal = False
        self.fps = fps
        self.gamespeed = gamespeed

    def reset(self, x, y):
        self.coords = np.array([x, y])
        self.velocity = np.array([0.001, 0.001])
        self.in_goal = False

    def update_velocity(self):
        slow_param = (0.96)**(self.gamespeed)

        self.velocity = self.velocity * slow_param
        # sometimes ball gets too fast for some reason
        speed = math.sqrt(np.sum(self.velocity**2))
        if speed > 40:
            self.velocity = self.velocity * 40/speed
        # avoiding a 0 in velocity vector to simplify math
        self.velocity = np.where(self.velocity == 0, 0.001, self.velocity)

    def apply_velocity(self):
        self.coords = self.coords + self.velocity * self.gamespeed