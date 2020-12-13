import numpy as np

class Player:
    def __init__(self, x=None, y=None, radius=None, fps=None, gamespeed=None):
        self.action = None
        self.reward = 0
        self.Target = None
        self.Train = None
        self.losses = []

    def init_values(self, x=None, y=None, radius=None, fps=None, gamespeed=None):
        self.coords = np.array([x, y])
        self.radius = radius
        self.kick_radius = radius+10
        self.velocity = np.array([0, 0])
        self.fps = fps
        self.gamespeed = gamespeed

    def reset(self, x, y):
        self.coords = np.array([x, y])
        self.velocity = np.array([0.0, 0.0])

    def update_velocity(self):
        slow_param = (0.9)**(self.gamespeed)

        self.velocity = self.velocity * slow_param

    def apply_velocity(self):
        self.coords = self.coords + self.velocity * self.gamespeed