import math

class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.x_velocity = 0
        self.y_velocity = 0

    def update_velocity(self):
        self.x_velocity = self.x_velocity*0.95
        self.y_velocity = self.y_velocity*0.95

    def apply_velocity(self):
        self.x += self.x_velocity
        self.y += self.y_velocity