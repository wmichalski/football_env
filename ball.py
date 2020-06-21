import math

class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.x_velocity = 0.001
        self.y_velocity = 0.001

    def update_velocity(self):
        self.x_velocity = self.x_velocity*0.95
        self.y_velocity = self.y_velocity*0.95
        if self.x_velocity > 40:
            self.x_velocity = 40
        if self.y_velocity > 40:
            self.y_velocity = 40
        if self.x_velocity == 0:
            self.x_velocity = 0.001
        if self.y_velocity == 0:
            self.y_velocity = 0.001

    def apply_velocity(self):
        self.x += self.x_velocity
        self.y += self.y_velocity