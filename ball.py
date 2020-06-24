import math

class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        # avoiding a 0 in velocity vector to simplify math
        self.x_velocity = 0.001
        self.y_velocity = 0.001
        self.in_goal = False

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.x_velocity = 0.001
        self.y_velocity = 0.001
        self.in_goal = False

    def update_velocity(self):
        self.x_velocity = self.x_velocity*0.96
        self.y_velocity = self.y_velocity*0.96
        # sometimes ball gets too fast for some reason
        speed = math.sqrt(self.x_velocity**2 + self.y_velocity**2)
        if speed > 40:
            self.x_velocity *= 40/speed
            self.y_velocity *= 40/speed
        # avoiding a 0 in velocity vector to simplify math
        if self.x_velocity == 0:
            self.x_velocity = 0.001
        if self.y_velocity == 0:
            self.y_velocity = 0.001

    def apply_velocity(self):
        self.x += self.x_velocity
        self.y += self.y_velocity