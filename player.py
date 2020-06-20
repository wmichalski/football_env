class Player:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.kick_radius = radius+10
        self.x_velocity = 0
        self.y_velocity = 0

    def update_velocity(self):
        self.x_velocity = self.x_velocity*0.9
        self.y_velocity = self.y_velocity*0.9

    def apply_velocity(self):
        self.x += self.x_velocity
        self.y += self.y_velocity