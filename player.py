class Player:
    def __init__(self, x, y, radius, fps, gamespeed):
        self.x = x
        self.y = y
        self.radius = radius
        self.kick_radius = radius+10
        self.x_velocity = 0
        self.y_velocity = 0
        self.fps = fps
        self.gamespeed = gamespeed

    def update_velocity(self):
        slow_param = (0.9)**(60/self.fps * self.gamespeed)

        self.x_velocity = self.x_velocity*slow_param
        self.y_velocity = self.y_velocity*slow_param

    def apply_velocity(self):
        self.x += self.x_velocity * 60 / self.fps * self.gamespeed
        self.y += self.y_velocity * 60 / self.fps * self.gamespeed