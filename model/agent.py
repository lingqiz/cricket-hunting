import numpy as np

class Agent():
    def __init__(self, arena):
        self.arena = arena
        self.loc = arena.sample_tile().reshape([2, -1]) + \
                    np.random.random([2, 1]) * 50.0
        self.ori = np.random.random() * 2.0 * np.pi

    def get_loc(self):
        return self.loc

class GameAgent(Agent):
    def __init__(self, arena):
        super().__init__(arena)

        self.ang_veloc = 2.5 / 180.0 * np.pi
        self.velocity = 5.0

    def turn_left(self):
        self.ori += self.ang_veloc
        self.ori %= 2.0 * np.pi

    def turn_right(self):
        self.ori -= self.ang_veloc
        self.ori %= 2.0 * np.pi

    def move_forward(self):
        new_x = self.loc[0] + self.velocity * np.cos(self.ori)
        new_y = self.loc[1] + self.velocity * np.sin(self.ori)

        new_loc = np.array([new_x, new_y]).reshape([2, -1])
        if self.arena.check_boundary(new_loc):
            self.loc = new_loc