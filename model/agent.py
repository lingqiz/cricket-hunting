import numpy as np

class Agent():
    def __init__(self, arena):
        self.arena = arena
        self.loc = arena.sample_tile().reshape([2, -1])

    def get_loc(self):
        return self.loc