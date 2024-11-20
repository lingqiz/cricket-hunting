from utils.data_loader import ccf_map
from utils.data_struct import ArenaMap
import numpy as np

class TaskEnv(ArenaMap):
    def __init__(self):
        super().__init__()
        self.tile_centers = ccf_map()
        self.tiles = np.array([self.tile_centers[0],
                               self.tile_centers[1]]).T
        self.target = None
        self.arena_center = None
