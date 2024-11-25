from utils.data_loader import ccf_map, TRIG_RADIUS
from utils.data_struct import ArenaMap
import numpy as np
import pygame, pathlib, os
PKG_ROOT = pathlib.Path(__file__).parent.resolve()

class Modulo(ArenaMap):
    def __init__(self):
        super().__init__()

        # arena information
        self.tile_center = ccf_map()
        self.tiles = np.array([self.tile_center[0],
                               self.tile_center[1]])
        self.n_tiles = self.tiles.shape[1]
        self.arena_center = None

        self._init_target()
        self.current = self.target[:, self.target_count].reshape([2, -1])

        # auditory information
        # dB SPL
        self.base = 73
        self.bg = 20
        self.noise = 5
        # ref distance (mm)
        self.dr = 100

        pygame.mixer.init()
        file_path = os.path.join(PKG_ROOT, 'chirp.wav')
        self.chirp = pygame.mixer.Sound(file_path)

    def _init_target(self, n=16, stratified=True):
        # n_tiles choose n
        if stratified:
            n_split = np.floor(self.n_tiles / n).astype(int)
            self.target_idx = np.concatenate([np.random.choice(range(i * n_split, (i + 1) * n_split),
                                                               1, replace=False) for i in range(n)])
            np.random.shuffle(self.target_idx)

        else:
            self.target_idx = np.random.choice(self.n_tiles, n, replace=False)

        self.target = self.tiles[:, self.target_idx]
        self.target_count = 0

    def sample_tile(self):
        return self.tiles[:, np.random.choice(self.n_tiles, 1, replace=False)]

    def draw_current(self, plot_ax):
        plot_ax.plot(self.current[0], self.current[1], 'gx')

    def check_capture(self, pos):
        dist = self.distance(pos)

        if dist <= TRIG_RADIUS:
            self.capture()
            return True
        
        return False

    def capture(self):
        self.target_count += 1
        if self.target_count < self.target.shape[1]:
            self.current = self.target[:, self.target_count].reshape([2, -1])

    def distance(self, pos):
        '''
        Calculate the distance to the target position
        pos: numpy array of positions [2, N]
        '''
        return np.linalg.norm(self.current - pos, axis=0)

    def sound_level(self, dist=-1, pos=None):
        '''
        Calculate the sound level (dB)
        '''
        # add 10 mm to distance to avoid log(0)

        if dist == -1:
            dist = self.distance(pos)
        
        return self.base - self.bg - 20 * np.log10((dist + 10) / self.dr)
    
    def sound_volume(self, dist=-1, pos=None):
        '''
        Calculate sound volume (for demo, not calibrated)
        '''
        if dist == -1:
            dist = self.distance(pos)

        return np.clip(self.dr / dist, 0, 1)