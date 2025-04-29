from utils.data_loader import ccf_map, TRIG_RADIUS
from utils.data_struct import ArenaMap
import numpy as np

class Modulo(ArenaMap):
    def __init__(self):
        super().__init__()

        # sound model: 'polynomial', 'logrithmic'
        self.coeff = np.load('./data/loudness_fit.npy')
        self.sound_model = 'logarithmic'

        # arena information
        self.tile_center = ccf_map()
        # tile coordinates [2, N]
        self.tiles = np.array([self.tile_center[0],
                               self.tile_center[1]])
        self.n_tiles = self.tiles.shape[1]
        self.arena_center = None

        # recompute boundary information
        self._init_boundary()

        # auditory information
        # dB SPL
        self.base = 73
        self.bg = 20
        self.noise = 5
        # ref distance (mm)
        self.dr = 100

    def sample_tile(self):
        return self.tiles[:, np.random.choice(self.n_tiles, 1, replace=False)]

    def get_tile(self, index):
        return self.tiles[:, index].reshape([2, -1])

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
        # add fix base distance (25 mm) to avoid log(0)

        if np.any(dist == -1):
            dist = self.distance(pos)

        if self.sound_model == 'logarithmic':
            return self.base - self.bg - 20 * np.log10((dist + 25) / self.dr)

        elif self.sound_model == 'polynomial':
            return np.polyval(self.coeff, dist)


class ModuloData(Modulo):
    def __init__(self, target, current):
        super().__init__()

        self.target = target
        self.current = current

    def draw_current(self, plot_ax):
        plot_ax.plot(self.current[0], self.current[1], 'rx')

    def update_current(self, pos):
        self.current = pos


class ModuloSim(Modulo):
    def __init__(self):
        super().__init__()

        self._init_target()
        self.current = self.target[:, self.target_index].reshape([2, -1])

    def _init_target(self, n_target=16, stratified=True):
        self.n_target = n_target
        # n_tiles choose n
        if stratified:
            n_split = np.floor(self.n_tiles / n_target).astype(int)
            self.target_idx = np.concatenate([np.random.choice(range(i * n_split, (i + 1) * n_split),
                                                        1, replace=False) for i in range(n_target)])
            np.random.shuffle(self.target_idx)

        else:
            self.target_idx = np.random.choice(self.n_tiles, n_target, replace=False)

        # target coordinates
        self.target = self.tiles[:, self.target_idx]
        self.target_index = 0

    def draw_current(self, plot_ax):
        plot_ax.plot(self.current[0], self.current[1], 'gx')

    def check_capture(self, pos):
        dist = self.distance(pos)

        if dist <= TRIG_RADIUS:
            self.capture()
            return True

        return False

    def capture(self):
        self.target_index += 1
        if self.target_index < self.target.shape[1]:
            self.current = self.target[:, self.target_index].reshape([2, -1])

    def sound_volume(self, dist=-1, pos=None):
        '''
        Calculate sound volume (for demo, not calibrated)
        '''
        if np.any(dist == -1):
            dist = self.distance(pos)

        return np.clip(self.dr / dist, 0, 1)