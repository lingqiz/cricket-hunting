import model.env
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesMap:
    '''
    Probabilitstic integration over 2D map
    '''

    @staticmethod
    def plot_map(ax, X, Y, Z):
        c = ax.pcolormesh(X, Y, Z, cmap='viridis')
        plt.colorbar(c, fraction=0.042, pad=0.05)
        ax.set_aspect('equal')

    def __init__(self, x_range=(0, 2300), y_range=(-100, 2200), n_step=25, n_chunk=256):
        '''
        Initialize the map
        '''
        # map resolution
        self.x_range = x_range
        self.y_range = y_range
        self.n_step = n_step
        self.env = model.env.ModuloData(target=None,
                                        current=None)

        # map grid
        self.x = torch.arange(*x_range, n_step, dtype=torch.float32)
        self.y = torch.arange(*y_range, n_step, dtype=torch.float32)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='xy')
        self.Z = torch.zeros_like(self.X)
        self.gridshape = self.Z.shape

        # xy coordinates
        self.xy = torch.stack([self.X.flatten(),
                               self.Y.flatten()], dim=0)

        # map coordinates within bounds
        inbnd = self.env.check_boundary(self.xy.numpy())
        self.inbnd = torch.tensor(inbnd, dtype=torch.float32).reshape(self.gridshape)

        # likehood model
        self.exp = 1.0
        self.sigma = 20

        # chunk area size for computing probability
        chunk_ratio = self.inbnd.sum() / self.inbnd.numel() / n_chunk
        self.chunk_area = chunk_ratio * (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.chunk_rad = torch.sqrt(self.chunk_area / np.pi)

    def init(self, current, target=None):
        self.Z = torch.zeros_like(self.Z)

        self.env.current = current.reshape([2, 1])
        if target is not None:
            self.env.target = target

    def prob_map(self, density=False):
        '''
        Compute the normalized probability map
        '''
        norm_Z = self.Z - torch.max(self.Z)
        norm_Z = torch.exp(norm_Z) * self.inbnd

        if density:
            return norm_Z / (torch.sum(norm_Z) * self.dx * self.dy)

        return norm_Z / (torch.sum(norm_Z))

    def prob_loc(self, loc):
        '''
        Compute the probability of a location
        '''
        prob_map = self.prob_map(density=False)

        loc = loc.reshape([2, 1])
        dist = torch.norm(self.xy - loc, dim=0)
        mask = (dist <= self.chunk_rad).reshape(self.gridshape)
        bound_ratio = self.inbnd[mask].sum() / mask.sum()

        return prob_map[mask].sum() / bound_ratio

    def loudness(self, dist):
        '''
        logarithmic loudness model, same as Env.Modulo
        add a fix base distance (25 mm) to avoid log(0)
        '''
        base = 73
        bg = 20
        ref_dir = 100

        return base - bg - 20 * torch.log10((dist + 25) / ref_dir)

    def log_l(self, m, p):
        '''
        Compute the log likelihood map given
        sound measurement m and mouse location p,
        based on the loudness model
        '''
        dist = torch.norm(self.xy - p, dim=0)
        m_exp = self.loudness(dist)

        log_l = -torch.pow(torch.abs(m_exp - m) / self.sigma, self.exp)
        return log_l.reshape(self.gridshape)

    def stop_llhd(self, p):
        # simulate the sound level (measurement),
        # no noise is added
        p = p.reshape([2, 1])
        m = self.env.sound_level(pos=p)

        log_l = self.log_l(torch.tensor(m, dtype=torch.float32),
                           torch.tensor(p, dtype=torch.float32))
        return log_l

    def step(self, m, p, mode='integrate'):
        '''
        Update the map with the measurement m
        at position p
        '''
        # compute the log likelihood
        log_l = self.log_l(m, p)

        # update the map
        if mode == 'integrate':
            self.Z += log_l

        elif mode == 'current':
            self.Z = log_l

        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        return

    def stop_step(self, p, mode='integrate'):
        '''
        Update the map with the measurement m
        at position p
        '''
        p = p.reshape([2, 1])
        m = self.env.sound_level(pos=p)

        self.step(torch.tensor(m, dtype=torch.float32),
                  torch.tensor(p, dtype=torch.float32),
                  mode=mode)

    def plot_llhd(self, loc, log_l, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        self.plot_map(ax, self.X, self.Y, log_l)
        ax.plot(loc[0], loc[1], 'o', color='orange')
        ax.plot(self.env.current[0],
                self.env.current[1], 'ro')

        self.env.draw_boundary(ax)
        ax.set_title('Log Likelihood')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

    def plot_density(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        self.plot_map(ax, self.X, self.Y, self.prob_map(density=True))
        self.env.draw_boundary(ax)
        ax.set_title('Probability')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')