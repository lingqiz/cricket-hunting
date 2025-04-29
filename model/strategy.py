import model.env
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesMap:
    '''
    Probabilitstic integration over 2D map
    '''
    def __init__(self, x_range=(0, 2300), y_range=(-100, 2200), n_step=25):
        # map resolution
        self.x_range = x_range
        self.y_range = y_range
        self.n_step = n_step
        self.env = model.env.ModuloData(target=None,
                                        current=None)

        # map grid
        self.x = torch.arange(0, 2300, 25)
        self.y = torch.arange(-100, 2200, 25)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='xy')
        self.Z = torch.zeros_like(self.X)

        # xy coordinates
        self.xy = torch.stack([self.X.flatten(),
                               self.Y.flatten()], dim=0)

        # map coordinates within bounds
        inbnd = self.env.check_boundary(self.xy.numpy())
        self.inbnd = torch.tensor(inbnd, dtype=torch.float32).reshape(self.Z.shape)

    def init(self, current, target=None):
        self.Z = torch.zeros_like(self.Z)

        self.env.current = current
        if target is not None:
            self.env.target = target
