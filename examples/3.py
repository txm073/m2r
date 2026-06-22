import sys
sys.path.extend(('.', '..'))
from geodesics import GeodesicSolver, Plotter, Torus
from geodesics.utils import const
import sympy as sp
import torch

if __name__ == '__main__':
  # Parameters
  n_points = 10
  n_its = 200
  lr = 0.0025
  homotopy = (2, 1)
  R = 3
  r = lambda x, y: 1 + sp.cos(x) ** 2
  torus = Torus(R, r)

  g11 = lambda x, y: 4 * torch.cos(x) ** 2 * torch.sin(x) ** 2 \
    + (3 + torch.cos(y) + torch.cos(y) * torch.cos(x) ** 2) ** 2
  g22 = lambda x, y: 1 + 2 * torch.cos(x) ** 2 + torch.cos(x) ** 4
  G = [
    [g11, const(0.0)],
    [const(0.0), g22]
  ]
  solver = GeodesicSolver(n_points, homotopy, G)
  solver.initialise(start=(0, 0), init_mode='random')
  solver.minimise(n_its, lr, verbose=True)
  xlim, ylim = (-1, 4), (-1, 4)
  plotter = Plotter(solver, True, True)
  plotter.plot_2d(xlim, ylim)
  plotter.ax3d.view_init(40, 15, 5)
  plotter.plot_3d(torus)
  plotter.show(0, save='./animations/3/', frame_save_interval=5)