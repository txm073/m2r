import sys
sys.path.extend(('.', '..'))
from geodesics import GeodesicSolver, Plotter, Torus
from geodesics.utils import const
import sympy as sp
import torch

if __name__ == '__main__':
  # Parameters
  n_points = 10
  n_its = 1000
  lr = 0.01
  homotopy = (0, 1)
  R = 3
  r = 1
  torus = Torus(R, r)
  G = [
    [lambda x, y: (R + r * torch.cos(2 * torch.pi * x)), const(0.0)],
    [const(0.0), const(r ** 2)]
  ]
  # Compute the optimal poly-line
  solver = GeodesicSolver(n_points, homotopy, G)
  solver.initialise(start=(0.1, 0.1), init_mode='random')
  solver.minimise(n_its, lr, verbose=True)
  # Create a 2D animated plot
  # solver.load('points.bin')
  xlim, ylim = (-1, 2), (-1, 2)
  plotter = Plotter(solver, plot2d=True, plot3d=True)
  plotter.plot_2d(xlim, ylim)
  plotter.ax3d.view_init(40, 15, 5)
  plotter.plot_3d(torus, zlim=(-2, 2))
  plotter.show(0.001) #, save='./animations/2/', frame_save_interval=5)