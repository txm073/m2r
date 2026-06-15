import sys
sys.path.extend(('.', '..'))
from geodesics import GeodesicSolver, Plotter, Torus
from geodesics.utils import const
import sympy as sp
import torch

if __name__ == '__main__':
  # Parameters
  n_points = 5
  n_its = 50
  lr = 0.05
  homotopy = (2, 3)
  G = [
    [const(1.0), const(0.0)],
    [const(0.0), const(1.0)]
  ]
  # Compute the optimal poly-line
  solver = GeodesicSolver(n_points, homotopy, G)
  solver.initialise()
  solver.minimise(n_its, lr, verbose=True)
  # Create a 2D animated plot
  animation_interval = 1.0 / n_its
  xlim, ylim = (0, 5), (0, 5)
  plotter = Plotter(solver)
  plotter.plot_2d(xlim, ylim)
  plotter.show(0.001, save='./animations/1/', frame_save_interval=5)