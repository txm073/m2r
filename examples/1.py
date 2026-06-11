import sys
sys.path.extend(('.', '..'))
from geodesics import GeodesicSolver, Plotter, Torus
import sympy as sp
import torch

if __name__ == '__main__':
  # Parameters
  n_points = 10
  n_its = 1000
  lr = 0.001
  homotopy = (1, 1)
  # Torus definition
  torus = Torus(3.0, lambda theta, phi: sp.cos(theta) ** 2 + 0.5)
  r = torus.r_torch
  # Induced metric
  G = torus.metric

  solver = GeodesicSolver(n_points, homotopy, G)
  solver.initialise(init_mode='linear')
  solver.minimise(n_its, lr)

  delay = 1.0 / n_its
  xlim, ylim = (0, 4), (0, 4)
  plotter = Plotter(solver)
  plotter.plot_2d(xlim, ylim)
  plotter.plot_3d(xlim, ylim)
  plotter.show(delay)
