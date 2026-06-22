from geodesics.torus import Torus
from geodesics import GeodesicSolver, Plotter
from geodesics.utils import input_function, const
import torch
import sys


def main(argv: list[str]) -> int:
  g11 = input_function('g11(x, y) = ', 'torch')
  g12 = input_function('g12(x, y) = ', 'torch')
  g22 = input_function('g22(x, y) = ', 'torch')
  N = int(input('N = '))
  cls = eval(input('(m, n) = '))
  G = [[g11, g12], [g12, g22]]
  # N = 10
  # cls = (0, 1)
  # g11 = lambda x, y: 4 * torch.cos(x) ** 2 * torch.sin(x) ** 2 + (3 + torch.cos(y) * (1 + torch.cos(x) ** 2)) ** 2
  # g22 = lambda x, y: 1 + 2 * torch.cos(x) ** 2 + torch.cos(x) ** 4
  # g12 = const(0.0)
  # G = [[g11, g12], [g12, g22]]
  # it = 1000
  # det = lambda x, y: G[0][0](x, y) * G[1][1](x, y) - G[0][1](x, y) ** 2
  # xlim, ylim = (0, 4), (0, 4)
  # X, Y = torch.meshgrid(
  #   torch.linspace(*xlim, 4), 
  #   torch.linspace(*ylim, 4), 
  #   indexing='ij'
  # )

  solver = GeodesicSolver(N, cls, G)
  solver.initialise(start=eval(input('p0: ').strip() or 'None'), init_mode=input('init: ').strip() or 'random')
  solver.minimise(iterations=int(input('iterations: ') or 1000), step=float(input('step: ') or 0.01))
  
  plot2d = input('Display 2D colourmap plot (Y/n): ').strip().lower()
  if not plot2d: plot2d = 'y'
  plot3d = input('Display 3D torus (Y/n): ').strip().lower()
  print(plot3d, plot3d!='n')
  if not plot3d: plot3d = 'y'
  plotter = Plotter(solver, plot2d=plot2d!='n', plot3d=False)
  # xlim, ylim = (0.0, 3.0), (0.0, 3.0)
  if plot2d != 'n':
    plotter.plot_2d()
  if plot3d != 'n':
    R = float(input('Enter the constant R: '))
    
    import sympy as sp
    r = lambda x, y: sp.cos(x) ** 2 + 1
    # r = input_function('r(x, y) = ', prefix='sp')
    torus = Torus(R, r)
    plotter.plot_3d(torus)
  plotter.show(delay=2.0 / solver.iterations)#, save='animations/5', frame_save_interval=5)
  
if __name__ == '__main__':
  main(sys.argv)