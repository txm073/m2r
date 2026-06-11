from . import *
from .solver import GeodesicSolver
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import LinearNDInterpolator


class Plotter:
   
  def __init__(self, solver: GeodesicSolver) -> None:
    self.solver = solver
    self.fig = plt.figure()
    self.ax2d = self.fig.add_subplot(1, 2, 1)
    self.ax3d = self.fig.add_subplot(1, 2, 2, projection='3d')  
    self.plotted_2d, self.plotted_3d = False, False  
    self.det = lambda x, y: self.solver.G[0][0](x, y) \
      * self.solver.G[1][1](x, y) - self.solver.G[0][1](x, y) ** 2
    self.interp = None

  def plot_2d(
    self, 
    xlim: tuple[float, float] = None, 
    ylim: tuple[float, float] = None
   ) -> None:
    self.ax2d.set_title('Colour map')
    if xlim is None:
      xlim = (-abs(self.solver.homotopy[0] * 1.5), abs(self.solver.homotopy[0] * 1.5))
    if ylim is None:
      ylim = (-abs(self.solver.homotopy[1] * 1.5), abs(self.solver.homotopy[1] * 1.5))
    self.ax2d.set_xlim(xlim)
    self.ax2d.set_ylim(ylim)
    self.line2d, *_ = self.ax2d.plot([], [], marker='o', color='red')
    self.text = self.ax2d.text(
      0.02, 0.95, '',
      transform=self.ax2d.transAxes,
      va='top',
      color='red'
    )
    im = self.ax2d.imshow(
      self.det(*torch.meshgrid(torch.linspace(*xlim, 100), torch.linspace(*ylim, 100), indexing='xy')),
      extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
      origin='lower',
      cmap='viridis'
    )
    self.fig.colorbar(im, ax=self.ax2d)
    self.plotted_2d = True

  def plot_3d(
    self, 
    xlim: tuple[float, float] = None, 
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None
   ) -> None:
    self.ax3d.set_title('Surface plot')
    if xlim is None:
      xlim = (-abs(self.solver.homotopy[0] * 1.5), abs(self.solver.homotopy[0] * 1.5))
    if ylim is None:
      ylim = (-abs(self.solver.homotopy[1] * 1.5), abs(self.solver.homotopy[1] * 1.5))
    if zlim is None:
      zlim = (-5, 5)
    self.ax3d.set_xlim(*xlim)
    self.ax3d.set_ylim(*ylim)
    # self.ax3d.set_zlim(*zlim)
    X, Y = torch.meshgrid(
      torch.linspace(*xlim, 50),
      torch.linspace(*ylim, 50),
      indexing='xy'
    )
    Z = self.det(X, Y)
    self.interp = LinearNDInterpolator(
      list(zip(X.ravel(), Y.ravel())),
      Z.ravel()
    )
    self.ax3d.plot_surface(X, Y, Z, alpha=0.65)
    self.line3d, *_ = self.ax3d.plot([], [], [], marker='o', color='red')
    self.plotted_3d = True

  def plot_surface(
    self, 
    torus: Torus,
    xlim: tuple[float, float] = None, 
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None
  ) -> None:
    self.ax3d.set_title('Surface plot')
    theta, phi = torch.meshgrid(
      torch.linspace(0, 2 * torch.pi, 50), 
      torch.linspace(0, 2 * torch.pi, 50),
      indexing='xy'
    )
    r = torus.r_torch
    X = (torus.R + r(theta, phi) * torch.cos(phi)) * torch.cos(theta)
    Y = (torus.R + r(theta, phi) * torch.cos(phi)) * torch.sin(theta)
    Z = r(theta, phi) * torch.sin(phi)
    self.ax3d.plot_surface(X, Y, Z, antialiased=False)

  def show(
    self,
    delay: float = 1.0
  ) -> None:
    def update(frame_index: int) -> list:
      items = []
      X, Y = zip(*self.solver.curves[frame_index])
      if self.plotted_2d:
        self.line2d.set_data(X, Y)
        self.text.set_text(
          f'{frame_index+1}/{self.solver.iterations}: length={self.solver.lengths[frame_index]:.6f}'
        )
        items.append(self.line2d)
        items.append(self.text)
      if self.plotted_3d:
        # print(self.solver.curves[frame_index].shape)
        # exit(0)
        assert self.interp is not None
        Z = self.interp(X, Y)
        self.line3d.set_data(X, Y)
        self.line3d.set_3d_properties(Z)
        items.append(self.line3d)
      return items  
    anim = FuncAnimation(
      self.fig, 
      update, 
      frames=range(self.solver.iterations), 
      interval=delay * 1000, 
      repeat=False,
      blit=True
    )
    plt.get_current_fig_manager().resize(1600, 900)
    plt.tight_layout()
    plt.show()
