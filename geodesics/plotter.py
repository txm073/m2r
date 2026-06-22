import os
from . import *
from .solver import GeodesicSolver
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator


class Plotter:
   
  def __init__(
    self, 
    solver: GeodesicSolver, 
    plot2d: bool = True, 
    plot3d: bool = False
  ) -> None:
    self.solver = solver
    self.fig = plt.figure()
    args = (1, 2, 1) if plot2d and plot3d else (1, 1, 1)
    self.ax2d = self.fig.add_subplot(*args)
    args = (1, 2, 2) if plot2d and plot3d else (1, 1, 1)
    self.ax3d = self.fig.add_subplot(*args, projection='3d')  
    # self.ax3d.set_visible(False)
    self.plotted_2d, self.plotted_3d = plot2d, plot3d  
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
      xlim = (
        -(abs(self.solver.homotopy[0]) + 1) * 1.2,
        (abs(self.solver.homotopy[0]) + 1) * 1.2
      )
    if ylim is None:
      ylim = (
        -(abs(self.solver.homotopy[1]) + 1) * 1.2,
        (abs(self.solver.homotopy[1]) + 1) * 1.2
      )
    self.ax2d.set_xlim(*xlim)
    self.ax2d.set_ylim(*ylim)
    self.line2d, *_ = self.ax2d.plot([], [], marker='o', color='red')
    self.text = self.ax2d.text(
      0.02, 0.95, '',
      transform=self.ax2d.transAxes,
      va='top',
      color='red'
    )
    X, Y = torch.meshgrid(
      torch.linspace(*xlim, 100), 
      torch.linspace(*ylim, 100), 
      indexing='ij'
    )
    # print(X, Y)
    data = self.det(X, Y)
    print(type(data))
    print(data.shape)
    im = self.ax2d.imshow(
      data,
      extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
      origin='lower',
      cmap='viridis'
    )
    self.fig.colorbar(im, ax=self.ax2d)
    self.plotted_2d = True

  def plot_3d(
    self, 
    torus: Torus,
    xlim: tuple[float, float] = None, 
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None
  ) -> None:
    self.ax3d.set_title('Torus plot')
    if xlim is not None:
      self.ax3d.set_xlim(*xlim)
    if ylim is not None:
      self.ax3d.set_ylim(*ylim)
    if zlim is not None:
      self.ax3d.set_zlim(*zlim)
    self.torus = torus
    theta_ax = torch.linspace(0, 2 * torch.pi, 50)
    phi_ax = torch.linspace(0, 2 * torch.pi, 50)
    theta, phi = torch.meshgrid(theta_ax, phi_ax, indexing='xy')
    r = torus.r_torch
    X = (torus.R + r(theta, phi) * torch.cos(phi)) * torch.cos(theta)
    Y = (torus.R + r(theta, phi) * torch.cos(phi)) * torch.sin(theta)
    Z = r(theta, phi) * torch.sin(phi)
    self.ax3d.plot_surface(X, Y, Z, alpha=0.5, antialiased=False)
    self.line3d, *_ = self.ax3d.plot([], [], [], marker='o', color='red')
    self.interp = RegularGridInterpolator((theta_ax, phi_ax), Z, bounds_error=False)
    self.plotted_3d = True
    
  def show(
    self,
    delay: float = 1.0,
    save: str = None,
    frame_save_interval: int = 1
  ) -> None:
    def update(frame_index: int) -> list:
      items = []
      if self.plotted_2d:
        X, Y = zip(*self.solver.curves[frame_index])
        self.line2d.set_data(X, Y)
        self.text.set_text(
          f'{frame_index+1}/{self.solver.iterations}: length={self.solver.lengths[frame_index]:.6f}'
        )
        items.append(self.line2d)
        items.append(self.text)
      if self.plotted_3d:
        X, Y, Z = self.torus(
          self.solver.curves[frame_index][:, 0], 
          self.solver.curves[frame_index][:, 1] 
        )
        self.line3d.set_data(X, Y)
        self.line3d.set_3d_properties(Z)
        items.append(self.line3d)
      return items  
    if not self.plotted_2d:
      self.fig.delaxes(self.ax2d)
    if not self.plotted_3d:
      self.fig.delaxes(self.ax3d)
    plt.get_current_fig_manager().resize(1600, 900)
    plt.tight_layout()
    # plt.show(block=False)
    if save is not None:
      for i in range(self.solver.iterations):
        update(i)
        # self.fig.canvas.draw()
        self.fig.canvas.draw()
        if delay != 0:  
          plt.pause(delay)
        if i % frame_save_interval == 0:
          if not os.path.exists(save):
            os.makedirs(save)
          frame_index = i // frame_save_interval
          print(f'Saving frame {frame_index} ({i}/{self.solver.iterations})')
          self.fig.savefig(
            f'{save}/frame_{frame_index:04d}.png',
            dpi=200,
            bbox_inches='tight',
          )
    else:
      anim = FuncAnimation(
        self.fig, update, frames=range(0, self.solver.iterations),
        blit=True, interval=delay * 1000, repeat=False
      )  
    plt.show()