import re
import sys
import torch
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

Vec2 = tuple[torch.Tensor | float, torch.Tensor | float]
ScalarFunc = Callable[[torch.Tensor | float, torch.Tensor | float], torch.Tensor | float]
Metric = list[list[ScalarFunc]]

def innerprod(G: Metric, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
  m = G[0][0](*w) * u[0] * v[0] \
    + G[0][1](*w) * (u[0] * v[1] + u[1] * v[0]) \
    + G[1][1](*w) * u[1] * v[1]
  assert not m.isnan(), f'overflow error'
  assert m.item() >= 0, f'provided metric is not SPD <{u.tolist()},{v.tolist()}>_{w.tolist()} = {m.item()}'
  return m

def calculate_length(G: Metric, points: torch.Tensor, separation: float = 0.0) -> torch.Tensor:
  l = 0.0
  segments = torch.zeros(len(points), dtype=torch.float)
  for i in range(len(points) - 1):
    u = points[i+1] - points[i]
    s = torch.sqrt(innerprod(G, u, u, points[i]))
    l += s 
    if separation:
      segments[i].copy_(s)
  if not separation:
    return l
  m = segments.mean().detach()
  return l + abs(separation) * ((segments - m) ** 2).mean()


@dataclass
class Curve:
  N: int
  homotopy: tuple[int, int]
  G: Metric

  def initialise(
    self,
    init_mode: str = 'random', 
    start: tuple[float, float] = None
  ) -> None:
    if start is None:
      self.start = torch.rand(2, dtype=torch.float) 
    else:
      self.start = torch.tensor(start, dtype=torch.float).frac()
    self.end = self.start + torch.tensor(self.homotopy)
    if init_mode == 'linear':
      self.points = torch.tensor(np.linspace(self.start.numpy(), self.end.numpy(), self.N + 1))
    elif init_mode == 'random':
      self.points = self.start + (self.end - self.start) * torch.rand(self.N + 1, 2)
      self.points[0] = self.start
      self.points[-1] = self.end
    else:
      raise Exception('init_mode must be "linear" or "random"')
    self.points.requires_grad_(True)

  def minimise(
    self, 
    iterations: int, 
    step: float,
    separation: float = None,
    verbose: bool = False
  ) -> None:
    self.iterations = iterations
    self.step = step
    self.separation = separation
    if self.separation is None:
      self.separation = self.N ** 2
    initial_length = calculate_length(self.G, self.points)
    self.curves = [self.points.detach()]
    self.lengths = [initial_length.detach().item()]
    points = self.points.detach().clone()
    points.requires_grad_(True)
    stage = max(self.iterations // 10, 1)
    for it in range(self.iterations):
      if verbose and it % stage == 0:
        print(f'Computing {it}/{self.iterations}...')
      loss = calculate_length(self.G, points, separation=self.separation)
      loss.backward()
      with torch.no_grad():
        self.lengths.append(calculate_length(self.G, points).detach().clone().item())
        if points[0][0] >= 1.0 or points[0][1] >= 1.0 or points[0][0] < 0.0 or points[0][1] < 0.0:
          delta = -self.step * points.grad[0]
        else:
          delta = torch.tensor(0.0)
        points -= delta.repeat(len(points), 1)
        points -= self.step * points.grad
        points[-1].copy_(points[0] + torch.tensor(self.homotopy))
      points.grad.zero_()
      self.curves.append(points.detach().clone())

  def plot(
    self, 
    delay: float = 1.0, 
    xlim: tuple[float, float] = None, 
    ylim: tuple[float, float] = None
  ) -> None:
    fig, ax = plt.subplots()
    if xlim is None:
      xlim = (-abs(self.homotopy[0] * 1.5), abs(self.homotopy[0] * 1.5))
    if ylim is None:
      ylim = (-abs(self.homotopy[1] * 1.5), abs(self.homotopy[1] * 1.5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    line, *_ = ax.plot([], [], marker='o', color='red')
    text = ax.text(
      0.02, 0.95, '',
      transform=ax.transAxes,
      va='top',
      color='red'
    )
    det = lambda x, y: self.G[0][0](x, y) * self.G[1][1](x, y) - self.G[0][1](x, y) ** 2
    im = ax.imshow(
      det(*torch.meshgrid(torch.linspace(*xlim, 100), torch.linspace(*ylim, 100), indexing='xy')),
      extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
      origin='lower',
      cmap='viridis'
    )
    fig.colorbar(im, ax=ax)
    def update(frame_index: int) -> list:
      line.set_data(*zip(*self.curves[frame_index]))
      text.set_text(f'{frame_index+1}/{self.iterations}: length={self.lengths[frame_index]:.6f}')
      return line, text  
    anim = FuncAnimation(
      fig, 
      update, 
      frames=range(self.iterations), 
      interval=delay * 1000, 
      repeat=False,
      blit=True
    )
    plt.show()
  
  def plot3D(
    self, 
    delay: float = 1.0,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None
  ) -> None:
    fig, ax = plt.subplots()
    if xlim is None:
      xlim = (-abs(self.homotopy[0] * 1.5), abs(self.homotopy[0] * 1.5))
    if ylim is None:
      ylim = (-abs(self.homotopy[1] * 1.5), abs(self.homotopy[1] * 1.5))
    if zlim is None:
      zlim = (-abs())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    line, *_ = ax.plot([], [], marker='o', color='red')
    text = ax.text(
      0.02, 0.95, '',
      transform=ax.transAxes,
      va='top',
      color='red'
    )
    det = lambda x, y: self.G[0][0](x, y) * self.G[1][1](x, y) - self.G[0][1](x, y) ** 2
    im = ax.imshow(
      det(*torch.meshgrid(torch.linspace(*xlim, 100), torch.linspace(*ylim, 100), indexing='xy')),
      extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
      origin='lower',
      cmap='viridis'
    )
    fig.colorbar(im, ax=ax)
    def update(frame_index: int) -> list:
      line.set_data(*zip(*self.curves[frame_index]))
      text.set_text(f'{frame_index+1}/{self.iterations}: length={self.lengths[frame_index]:.6f}')
      return line, text  
    anim = FuncAnimation(
      fig, 
      update, 
      frames=range(self.iterations), 
      interval=delay * 1000, 
      repeat=False,
      blit=True
    )
    plt.show()

def input_function(prompt: str) -> ScalarFunc:
  s = input(prompt).replace(' ', '').lower()
  s = re.sub('[A-Za-z_][A-Za-z0-9_]*(?=\\()', lambda m: f'torch.{m.group()}', s)
  return eval(f'lambda x, y: {s}')

def const(v: float) -> ScalarFunc:
  return lambda x, y: torch.full_like(x, v)

def example() -> tuple[int, tuple[int, int], Metric]:
  N = 10
  cls = (2, 3)
  f = lambda x, y: torch.cos(x) ** 2 + torch.sin(y) ** 2 + 1.0
  # f = const(1.0)
  zero = const(0.0)
  G = [[f, zero], [zero, f]]
  it = 1000
  curve = Curve(N, cls, G)
  curve.initialise(start=(0.0, 0.0), init_mode='random')
  curve.minimise(it, 0.0005, separation=100, verbose=True)

  # curve.load_points('points.bin')
  curve.plot(2.0 / float(it), xlim=(-1, 5), ylim=(-1, 5))

def main(argv: list[str]) -> int:
  if len(argv) >= 2 and argv[1] == 'input':
    g11 = input_function('g11(x, y) = ')
    g12 = input_function('g12(x, y) = ')
    g22 = input_function('g22(x, y) = ')
    N = int(input('N = '))
    cls = eval(input('(m, n) = '))
    G = [[g11, g12], [g12, g22]]
    curve = Curve(N, cls, G)
    curve.initialise(start=eval(input('p0: ').strip() or 'None'), init_mode=input('init: ').strip() or 'random')
    curve.minimise(iterations=int(input('iterations: ') or 1000), step=float(input('step: ') or 0.01))
    curve.plot(delay=float(input('delay: ') or 0.01))  
  else:
    example()
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
