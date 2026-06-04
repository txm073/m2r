import torch
import numpy as np
from typing import Iterable, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field

Vec2 = tuple[torch.Tensor | float, torch.Tensor | float]
Metric = list[list[Callable[[torch.Tensor | float, torch.Tensor | float], torch.Tensor | float]]]

def innerprod(G: Metric, u: Vec2, v: Vec2, w: Vec2) -> torch.Tensor:
  return G[0][0](*w) * u[0] * v[0] \
    + G[0][1](*w) * (u[0] * v[1] + u[1] * v[0]) \
    + G[1][1](*w) * u[1] * v[1]

def calculate_length(G: Metric, points: torch.Tensor, separation: float = 0.0) -> torch.Tensor:
  l = 0.0
  # pieces = []
  for i in range(len(points) - 1):
    u = points[i+1] - points[i]
    s = 1/torch.sqrt(innerprod(G, u, u, points[i]))
    # pieces.append(s)
    l += s 
  # m = l / (len(points) - 1)
  # print(type(m))
  # m = torch.tensor(pieces, requires_grad=False).mean()
  # for i in range(len(points) - 1):
  #   s += abs(pieces[i] - m) * separation
  return l

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
    step: float
  ) -> None:
    self.iterations = iterations
    self.step = step
    initial_length = calculate_length(self.G, self.points)
    self.curves = [self.points.detach()]
    self.lengths = [initial_length.detach().item()]
    points = self.points.detach().clone()
    points.requires_grad_(True)
    for it in range(self.iterations):
      length = calculate_length(self.G, points, separation=1.0)
      length.backward()
      with torch.no_grad():
        delta = (points[0] - self.step * points.grad[0]) - points[0]
        points -= delta.repeat(len(points), 1)
        points -= self.step * points.grad
        points[-1].copy_(self.end)
      points.grad.zero_()
      self.curves.append(points.detach().clone())
      self.lengths.append(length.detach().clone().item())

  def plot(self, delay: float = 1.0) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(-self.homotopy[0] * 1.5, self.homotopy[0] * 1.5)
    ax.set_ylim(-self.homotopy[1] * 1.5, self.homotopy[1] * 1.5)
    line, *_ = ax.plot([], [], marker='o')
    text = ax.text(
      0.02, 0.95, '',
      transform=ax.transAxes,
      va='top'
    )
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
  
    
def print_points(points: torch.Tensor) -> None:
  for p in points:
    print(f'({p[0]:.5f}, {p[1]:.5f})')

# Number of points
N = 3
# Homotopy class
cls = (3, 3)
# Metric
g11 = lambda x, y: torch.tensor(1.0)
g22 = lambda x, y: torch.tensor(1.0)
g12 = lambda x, y: torch.tensor(0.0)
G = [[g11, g12], [g12, g22]]

curve = Curve(N, cls, G)
curve.initialise(start=(0, 0), init_mode='random')
# with torch.no_grad():
#   curve.points[5].copy_(torch.tensor((-3.0, -2.0)))
curve.minimise(500, 0.01)
# for i in range(len(curve.curves) - 1):
#   print(np.all(np.array(curve.curves[i+1]) - np.array(curve.curves[i]) == np.zeros((N+1, 2))))
curve.plot(0.01)
# print(curve.points)
# curve.minimise(200, 0.01)
# print(curve.points)
# curve.minimise(50, 0.01)
# print("Points:")
# print_points(curve.curves[-1])
# print("Length:", curve.lengths[-1])
# curve.plot()
# for i in range(len(curve.curves)):
#   print("\nPoints:")
#   print_points(curve.curves[i])
#   print(f"Length: {curve.lengths[i]}")
# curve.plot()
exit(0)



# print("Initial points:")
# print_points(points)
# # ax.set_xlim(0, 4)
# # ax.set_ylim(0, 5)
# with torch.no_grad():
#   fig, ax = plt.subplots()
#   # X, Y = zip(*points.tolist())
#   # ax.plot(X, Y, marker='o')
#   line, = ax.plot(*zip(*points.tolist()), marker='o')
#   def update(frame):
#     shifted = points + torch.full(points.size(), frame)
#     line.set_data(*zip(*shifted.tolist()))
#     return line,
#   anim = FuncAnimation(fig, update, frames=iters, interval=1000, blit=True)
#   plt.show()
