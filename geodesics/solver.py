import pickle
from dataclasses import dataclass
import numpy as np
from . import *

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
class GeodesicSolver:
  N: int
  homotopy: tuple[int, int]
  G: Metric
  periods: tuple[float, float] = (1.0, 1.0)

  def initialise(
    self,
    init_mode: str = 'random', 
    start: tuple[float, float] = None
  ) -> None:
    if start is None:
      self.start = torch.rand(2, dtype=torch.float) * torch.tensor(self.periods)
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

  def backward(
    self,
    points: list[torch.Tensor]
  ) -> tuple[float, list[torch.Tensor]]:
    loss = calculate_length(self.G, points, separation=self.separation)
    loss.backward()
    with torch.no_grad():
      length = calculate_length(self.G, points).detach().clone().item()
      shift = points[0][0] >= self.periods[0] or \
              points[0][1] >= self.periods[1] or \
              points[0][0] < 0.0 or \
              points[0][1] < 0.0
      if shift:
        delta = -self.step * points.grad[0]
      else:
        delta = torch.tensor(0.0)
      points -= delta.repeat(len(points), 1)
      points -= self.step * points.grad
      points[-1].copy_(points[0] + torch.tensor(self.homotopy))
    points.grad.zero_()
    return length, points

  def minimise(
    self, 
    iterations: int, 
    step: float,
    separation: float = None,
    verbose: bool = False,
    store: bool = True,
    run: bool = True
  ) -> None:
    self.iterations = iterations
    self.step = step
    self.separation = separation
    if self.separation is None:
      self.separation = self.N ** 2
    initial_length = calculate_length(self.G, self.points)
    self.curves = [self.points.detach()]
    self.lengths = [initial_length.detach().item()]
    if not run:
      return
    points = self.points.detach().clone()
    points.requires_grad_(True)
    stage = max(self.iterations // 10, 1)
    for it in range(self.iterations):
      if verbose and it % stage == 0:
        print(f'Computing {it}/{self.iterations}...')
      length, points = self.backward(points)
      if store:
        self.curves.append(points.detach().clone())
        self.lengths.append(length)

  def save(self, path: str) -> None:
    with open(path, 'wb') as f:
      data = {
        'curves': self.curves,
        'lengths': self.lengths,
        'seed': torch.initial_seed()
      }
      pickle.dump(data, f)

  def load(self, path: str) -> None:
    with open(path, 'rb') as f:
      data = pickle.load(f)
    self.curves = data['curves']
    self.lengths = data['lengths']