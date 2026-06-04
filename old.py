import random
import torch
import numpy as np
from typing import Iterable, Callable
import matplotlib.pyplot as plt

Vec2 = tuple[torch.Tensor, torch.Tensor]
Metric = list[list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]

# Number of points
N = 3
# Homotopy class
cls = torch.tensor((3, 3), dtype=torch.int)
# Pick random starting point
# start = np.array([random.uniform(0, 1), random.uniform(0, 1)])
start = torch.tensor((0, 0), dtype=torch.float)
end = start + cls
# Initialise points as a straight line
points = torch.tensor(np.linspace(start.numpy(), end.numpy(), N + 1), requires_grad=True)
# print(points)

g11 = lambda x, y: torch.tensor(1.0, requires_grad=True)
g22 = lambda x, y: torch.tensor(1.0, requires_grad=True)
g12 = lambda x, y: torch.tensor(0.0, requires_grad=True)
G = [[g11, g12], [g12, g22]]

def innerprod(G: Metric, u: Vec2, v: Vec2, w: Vec2) -> torch.Tensor:
  return G[0][0](*w) * u[0] * v[0] \
    + G[0][1](*w) * (u[0] * v[1] + u[1] * v[0]) \
    + G[1][1](*w) * u[1] * v[1]

def length(G: Metric, points: torch.Tensor) -> torch.Tensor:
  l = 0.0
  for i in range(len(points) - 1):
    u = points[i+1] - points[i]
    l += torch.sqrt(innerprod(G, u, u, points[i]))
  return l

def print_points(points: torch.Tensor) -> None:
  for p in points:
    print(f"({p[0]:.5f},{p[1]:.5f})")

lr = 0.01
iters = 5

print("Initial points:")
print_points(points)
# fig, ax = plt.subplots()
# # ax.set_xlim(0, 4)
# # ax.set_ylim(0, 5)
# shifted = points + torch.full(points.size(), 1)
# with torch.no_grad():
#   # X, Y = zip(*points.tolist())
#   # ax.plot(X, Y, marker='o')
#   ax.plot(*zip(*points.tolist()), marker='o')
#   # for i in range(len(points) - 1):
#   #   ax.plot(
#   #     (points[i][0].item(), points[i][1].item()), 
#   #     (points[i+1][0].item(), points[i+1][1].item()),
#   #     marker='o'
#   #   )
# plt.show()
# exit(0)
curves = [points.detach().tolist()]
lengths = [length(G, points)]

for it in range(iters):
  l = length(G, points)
  l.backward()
  # print(f"\nIteration {it+1}:")
  # print(f"Length: {l.item()}")
  with torch.no_grad():
    delta = (points[0] - lr * points.grad[0]) - points[0]
    points -= lr * points.grad
    # print("Updated points:")
    # print_points(points)
    points -= delta.repeat(len(points), 1)
    points[-1].copy_(end)
    # print("Translated points:")
    # print_points(points)
    # print(f"Closed: {points[-1] - points[0]}")
  points.grad.zero_()
  curves.append(points.detach().tolist())
  lengths.append(l.item())
# print(points)
# print(l)
# print(l.is_leaf)

for points, l in zip(curves, lengths):
  print("Points:")
  print_points(points)
  print(f"Length: {l}")

# print(length(G, points))