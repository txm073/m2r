import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

surf = Callable[[float, float], float] # Function from R^2 to R

def example_numeric(
    f: surf, 
    x_range: tuple[float, float], 
    y_range: tuple[float, float], 
    p: tuple[int, int], 
    n: int = 25,
    N: int = 50,
    lr: float = 0.1,
    eps: float = 0.01,
    plot: bool = True
) -> list[tuple[float, float]]:
    x_ax = np.linspace(*x_range, n)
    y_ax = np.linspace(*y_range, n)
    x, y = np.meshgrid(x_ax, y_ax)
    z = f(x, y)
    dz_dy, dz_dx = np.gradient(z, y_ax, x_ax)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # Plot initial point
    x0, y0 = p
    z0 = f(x0, y0)
    ax.scatter(x0, y0, z0, color="magenta")
    # Plot surface
    ax.plot_surface(x, y, z, zorder=0)
    current_point = p
    path = [current_point]
    for i in range(N):
        idx_x = np.argmin(np.abs(x_ax - [current_point[0]]))
        idx_y = np.argmin(np.abs(y_ax - [current_point[1]]))
        grad = [dz_dx[idx_y, idx_x], dz_dy[idx_y, idx_x]]
        normal = np.array([-dz_dy[idx_y, idx_x], dz_dx[idx_y, idx_x], 1])
        normal /= np.linalg.norm(normal)
        if (np.linalg.norm(grad) < eps):
            break
        current_point = (
            current_point[0] - lr * grad[0],
            current_point[1] - lr * grad[1]
        )
        path.append(current_point)
        ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], s=5, color='red')
        plt.pause(0.1)
    if plot: plt.show()
    return path


def example_analytic(
    f: surface, 
    Df: list[Callable], 
    x_range: tuple[float, float], 
    y_range: tuple[float, float], 
    p: tuple[int, int], 
    n: int = 25,
    N: int = 50,
    lr: float = 0.1,
    eps: float = 0.01
) -> None:
    x_ax = np.linspace(*x_range, n)
    y_ax = np.linspace(*y_range, n)
    x, y = np.meshgrid(x_ax, y_ax)
    z = f(x, y)
    dz_dy, dz_dx = np.gradient(z, y_ax, x_ax)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # Plot initial point
    x0, y0 = p
    z0 = f(x0, y0)
    ax.scatter(x0, y0, z0, color="magenta")
    # Plot surface
    ax.plot_surface(x, y, z, zorder=0)
    current_point = p
    for i in range(N):
        grad = [Df[0](*current_point), Df[1](*current_point)]
        #if (np.linalg.norm(grad) < eps):
        #    break
        current_point = (
            current_point[0] - lr * grad[0],
            current_point[1] - lr * grad[1]
        )
        ax.plot(*current_point, f(*current_point), color='magenta')
        #plt.clf()
        plt.pause(0.1)
    plt.show()


  
# Function and derivative definition
f = lambda x, y: (x + 1) ** 2 * (x - 1) ** 2 + (y + 1) ** 2 * (y - 1) ** 2
#f = lambda x, y: x ** 2 + y ** 2 + x * (y + 1)
#Df = (lambda x, y: 2 * x + y + 1, lambda x, y: 2 * y + x)
#example_numeric(f, (-1, 1), (-1, 1), (-0.5, 0.5))
path = example_numeric(f, (-2, 2), (-2, 2), (-0.5, 0.5))
#print(path[-1], f(*path[-1]))

