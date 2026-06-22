import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from scipy.interpolate import RegularGridInterpolator

@dataclass
class Metric:
    mat: list[list[Callable[[float, float], float]]]

    def __call__(self, x: float, y: float) -> list[list[float]]:
        return [[self.mat[i][j](x, y) for j in range(len(self.mat[i]))] for i in range(len(self.mat))]

g11 = lambda x, y: np.sin(x)
g12 = lambda x, y: np.cos(y)
g22 = lambda x, y: np.sin(y)
vec2 = tuple[float, float]

def prod(u: vec2, v: vec2, z: vec2) -> float:
    return g11(*z)*u[0]*v[0]+g12(*z)*(u[0]*v[1]+u[1]*v[0])+g22(*z)*u[1]*v[1]

def length(points: list[vec2]) -> float:
    l = 0
    for i in range(len(points) - 1):
        l += np.sqrt(prod(points[i+1] - points[i], points[i+1] - points[i], points[i]))
    return l 

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
z = np.linspace(-2, 2, 20)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Unit sphere
F = X**2 + Y**2 + Z**2 - 1 

# Simple torus
r, R = 1, 2
# F = (R ** 2 - r ** 2 + x ** 2 + y ** 2 + z ** 2) ** 2 - 4 * R ** 2 * (x ** 2 + y ** 2)

theta_range = np.linspace(0, 2 * np.pi, 40)
phi_range = np.linspace(0, 2 * np.pi, 40)
theta, phi = np.meshgrid(theta_range, phi_range)
r = np.cos(theta) ** 2 + 0.5
#dF_dtheta, dF_dphi = np.gradient()
x_param = (R + r * np.cos(phi)) * np.cos(theta)
y_param = (R + r * np.cos(phi)) * np.sin(theta)
z_param = r * np.sin(phi)

#dF_dx, dF_dy, dF_dz = np.gradient(F, x, y, z)

fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
axs[0, 0].set_zlim(-4, 4)
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_zlabel('Z')
axs[0, 0].plot_surface(x_param, y_param, z_param, antialiased=False)
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].plot()
plt.tight_layout()
plt.show()
#plt.plot()
