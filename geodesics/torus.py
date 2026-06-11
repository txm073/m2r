from . import Metric, ScalarFunc, SymbolicFunc
import sympy as sp
import torch
from dataclasses import dataclass
from .utils import _make_torch_func


@dataclass
class Torus:
  R: float
  r: SymbolicFunc | float

  @property
  def metric(self) -> Metric:
    theta, phi = sp.symbols('theta, phi')
    r = self.r if isinstance(self.r, float | int) else self.r(theta, phi).simplify()
    torus = sp.Matrix([
      (self.R + r * sp.cos(phi)) * sp.cos(theta),
      (self.R + r * sp.cos(phi)) * sp.sin(theta),
      r * sp.sin(phi)
    ])
    vars = sp.Matrix([theta, phi])
    J = torus.jacobian(vars)
    J.simplify()
    metric = J.transpose() * J
    make_fn = lambda i, j: _make_torch_func(str(
      metric[i, j].simplify() if hasattr(metric[i, j], 'simplify') else metric[i, j]
    ))
    torch_metric = [
      [make_fn(0, 0), make_fn(0, 1)],
      [make_fn(1, 0), make_fn(1, 1)]
    ]
    return torch_metric
  
  @property
  def r_torch(self) -> ScalarFunc:
    if isinstance(self.r, float | int):
      return lambda theta, phi: torch.full_like(theta, self.r)
    theta, phi = sp.symbols('theta, phi')
    return _make_torch_func(str(
      self.r(theta, phi).simplify() if hasattr(self.r, 'simplify') else self.r(theta, phi)
    ))