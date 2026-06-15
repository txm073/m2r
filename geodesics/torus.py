from . import Metric, ScalarFunc, SymbolicFunc
import sympy as sp
from sympy.calculus.util import periodicity
import torch
from dataclasses import dataclass, field
from .utils import _make_torch_func


@dataclass
class Torus:
  R: float
  r: SymbolicFunc | float
  theta: sp.Symbol = field(init=False)
  phi: sp.Symbol = field(init=False)
  theta_period: float = field(init=False)
  phi_period: float = field(init=False)
  _r: ScalarFunc = field(init=False, default=None)

  def __post_init__(self) -> None:
    self.theta = sp.Symbol('theta')
    self.phi = sp.Symbol('phi')
    if isinstance(self.r, int | float):
      self.theta_period = 1
      self.phi_period = 1
    else:
      self.theta_period = periodicity(self.r(self.theta, self.phi), self.theta)
      self.phi_period = periodicity(self.r(self.theta, self.phi), self.phi)
      assert self.theta_period is not None, 'r is not periodic w.r.t. theta'
      assert self.phi_period is not None, 'r is not periodic w.r.t. phi'
      self.theta_period = float(self.theta_period.evalf())
      self.phi_period = float(self.phi_period.evalf())
    if self.theta_period and not self.phi_period: 
      self.phi_period = self.theta_period
    elif self.phi_period and not self.theta_period:
      self.theta_period = self.phi_period

  @property
  def metric(self) -> Metric:
    a, b = 2 * torch.pi / self.theta_period, 2 * torch.pi / self.phi_period
    # print(f'scaling factors: {a=}, {b=}')
    r = self.r if isinstance(self.r, float | int) else self.r(self.theta, self.phi).simplify()
    torus = sp.Matrix([
      (self.R + r * sp.cos(self.phi * b)) * sp.cos(self.theta * a),
      (self.R + r * sp.cos(self.phi * b)) * sp.sin(self.theta * a),
      r * sp.sin(self.phi * b)
    ])
    vars = sp.Matrix([self.theta, self.phi])
    J = torus.jacobian(vars)
    J.simplify()
    metric = J.transpose() * J
    # tol = 1e-10
    # make_fn = lambda i, j: _make_torch_func(str(
    #   sp.nsimplify(sp.trigsimp(sp.simplify(metric[i, j])), tolerance=tol).evalf(chop=tol)
    # ))
    make_fn = lambda i, j: _make_torch_func(metric[i, j])
    torch_metric = [
      [make_fn(0, 0), make_fn(0, 1)],
      [make_fn(1, 0), make_fn(1, 1)]
    ]
    return torch_metric
  
  @property
  def r_torch(self) -> ScalarFunc:
    if self._r is not None:
      return self._r
    if isinstance(self.r, float | int):
      return lambda theta, phi: torch.full_like(theta, self.r)
    # self._r = _make_torch_func(str(
    #   self.r(self.theta, self.phi).simplify() if hasattr(self.r, 'simplify') else self.r(self.theta, self.phi)
    # ))
    self._r = _make_torch_func(self.r(self.theta, self.phi))
    return self._r
  
  def __call__(
    self, 
    theta: torch.Tensor, 
    phi: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = 2 * torch.pi / self.theta_period, 2 * torch.pi / self.phi_period
    # print(f'scaling factors: {a=}, {b=}')
    return (
      (self.R + self.r_torch(theta, phi) * torch.cos(phi * b)) * torch.cos(theta * a),
      (self.R + self.r_torch(theta, phi) * torch.cos(phi * b)) * torch.sin(theta * a),
      self.r_torch(theta, phi) * torch.sin(phi * b)
    )
