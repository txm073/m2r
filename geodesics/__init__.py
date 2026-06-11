import torch
from typing import Callable, NamedTuple
import sympy as sp

Vec2 = tuple[torch.Tensor | float, torch.Tensor | float]
ScalarFunc = Callable[[torch.Tensor | float, torch.Tensor | float], torch.Tensor | float]
Metric = list[list[ScalarFunc]]
SymbolicFunc = Callable[[sp.Symbol, sp.Symbol], sp.Symbol]
from .torus import Torus

from .solver import GeodesicSolver
from .plotter import Plotter