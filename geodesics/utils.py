import re
import torch
import sympy as sp
from .import ScalarFunc

def _prepend_functions(s: str, p: str) -> str:
  return re.sub(
    '[A-Za-z_][A-Za-z0-9_]*(?=\\()', 
    lambda m: f'{p}{m.group()}', 
    s
  )

def _make_torch_func(sp_func: sp.Expr) -> ScalarFunc:
  tol = 1e-10
  simplified = sp.nsimplify(
    sp.trigsimp(sp.simplify(sp_func)), 
    tolerance=tol
  ).evalf(chop=tol)
  expr = str(simplified)
  print(f'torch func: {expr}')
  prefix = 'lambda theta, phi:'
  const = 'theta' not in expr and 'phi' not in expr
  if const:
    val = eval(expr)
    return eval(f'{prefix} torch.full_like(theta, {val}, dtype=torch.float32)')
  return eval(f'{prefix} {_prepend_functions(expr, 'torch.')}')

def const(value: float) -> ScalarFunc:
  return lambda theta, phi: torch.full_like(theta, value)

def input_function(prompt: str, prefix: str) -> ScalarFunc:
  s = input(prompt).replace(' ', '').lower()
  s = re.sub('[A-Za-z_][A-Za-z0-9_]*(?=\\()', lambda m: f'{prefix}.{m.group()}', s)
  try:
    v = float(s)
    return const(v)
  except ValueError:
    return eval(f'lambda x, y: {s}')
