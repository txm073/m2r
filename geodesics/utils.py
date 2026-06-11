import re
import torch
from .import ScalarFunc

def _prepend_functions(s: str, p: str) -> str:
  return re.sub(
    '[A-Za-z_][A-Za-z0-9_]*(?=\\()', 
    lambda m: f'{p}{m.group()}', 
    s
  )

def _make_torch_func(sp_func: str) -> ScalarFunc:
  prefix = 'lambda theta, phi:'
  const = 'theta' not in sp_func and 'phi' not in sp_func
  if const:
    val = eval(sp_func)
    return eval(f'{prefix} torch.full_like(theta, {val}, dtype=torch.float32)')
  return eval(f'{prefix} {_prepend_functions(sp_func, 'torch.')}')

