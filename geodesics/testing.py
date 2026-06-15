from dataclasses import dataclass, field
from . import Metric
from typing import Any, Callable
from .solver import GeodesicSolver, calculate_length
import traceback
import torch


@dataclass
class Parameters:
  N: int
  lr: float
  its: int
  start: tuple[float, float] = field(default=None)
  separation: float = field(default=None)
  init_mode: str = field(default='random')
  threshold: float = field(default=None)
  target: float = field(default=None)
  seed: float = field(default=None)

  def __dict__(self) -> dict[str, Any]:
    info = {
      'N': self.N,
      'learning_rate': self.lr,
      'iterations': self.its,
      'start': self.start,
      'separation': self.separation,
      'init_mode': self.init_mode,
      'threshold': self.threshold
    }
    return info


def minimise(
  cls: tuple[int, int], 
  G: Metric, 
  params_list: list[Parameters]
) -> list[dict]:
  data = []
  for params in params_list:
    try:
      if params.seed is not None:
        torch.manual_seed(params.seed)
      solver = GeodesicSolver(params.N, cls, G)
      solver.initialise(params.init_mode, params.start)
      solver.minimise(params.its, params.lr, verbose=True, store=True)
      idx = solver.lengths.index(min(solver.lengths))
      points = solver.curves[idx]
      length = solver.lengths[idx]
      item = params.__dict__()
      item.update({
        # 'points': points.detach().tolist(),
        'length': length,
        'required_iterations': idx
      })
      data.append(item)
    except Exception as e:
      traceback.print_exc(e)
  return data


def get_within(
  cls: tuple[int, int], 
  G: Metric, 
  params: Parameters
) -> dict:
  assert params.threshold is not None, 'provide a threshold'
  max_its = params.its
  it = -1
  length = None
  close_enough = False
  if params.seed is not None:
    torch.manual_seed(params.seed)
  solver = GeodesicSolver(params.N, cls, G)
  solver.initialise(params.init_mode, params.start)
  solver.minimise(0, params.lr, params.separation, run=False)
  points = solver.points.detach().clone()
  points.requires_grad_(True)
  prev_length = None
  while not close_enough and it < max_its:
    length, points = solver.backward(points)
    if prev_length is not None:
      if params.target is not None:
        diff = abs(length - params.target)
      else:
        diff = abs(prev_length - length)
      close_enough = diff < params.threshold
    prev_length = length  
    it += 1
  if not close_enough:
    raise Exception(
      'failed to compute length to the required accuracy threshold'
    )
  data = params.__dict__()
  data.update({
    # 'points': points.detach().tolist(),
    'length': length,
    'required_iterations': it
  })
  return data


def vary_param(
  cls: tuple[int, int],
  G: Metric,
  func: Callable[[tuple[int, int], Metric, Parameters], dict],
  params: Parameters, 
  param_name: str,
  values: list[Any]
) -> list[dict]:
  data = []
  assert hasattr(params, param_name), f'invalid parameter {param_name!r}'
  for value in values:
    try:
      setattr(params, param_name, value)
      data.append(func(cls, G, params))
    except Exception as e:
      print(f'{param_name}={value} resulted in the following error:\n{traceback.format_exc()}')
  return data
