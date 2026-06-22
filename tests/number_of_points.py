import sys
sys.path.extend(('.', '..'))
from geodesics.testing import *
from geodesics.utils import const
from pprint import pprint


if __name__ == '__main__':
  n_points = (5, 10, 15)
  params = Parameters(
    N=None, 
    lr=0.01,
    its=2000,
    start=(0.1, 0.1),
    init_mode='random',
    seed=5,
    threshold=0.05,
    target=1
  )
  homotopy = (0, 1)
  R = 3
  r = 1
  G = [
    [lambda x, y: (R + r * torch.cos(2 * torch.pi * x)), const(0.0)],
    [const(0.0), const(r ** 2)]
  ]
  data = []
  pprint(vary_param(homotopy, G, get_within, params, 'N', n_points))