import sys
sys.path.extend(('.', '..'))
from geodesics.testing import *
from geodesics.utils import const
from pprint import pprint


if __name__ == '__main__':
  learning_rates = (0.025, 0.01, 0.005, 0.0025, 0.0001)
  params = Parameters(
    N=10, 
    lr=learning_rates[-1],
    its=5000, 
    init_mode='random',
    threshold=1e-4,
    # target=2**0.5,
    seed=0
  )
  # cls = (1, 10)
  # G = [[const(1.0), const(0.0)], [const(0.0), const(1.0)]]

  homotopy = (1, 2)
  R = 3
  r = 1
  G = [
    [lambda x, y: (R + r * torch.cos(2 * torch.pi * x)), const(0.0)],
    [const(0.0), const(r ** 2)]
  ]
  data = vary_param(homotopy, G, get_within, params, 'lr', learning_rates)
  for item in data:
    pprint(item)
