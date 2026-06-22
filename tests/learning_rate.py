import sys
sys.path.extend(('.', '..'))
from geodesics.testing import *
from geodesics.utils import const
from pprint import pprint


if __name__ == '__main__':
  learning_rates = (0.05, 0.025, 0.01, 0.00001)
  params = Parameters(
    N=10, 
    lr=None,
    its=1000,
    start=(0.2, 0.2),
    init_mode='random',
    seed=5
  )
  homotopy = (0, 1)
  R = 3
  r = 1
  G = [
    [lambda x, y: (R + r * torch.cos(2 * torch.pi * x)), const(0.0)],
    [const(0.0), const(r ** 2)]
  ]
  data = []
  for lr in learning_rates:
    params.lr = lr
    data.extend(minimise(homotopy, G, [params]))
  for item in data:
    pprint(item)
