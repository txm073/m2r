import math
import sys
sys.path.extend(('.', '..'))
from geodesics.testing import *
from geodesics.utils import const
from pprint import pprint
import matplotlib.pyplot as plt


if __name__ == '__main__':
  # params = Parameters(
  #   N=10, 
  #   lr=None,
  #   its=1000,
  #   start=(0.2, 0.2),
  #   init_mode='random',
  #   seed=5
  # )
  params = [Parameters(10, 0.5, it, start=(0.1, 0.1)) for it in (4, 8, 12, 16, 20)]
  homotopy = (0, 1)
  R = 3
  r = 1
  G = [
    [lambda x, y: (R + r * torch.cos(2 * torch.pi * x)), const(0.0)],
    [const(0.0), const(r ** 2)]
  ]
  data = minimise(homotopy, G, params)
  # data = []
  # for lr in learning_rates:
  #   params.lr = lr
  #   data.extend(minimise(homotopy, G, [params]))
  fig, ax = plt.subplots(figsize=(9, 9))
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Length (order of magnitude)')
  bars = ax.bar([str(item['iterations']) for item in data], [math.log(item['final_length'], 10) for item in data])
  plt.show()
  # for item in data:
  #   pprint(item)
