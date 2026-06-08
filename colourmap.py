import torch
import matplotlib.pyplot as plt

g11 = lambda x, y: torch.cos(x) ** 2 + torch.sin(y) ** 2 + 1.0
g12 = lambda x, y: torch.full(x.size(), 0)
G = [[g11, g12], [g12, g11]]

def heatmap(G, X, Y):
    D =  G[0][0](X, Y) * G[1][1](X, Y) - G[0][1](X, Y) * G[1][0](X, Y)
    return D

x = torch.linspace(-5, 5, 200)
y = torch.linspace(-5, 5, 200)
X, Y = torch.meshgrid(x, y, indexing='xy')
Z = heatmap(G, X, Y)

fig, ax = plt.subplots()

im = ax.imshow(
    Z,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    cmap='viridis'
)

fig.colorbar(im, ax=ax)
plt.show()