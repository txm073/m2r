# M2R Project: Geodesics on a Torus

A Python package for computing and visualising geodesics on a parametrised torus embedded in $\mathbb{R}^3$.

The program uses PyTorch's automatic differentiation module `torch.autograd` and a gradient descent algorithm to compute a minimal-length curve with endpoints a fixed distance apart, corresponding to a non-trivial free homotopy class on the given torus.

## Overview

Given a Riemannian metric $G(x, y)$ and homotopy class $\left(m,n\right) \in \mathbb{Z}^2$, the program computes the minimal-length curve in the given homotopy class on the torus represented by the metric using a finite number of sample points. 

It can also produce animated plots:

- 2D plots displaying a colourmap representing the given Riemann metric and the points moving towards the minimising curve
- 3D plots displaying a given parameterised torus and the points moving on the surface towards the minimising curve

## Installation

```bash
git clone https://github.com/txm073/m2r.git
cd m2r
pip3 install -r requirements.txt
```

## Running

To run input a custom metric and homotopy class, simply run
```
python3 main.py
```
and follow the program instructions given

Alternatively, there are several examples given in the examples folder, to test them, run
```
python3 examples/##.py
```