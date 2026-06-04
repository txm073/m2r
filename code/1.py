# Function and derivative definition
f = lambda x, y: (x ** 2 - 1) ** 2 + (y ** 2 - 1) ** 2
Df = (lambda x, y: 2 * x * (x ** 2 - 1), lambda x, y: 2 * y * (y ** 2 - 1))

k_norm = lambda k: (lambda p, q: (sum(abs(p[i] - q[i]) ** k for i in range(len(p)))) ** (1/k))
# Euclidean metric on R^n
metric = k_norm(2)

# Parameters
lr = 0.01
eps = 0.001
max_iterations = 10000

# Known true minimum value
target = (1, -1) 
# Starting point
p = (5, -7)
iterations = -1
close_enough = False
try:
    while not close_enough and iterations < max_iterations:
        p = (p[0] - lr * Df[0](*p), p[1] - lr * Df[1](*p))
        close_enough = metric(p, target) < eps
        iterations += 1
except OverflowError:
    pass
if close_enough:
    print(f"{p} is within {eps} of target point {target} after {iterations} iterations")
else:
    print(f"Failed to get sufficiently close to target point {target} after {max_iterations} iterations")
