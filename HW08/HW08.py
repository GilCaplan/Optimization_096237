import cvxpy as cp
import numpy as np
x = cp.Variable(3)
b = 1
A = np.array([[1, 1 / 2, 0], [1, 0, 0], [0, 15 ** (1 / 2) / 2, 0], [0, 0, 0]])
D = np.array([1, -1, 1])
E = np.array([1, 1, 0])
y = np.array([0,0,0,2])
objective = cp.Minimize(cp.square(x[0] + x[1]) + cp.square(x[1]) + cp.square(x[2]) + 3 * x[0] - 4 * x[1])
constraints = [cp.norm(A @ x + y) + cp.quad_over_lin(D @ x + b, E @ x) <= 6, x >= 1]
result = cp.Problem(objective, constraints).solve()
print(f"Optimal value: {result}")
print(f"Optimal var: {x.value}")


def grad_proj(f, gf, proj, t, x0, eps):
    x_prev = x0
    xk = proj(x_prev - t * gf(x_prev))
    fs = [f(xk)]
    while np.linalg.norm(xk - x_prev) > eps:
        x_prev = xk
        xk = proj(x_prev - t * gf(x_prev))
        fs.append(f(xk))
    return xk, fs

def proj_section_b():
    return lambda p: np.array([(p[0]-p[1]+1)/2, (p[1]-p[0]+1)/2])

f = lambda x: np.linalg.norm(x)**2
gf = lambda x: 2*x
x, fs = grad_proj(f, gf, proj_section_b(), 0.5, np.array([100, 100]), 10**-8)
print(f"convered after: {len(fs)} iterations with point {x}")