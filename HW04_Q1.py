import numpy as np
import time
import matplotlib.pyplot as plt


# Q1
def generic_grad(f, gf, lsearch, x0, eps):
    start_time = time.time() * 1000
    x, i = [x0], 0
    x.append(x[0] - lsearch(f, x[0], gf(x0)))
    fs, gs = [f(x[0]), f(x[1])], [np.linalg.norm(gf(x[0])), np.linalg.norm(gf(x[1]))]
    ts = [0, (time.time()*1000 - start_time)]
    while abs(f(x[i]) - f(x[i+1])) > eps:
        i += 1
        gk = gf(x[i])
        t = lsearch(f, x[i], gk)
        x.append(x[i] - t * gk)
        fs.append(f(x[i+1]))
        gs.append(np.linalg.norm(gk))
        ts.append(time.time()*1000 - start_time)
    return x[i], ts, fs, gs


def const_step(f, xk, gk, s):
    return s


def exact_quad(f, xk, gk, A):
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise
    #return ((np.linalg.norm(gk) ** 2) / (2 * np.matmul(np.matmul(gk.T, A), gk)))
    t_values = np.linspace(0.01, 1, 100)
    return t_values[np.argmin([f(xk - t * gk) for t in t_values])]


def back(f, xk, gk, alpha, beta, s):
    t = s
    while f(xk - t * gk) >= f(xk) - alpha * t * (np.linalg.norm(gk) ** 2):
        t = beta * t
    return t


A = np.random.uniform(0, 1, (20, 5))
AA = np.matmul(A.T, A)
f = lambda x: np.linalg.norm(np.matmul(A, x)) ** 2
gf = lambda x: 2 * np.matmul(AA, x)
s = 1/(2 * max(np.linalg.eigvals(A.T @ A)))

eps = 10**-5
x0 = np.ones((5,1))

x1, ts1, fs1, gs1 = generic_grad(f, gf, lambda f, xk, gk: const_step(f, xk, gk, s), x0, eps)
x2, ts2, fs2, gs2 = generic_grad(f, gf, lambda f, xk, gk: back(f, xk, gk, 0.5, 0.5, 1), x0, eps)
x3, ts3, fs3, gs3 = generic_grad(f, gf, lambda f, xk, gk: exact_quad(f, xk, gk, AA), x0, eps)
# Create the graphs
plt.figure(figsize=(15, 5))

# f value in each iteration
plt.semilogy(fs1, label='constant step')
plt.semilogy(fs2, label='back')
plt.semilogy(fs3, label='exact_quad')
plt.title('f value in each iteration')
plt.legend()
plt.show()

# Norm of the gradient in each iteration
plt.semilogy(gs1, label='constant step')
plt.semilogy(gs2, label='back')
plt.semilogy(gs3, label='exact_quad')
plt.title('Norm of the gradient in each iteration')
plt.legend()
plt.show()

# Norm of the gradient in 3 methods as a function of time
plt.semilogy(ts1, gs1, label='constant step')
plt.semilogy(ts2, gs2, label='back')
plt.semilogy(ts3, gs3, label='exact_quad')
plt.title('Norm of the gradient in 3 methods as a function of time')
plt.legend()
plt.show()



# Q2

def f(x, y): return x ** 2 + y**4 - y**2
def gf(x, y): return (2 * x, 4 * y ** 3 - 2 * y)

mul = lambda x, t: (x[0] * t, x[1] * t) if type(x) is not int else x * t
min = lambda x, y: (x[0] - y[0], x[1] - y[1])
plus = lambda x, y: (x[0] + y[0], x[1] + y[1])


def ex3(mu, sigma, x0, eps):
    x, fs = gradient_descent(x0, eps)
    x_noise, fs_noise = gradient_descent(x0, eps, mu, sigma, 0.25, True)
    return x, x_noise, fs, fs_noise


def gradient_descent(x0, eps, mu=0, sigma=0, bias=0., r=False):
    x, i, t = [x0], 0, 0.1
    gk = gf(x[0][0], x[0][1])
    beta = np.random.normal(mu, sigma, size=2) if r else (0,0)
    x.append(plus(min(x[i], mul(gk, t)), beta))
    fs = [f(x[0][0], x[0][1])+0.25, f(x[1][0], x[1][1]) + bias]
    while abs(f(x[i][0], x[i][1]) - f(x[i + 1][0], x[i+1][1])) > eps:
        i += 1
        gk = gf(x[i][0], x[i][1])
        beta = np.random.normal(mu, sigma, size=2) if r else (0,0)
        x.append(plus(min(x[i], mul(gk, t)), beta))
        fs.append(f(x[i + 1][0], x[i+1][1]) + bias)
    return x, fs


[x, x_noise, fs, fs_noise] = ex3(0, 0.0005, (100,0), 10**-8)

iterations = range(len(fs))
iterations_noise = range(len(fs_noise))
plt.figure()
plt.semilogy(iterations, fs, label='fs')
plt.semilogy(iterations_noise, fs_noise, label='fs_noise')
plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title('Values of fs and fs_noise against the number of iterations')
plt.legend()
plt.show()
