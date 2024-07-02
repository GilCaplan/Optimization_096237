import numpy as np
import matplotlib.pyplot as plt


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