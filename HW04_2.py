import numpy as np

f = lambda x, y: x ** 2 + y**4 - y**2
gf = lambda x, y: 2 * x + 4 * y ** 3 - 2 * y


def ex3(mu, sigma, x0, eps):
    x_noise = gradient_descent_noise(x0, eps)
    fs_noises = [f(x[0], x[1]) + 0.25 for x in x_noise]
    x, fs = gradient_descent(x0, eps)
    return x, x_noise, fs, fs_noises

def gradient_descent_noise(x0, eps):
    x, i = [(x0, 0)], 0
    t = 0.1
    x.append(x[0] - t * gf(x[0][0], x[0][1]))
    fs, gs = [f(x[0][0], x[0][1]), f(x[1][0], x[1][1])], [np.linalg.norm(gf(x[0][0], x[0][1])), np.linalg.norm(gf(x[1]))]
    while abs(f(x[i][0], x[i][1]) - f(x[i + 1][0], x[i+1][1])) > eps:
        i += 1
        gk = gf(x[i][0], x[i][0])
        beta = np.random.normal(0, 1, size=2)
        x.append(x[i] - t * gk + beta)
        fs.append(f(x[i + 1][0], x[i+1][1]))
        gs.append(np.linalg.norm(gk))
    return x

def gradient_descent(x0, eps):
    x, i = [(x0, 0)], 0
    t = 0.1
    x.append(x[0] - t * gf(x[0][0], x[0][1]))
    fs, gs = [f(x[0][0], x[0][1]), f(x[1][0], x[1][1])], [np.linalg.norm(gf(x[0][0], x[0][1])), np.linalg.norm(gf(x[1]))]
    while abs(f(x[i][0], x[i][1]) - f(x[i + 1][0], x[i+1][1])) > eps:
        i += 1
        gk = gf(x[i][0], x[i][0])
        x.append(x[i] - t * gk)
        fs.append(f(x[i + 1][0], x[i+1][1]))
        gs.append(np.linalg.norm(gk))
    return x, fs

[x, x_noise, fs, fs_noise] = ex3(0, 0.0005, (100,0), 10**-8)

