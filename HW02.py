import copy

import matplotlib.pyplot as plt
import numpy as np


def generic_bisect(f, df, l, u, eps, k):
    fv = [f(u)]
    iter = 0
    while abs(u - l) > eps and iter < k:
        m = (l + u) / 2
        if df(m) * df(l) < 0:
            u = m
        else:
            l = m
        fv.append(f(m))
        iter += 1
    return m, fv


def generic_newton(f, df, ddf, x0, k):
    fv = [f(x0)]
    iter = 0
    while iter < k:
        x0 = x0 - (df(x0) / ddf(x0))
        fv.append(f(x0))
        iter += 1
    return x0, fv


def generic_hybrid(f, df, ddf, l, u, x0, eps, k):
    fv = [f(x0)]
    iter = 0
    x = x0
    while iter < k and abs(u - l) > eps and abs(df(x)) > eps:
        x = x0 - df(x0) / ddf(x0)
        if (x < l or x > u) or abs(df(x)) >= 0.99 * abs(df(x0)):
            x = (l + u) / 2
        x0 = x

        if df(x) * df(l) < 0:
            u = x
        else:
            l = x

        fv.append(f(x))
        iter += 1
    return x, fv


def generic_gs(f, l, u, eps, k):
    tau = (3 - 5 ** 0.5) / 2
    fv = [f(u)]
    iter = 0
    x2 = l + (u - l) * tau
    x3 = l + (u - l) * (1 - tau)

    while abs(u - l) > eps and iter < k:
        if f(x2) < f(x3):
            u = x3
            x3 = x2
            x2 = l + (u - l) * tau
        else:
            l = x2
            x2 = x3
            x3 = l + (u - l) * (1 - tau)
        fv.append(f(u))
        iter += 1
    return l, fv



def generic_gs_ex4(f, l, u, mu, a, b, c, eps):
    tau = (3 - 5 ** 0.5) / 2
    iter = 0
    x2 = l + (u - l) * tau
    x3 = l + (u - l) * (1 - tau)

    while abs(u - l) > eps:
        if f(x2, mu, a, b, c) < f(x3, mu, a, b, c):
            u = x3
            x3 = x2
            x2 = l + (u - l) * tau
        else:
            l = x2
            x2 = x3
            x3 = l + (u - l) * (1 - tau)
        iter += 1
    return l


def phi(t, mu, a, b, c):
    return mu * (t - a) ** 2 + abs(t-b) + abs(t-c)


def gs_denoise_step(mu, a, b, c):
    if mu < 0:
        return
    return (generic_gs_ex4(phi, max(a, b, c) + 1, min(a, b, c) - 1, mu, a, b, c, 10e-10))[0]


def gs_denoise(s, alpha, N):
    x = copy.deepcopy(s)
    for i in range(N+1):
        for k in range(len(s)):
            if k == 0:
                x[k] = gs_denoise_step(2*alpha, s[0], x[1], x[1])
            elif k == len(s) - 1:
                x[k] = gs_denoise_step(2*alpha, s[k], x[k - 1], x[k - 1])
            else:
                x[k] = gs_denoise_step(alpha, s[k], x[k - 1], x[k + 1])
    return x


k = 50
eps = 1e-5

p = lambda x: -3.55 * x ** 3 + 1.1 * x ** 2 + 0.765 * x - 0.74
dp = lambda x: -10.65 * x ** 2 + 2.2 * x + 0.765

x, fv = generic_hybrid(p, p, dp, -1, 1, 0, eps, k)

print(f'Root: {x}')

t = gs_denoise_step(1, -10, 10, k)
print(t)

# plotting the real discrete signal
real_s_1 = [1.]*40
real_s_0 = [0.]*40

plt.plot(range(40), real_s_1, 'black', linewidth=0.7)
plt.plot(range(41, 81), real_s_0, 'black', linewidth=0.7)


# solving the problem
s = np.array([[1.]*40 + [0.]*40]).T + 0.1*np.random.randn(80, 1) # noised signal
x1 = gs_denoise(s, 0.5, 10)
x2 = gs_denoise(s, 0.5, 20)
x3 = gs_denoise(s, 0.5, 30)

plt.plot(range(80), s, 'cyan', linewidth=0.7)
plt.plot(range(80), x1, 'red', linewidth=0.7)
plt.plot(range(80), x2, 'green', linewidth=0.7)
plt.plot(range(80), x3, 'blue', linewidth=0.7)

plt.show()




