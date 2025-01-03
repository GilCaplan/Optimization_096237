import time

import numpy as np
import scipy as scipy
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, kron
from matplotlib import pyplot as plt


def blur(N, band=3, sigma=0.7):
    z = np.block([np.exp(-(np.array([range(band)]) ** 2) / (2 * sigma ** 2)), np.zeros((1, N - band))])
    A = toeplitz(z)
    A = csr_matrix(A)
    A = (1 / (2 * 3.1415 * sigma ** 2)) * kron(A, A)

    x = np.zeros((N, N))
    N2 = round(N / 2)
    N3 = round(N / 3)
    N6 = round(N / 6)
    N12 = round(N / 12)

    # Large elipse
    T = np.zeros((N6, N3))
    for i in range(1, N6 + 1):
        for j in range(1, N3 + 1):
            if (i / N6) ** 2 + (j / N3) ** 2 < 1:
                T[i - 1, j - 1] = 1

    T = np.block([np.fliplr(T), T])
    T = np.block([[np.flipud(T)], [T]])
    x[2:2 + 2 * N6, N3 - 1:3 * N3 - 1] = T

    # Small elipse
    T = np.zeros((N6, N3))
    for i in range(1, N6 + 1):
        for j in range(1, N3 + 1):
            if (i / N6) ** 2 + (j / N3) ** 2 < 0.6:
                T[i - 1, j - 1] = 1

    T = np.block([np.fliplr(T), T])
    T = np.block([[np.flipud(T)], [T]])
    x[N6:3 * N6, N3 - 1:3 * N3 - 1] = x[N6:3 * N6, N3 - 1:3 * N3 - 1] + 2 * T
    x[x == 3] = 2 * np.ones((x[x == 3]).shape)

    T = np.triu(np.ones((N3, N3)))
    mT, nT = T.shape
    x[N3 + N12:N3 + N12 + nT, 1:mT + 1] = 3 * T

    T = np.zeros((2 * N6 + 1, 2 * N6 + 1))
    mT, nT = T.shape
    T[N6, :] = np.ones((1, nT))
    T[:, N6] = np.ones((mT))
    x[N2 + N12:N2 + N12 + mT, N2:N2 + nT] = 4 * T

    x = x[:N, :N].reshape(N ** 2, 1)
    b = A @ x

    return A, b, x


def plot(i, x):
    plt.figure(figsize=(6,6))
    plt.imshow(x.reshape(256,256), cmap='gray')
    plt.title(f"exact quad where {i} iterations")
    plt.show()


def generic_grad(f, gf, x0):
    start_time = time.time() * 1000
    x, i = [x0], 0
    ts = [0, (time.time() * 1000 - start_time)]
    x.append(x[0] - exact_quad(f, x[0], gf(x0), AA))
    fs, gs = [f(x[0]), f(x[1])], [np.linalg.norm(gf(x[0])), np.linalg.norm(gf(x[1]))]
    for i in range(1001):
        gk = gf(x[i])
        t = exact_quad(f, x[i], gk, AA)
        x.append(x[i] - t * gk)
        fs.append(f(x[i+1]))
        gs.append(np.linalg.norm(gk))
        ts.append(time.time() * 1000 - start_time)
        if(i in [1,10, 100, 1000]):
            plot(i, x[len(x)-1])


def generic_grad_b(f, gf, x0, t):
    start_time = time.time() * 1000
    x, i = [x0], 0
    ts = [0, (time.time() * 1000 - start_time)]
    x.append(x[0] - t)
    fs, gs = [f(x[0]), f(x[1])], [np.linalg.norm(gf(x[0])), np.linalg.norm(gf(x[1]))]
    for i in range(1001):
        gk = gf(x[i])
        x.append(x[i] - t * gk)
        fs.append(f(x[i+1]))
        gs.append(np.linalg.norm(gk))
        ts.append(time.time() * 1000 - start_time)
        if(i in [1,10, 100, 1000]):
            plot(i, x[len(x)-1])


def exact_quad(f, xk, gk, A):
    return (np.linalg.norm(gk) ** 2) / (((2 * np.linalg.norm(A.dot(gk)))**2))


#A = np.array([[1,2,1,1,1],[1,2,1,1,1],[2,2,2,2,2], [1,2,1,1,1],[1,2,1,1,1]])
A, b, x = blur(256, 5, 1)
AA = A.T.dot(A)
f = lambda x: np.linalg.norm(A.dot(x) - b) ** 2
gf = lambda x: 2 * AA.dot(x) - 2 * A.T.dot(b)
#l = 2 * max(np.real(scipy.sparse.linalg.eigs(AA)[0]))
l = 1.9994981175712248
x0 = np.zeros(np.shape(b))
#ls = generic_grad(f, gf, x0)

#lsb = generic_grad_b(f, gf, x0, 1/l)

def AG(f, gf, L, x0, max_iter):
    bstart_time = time.time() * 1000
    y, t = [0,x0], [0, 1]
    x = [x0, x0]
    gs = [np.linalg.norm(gf(y[0]))]
    fs = [f(y[0])]
    ts = [0, (time.time() * 1000 - bstart_time)]
    for k in range(1, max_iter):
        x.append(y[k] - [i/L for i in gf(y[k])])
        t.append((1 + np.sqrt(1 + 4 * t[k] ** 2)) / 2)
        y.append(x[k] + ((t[k] - 1) / t[k + 1]) * (x[k] - x[k - 1]))
        gs.append(np.linalg.norm(gf(y[k])))
        fs.append(f(y[k]))
        ts.append(time.time() * 1000 - bstart_time)
        if k in [1,10, 100, 1000]:
            plot(k, x[len(x) - 1])
            plt.figure()
            plt.semilogy(range(len(fs)), fs, label='fs')
            plt.semilogy(range(len(ts)), ts, label='ts')
            plt.xlabel('Iterations')
            plt.ylabel('Values')
            plt.title('Values of fs and fs_noise against the number of iterations')
            plt.legend()
            plt.show()
    return x, fs, gs, ts

start_time = time.time() * 1000

x ,fs ,gs ,ts = AG(f , gf , l, x0, max_iter=1000)

plt.figure()
plt.semilogy(range(len(fs)), fs, label='fs')
plt.semilogy(range(len(ts)), ts, label='ts')
plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title('Values of fs and fs_noise against the number of iterations')
plt.legend()
plt.show()
print(time.time()* 1000 - start_time)