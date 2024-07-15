import time
import matplotlib.pyplot as plt
import numpy as np

def check_pos_def(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
def hybrid_newton(f, gf, hf, lsearch, x0, eps):
    start = time.time() * 1000
    xs = [x0]
    fs,gs,ts, newton = [],[gf(x0[0], x0[0])],[hf(x0[0], x0[1])],[0]
    x, y = x0
    while np.linalg.norm(gf(x, y)) > eps:
        if check_pos_def(hf(x,y)):
            dk = [np.linalg.solve(hf(x, y), -gf(x, y)), 'newton']
            newton.append(1)
        else:
            dk = [-gf(x, y), 'grad']
            newton.append(0)
        tk = lsearch(x, y, gf(x, y), dk)
        xs.append(np.array([x + tk*dk[0][0],y + tk*dk[0][1]]))
        x, y = xs[-1]
        fs.append(f(x,y))
        gs.append(gf(x,y))
        ts.append(time.time() * 1000 - start)
    return x,fs,gs,ts,newton

def generic_grad(f, gf, lsearch, x0, eps):
    start_time = time.time() * 1000
    x, i, t = [x0], 0, lsearch(x0[0], x0[1], gf(x0[0], x0[1]), [gf(x0[0], x0[1]), 'grad'])
    x.append(x - t * gf(x0[0], x0[1]))
    xs, ys = x0
    xss, yss = x[-1][0]
    fs, gs = [f(xs, ys), f(xss, yss)], [np.linalg.norm(gf(xs, ys)), np.linalg.norm(gf(xss, yss))]
    ts = [0, (time.time()*1000 - start_time)]
    while np.linalg.norm(gf(xs, ys)) > eps and i < 100:
        i += 1
        gk = gf(xss, yss)
        t = lsearch(xss, yss, gk, [gk, 'grad'])
        xs, ys = x[-1][0]
        x.append([xs - t * gk[0], ys - t * gk[1]])
        fs.append(f(xss, yss))
        gs.append(np.linalg.norm(gf(xss, yss)))
        ts.append(time.time()*1000 - start_time)
    return x, fs, gs, ts

def hybrid_back(f, alpha, beta, s):
    return lambda x,y, gk, direction:hybrid_back_check(f, alpha, beta, s,x, y, gk, direction)

def hybrid_back_check(f, alpha, beta, s, x, y, gv, direction):
    if direction[1] == 'newton':
        return 1
    t = s
    while f(x-t*gv[0], y-t*gv[1]) >= f(x, y) - alpha*t*np.linalg.norm(gv)**2:
        t = beta*t
    return t

def plot(fs, newton, fs_grad, title):
    # y_cord_g = [i + 162 for i in fs_grad]
    x_cord_g = [i for i in range(len(fs_grad))]
    # plt.semilogy(x_cord_g, y_cord_g, label="generic grad", color="green")
    ycord = [i + 162 for i in fs]
    xcord = [i for i in range(len(fs))]
    ynewton, ygrad, xcord_newton, xcord_grad = [], [], [], []
    for i in range(len(fs)):
        if newton[i] == 0:
            ygrad.append(ycord[i])
            xcord_grad.append(xcord[i])
        else:
            ynewton.append(ycord[i])
            xcord_newton.append(xcord[i])
    plt.semilogy(xcord_newton, ynewton, label="newton", color="blue")
    # plt.semilogy(xcord_grad, ygrad, label="grad", color="red")
    plt.title(title)
    plt.legend()
    plt.show()


alpha, beta, s = 0.25, 0.5, 1

# Define the functions
f = lambda x, y: x**4 + y**4 - 36 * x * y
gf = lambda x, y: np.array([4 * x**3 - 36 * y, 4 * y**3 - 36 * x])
hf = lambda x, y: np.array([[12 * x**2, -36], [-36, 12 * y**2]])
x0 = np.array((200, 0)).T
eps = 10**-6

x, fs, gs, ts, newton = hybrid_newton(f, gf, hf, hybrid_back(f, 1/4, 1/2, 1), np.array([200, 0]), 10**(-6))
# xg, fsg, gsg, tsg = generic_grad(f, gf, hybrid_back(f, 1/4, 1/2, 1), np.array([200, 0]), 10**(-6))
plot(fs, newton,[], "hybrid newton vs generic grad")


