import numpy as np
import matplotlib.pyplot as plt

def ex1(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1] == x.shape[0]

    n = A.shape[0]
    B = A.T + np.outer(np.array(list(range(1, n + 1))), x.T)
    B = B - np.diag(np.diag(B))

    return B


def build_P(A, B, n):
    m = A.shape[0]
    P = np.zeros((n * m, n * m))

    for i in range(n):
        for j in range(n):
            if i == j:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = A
            elif j == i + 1:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = B.T
            elif i == j + 1:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = B
    return P


def ex2(A, B, n, b):
    """doesn't meet requirements"""
    if n < 4 or A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1] or B.shape[0] != B.shape[1] or A.shape[0] != A.shape[1]:
        return

    """ A, B \in R^(mxm), n >= 4 \in N, b \in R^m """
    y = np.concatenate([i * b for i in range(1, n+1)])
    P = build_P(A, B, n)
    Q = np.kron(A, P)
    z = np.concatenate([y for _ in range(len(A))])
    sol = np.linalg.solve(Q, z)
    return sol


def fit_rational(X: np.ndarray):
    m = X.shape[1]

    if X.shape[0] != 2:
        raise ValueError("incorrect dimensions")
    ones = np.ones(m)

    x = X[0]
    y = X[1]

    A = np.column_stack((ones, x, x ** 2, -x * y, - (x ** 2) * y))
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    coefficents = np.concatenate((sol[:3], [1], sol[3:]))
    return coefficents

def fit_rational_normed(X: np.ndarray):
    x = X[0]
    y = X[1]

    if X.shape[0] != 2 or x.shape != y.shape:
        raise ValueError("incorrect dimensions")
    A = np.array([[-1, -xi, -xi ** 2, yi, yi * xi, yi * (xi ** 2)] for xi, yi in zip(x, y)])
    eigenvalues, U = np.linalg.eig(np.dot(A.T,A))
    return U[:, np.argmin(eigenvalues)]


# Inputs and printing solutions to questions
print("Question 1")
A = np.array([[11, -7, 4], [4, 0, 8], [7, 8, 7]])
x = np.array([10, 11, 13])
print(ex1(A, x), '\n')
A = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
B = A + np.full((len(A), len(A)), 2)
b = np.array([1, 2, 3])
n = 4
print('q2 result: ', ex2(A, B, n, b))
print('\nQ5')
X = np.array([[-0.966175231649752, -0.920529100440521, -0.871040946427231, -0.792416754493313, -0.731997794083466, -0.707678784846507, -0.594776425699584, -0.542182374657374, -0.477652051223985, -0.414002394497506, -0.326351540865686, -0.301458382421319, -0.143486910424499, -0.0878464728184052, -0.0350835941699658, 0.0334396260398352, 0.0795033683251447, 0.202974351567305, 0.237382785959596, 0.288908922672592, 0.419851917880386, 0.441532730387388, 0.499570508388721, 0.577394288619662, 0.629734626483965, 0.690534081997171, 0.868883439039411, 0.911733893303862, 0.940260537535768, 0.962286449219438],[1.61070071922315, 2.00134259950511, 2.52365719332252, 2.33863055618848, 2.46787274461421, 2.92596278963705, 4.49457749339454, 5.01302648557115, 5.53887922607839, 5.59614305167494, 5.3790027966219, 4.96873291187938, 3.56249278950514, 2.31744895283007, 2.39921966442751, 1.52592143155874, 1.42166345066052, 1.19058953217964, 1.23598301133586, 0.461229833080578, 0.940922128674924, 0.73146046340835, 0.444386541739061, 0.332335616103906, 0.285195114684272, 0.219953363135822, 0.234575259776606, 0.228396325882262, 0.451944920264431, 0.655793276158532]])
a = fit_rational(X)
d = fit_rational_normed(X)
print('\nPart A solution: ', a)
print('\nPart B solution: ', d)

f = lambda xi, u: (u[0] + u[1]*xi + u[2]*xi**2) / (u[3] + u[4]*xi + u[5]*xi**2)
xi_range = np.linspace(min(X[0]), max(X[0]), 100)
f_a_values = f(xi_range, a)
f_d_values = f(xi_range, d)

plt.figure(figsize=(10, 6))
plt.plot(X[0], X[1], color='red', label='(xi, yi)')
plt.scatter(xi_range, f_a_values, color='blue', label='f(xi, 5a)')
plt.scatter(xi_range, f_d_values, color='green', label='f(xi, 5d)')

plt.title('Plot of Given Points and Functions')
plt.xlabel('xi')
plt.ylabel('yi / f(xi)')
plt.legend()
plt.grid(True)
plt.show()



