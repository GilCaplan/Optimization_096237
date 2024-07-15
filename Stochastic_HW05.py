import numpy as np
import matplotlib.pyplot as plt

# Given parameters
Q = np.array([
    [0, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0, 0.4, 0.3, 0.2],
    [0.2, 0.1, 0, 0.4, 0.3],
    [0.3, 0.2, 0.1, 0, 0.4],
    [0.4, 0.3, 0.2, 0.1, 0]
])
T = 100000
pi = [1/8, 1/8, 1/4, 1/3, 1/6]
S = [0,1,2,3,4]
def algo():
    Y = [np.random.choice(S)]
    visits = np.zeros(len(S))
    i = Y[0]
    for t in range(1, T):
        j = np.random.choice(S, p=Q[i])
        alpha = min([1, (pi[j] * Q[j, i]) / (pi[i] * Q[i, j])])
        Y.append(j if np.random.uniform(0, 1) < alpha else Y[t-1])
        visits[Y[-1]] += 1

    plt.bar([1,2,3,4,5], visits / T)
    plt.xlabel('States')
    plt.ylabel('Proportion of Visits')
    plt.title('Proportions of Visits in Each State')
    plt.show()

algo()