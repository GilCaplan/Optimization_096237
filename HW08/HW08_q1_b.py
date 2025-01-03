import cvxpy as cp
import numpy as np

# Define the variables
x = cp.Variable(2)


# Define the constraints
constraints = [
    cp.quad_over_lin(x[0] , x[0] + x[1]) + cp.power(cp.quad_over_lin(x[0], x[1]) + 1, 8) <= 100,
    x[0] + x[1] >= 4,
    x[1] >= 1
]

# Define the objective function
objective = cp.Maximize(cp.sqrt(2*x[1] + 5) - cp.square(x[0] + x[1]) - 2*x[0] - 3*x[1])



# Formulate and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output the solution
print("Optimal value:", problem.value)
print("Optimal x:", x[0].value)
print("Optimal y:", x[1].value)
