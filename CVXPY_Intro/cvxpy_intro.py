# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Katherine Collier
MTH 420
5/30/2025
"""

import numpy as np
import cvxpy as cp

""" All of these problems seem like they suffer from some level
    of rounding error as the calculator uses the Floating Point values.
    """
def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # constraints
    A = np.array([1, 2, 0])
    B = np.array([0, 1, -4])
    C = np.array([2, 1, 3])
    D = np.eye(3)

    constraints = [A @ x <= 3, B @ x <=1, C @ x >= 12, D @ x >= 0]

    problem = cp.Problem(objective, constraints)
    optimal = problem.solve()

    return x.value, optimal


# raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # This wasn't working for me with the example given
    n  = A.shape[1]
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [A @ x == b]

    problem = cp.Problem(objective, constraints)
    optimal = problem.solve()

    return x.value, optimal
    # raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    supply = np.array([7, 2, 4])
    demand = np.array([5, 8])
    # 2 by 3 array represents the cost between edges
    costs =  np.array([[4, 7],[6, 8],[8, 9]])

    # create the variable as a matrix- i, j th element represents
    # the optimized number of pianos between two edges
    x = cp.Variable((3, 2), nonneg = True)

    objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
    constraints = [cp.sum(x, axis = 1) <= supply, cp.sum(x, axis = 0) == demand]

    problem = cp.Problem(objective, constraints)
    optimal = problem.solve()

    return x.value, optimal
    # raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3)
    Q = np.array([[3, 2, 1],[2, 4, 2],[1, 2, 3]]) * 0.5
    r = np.array([3, 0, 1])

    problem = cp.Problem(cp.Minimize(cp.quad_form(x, Q) + r.T @ x))
    optimal = problem.solve()

    return x.value, optimal
    # raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = A.shape[1]
    x = cp.Variable(n, nonneg = True)

    objective = cp.Minimize(cp.norm(A @ x - b, 2))
    # leq allows us to get close to the ans without violating convex rules
    constraints = [cp.norm(x, 1) <= 1]

    problem = cp.Problem(objective, constraints)
    optimal = problem.solve()

    return x.value, optimal
    # raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")
