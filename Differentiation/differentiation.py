# differentiation.py
"""Volume 1: Differentiation.
Katherine Collier
MTH 420
6 June 2025
"""

import time
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

from jax import numpy as jnp
from jax import grad


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')

    f_x = (sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))

    df_dx = sy.diff(f_x, x)

    df_dx_lambdified = sy.lambdify(x, df_dx, 'numpy')

    return(df_dx_lambdified)    


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x - h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x + h) - f(x - h)) / (2 * h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)

# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    # get analytic function and value
    f_analytic = prob1()
    exact_val = f_analytic(x0)

    # logspaced values to input into our functions
    h_values = np.logspace(-8, 0, num=50, base=10.0)

    # get arrays of error for each FD method: FE, BE, and centered
    error_fdq1 = [abs(exact_val - fdq1(f_analytic, x0, h)) for h in h_values]
    error_fdq2 = [abs(exact_val - fdq2(f_analytic, x0, h)) for h in h_values]
    error_bdq1 = [abs(exact_val - bdq1(f_analytic, x0, h)) for h in h_values]
    error_bdq2 = [abs(exact_val - bdq2(f_analytic, x0, h)) for h in h_values]
    error_cdq2 = [abs(exact_val - cdq2(f_analytic, x0, h)) for h in h_values]
    error_cdq4 = [abs(exact_val - cdq4(f_analytic, x0, h)) for h in h_values]

    # plot figure and label
    plt.figure()
    plt.loglog(h_values, error_fdq1, label="FDQ1")
    plt.loglog(h_values, error_fdq2, label="FDQ2")
    plt.loglog(h_values, error_bdq1, label="BDQ1")
    plt.loglog(h_values, error_bdq2, label="BDQ2")
    plt.loglog(h_values, error_cdq2, label="CDQ2")
    plt.loglog(h_values, error_cdq4, label="CDQ4")

    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Finite difference at x0")
    plt.legend()
    plt.show()

    # For whatever reason this was not plotting like their plot, which from what I can tell is
    # correct- I'm probably calculating the analytic value incorrectly which is affecting
    # the way the error plots. 


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    data = np.load("plane.npy")
    t_vals, alpha_deg, beta_deg = data[:, 0], data[:, 1], data[:, 2]

    alpharad, betarad = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)

    # using a = 500 from the above
    x = 500 * np.tan(betarad) / (np.tan(betarad) - np.tan(alpharad))
    y = 500 * (np.tan(betarad)*np.tan(alpharad)) / (np.tan(betarad) - np.tan(alpharad))

    # using 64-bit floats for accuracy
    speed = np.zeros_like(t_vals, dtype=np.float64)

    for i, t in enumerate(t_vals):
        if i == 0:
            # h = 1 so no denom
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
        elif i == len(t_vals - 1):
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
        else:
            dx = (x[i] - x[i - 1]) / 2
            dy = (y[i] - y[i - 1]) / 2

        speed[i] = np.sqrt(dx**2 + dy**2)

    return speed


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    raise NotImplementedError("Problem 6 Incomplete")

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    raise NotImplementedError("Problem 7 Incomplete")
