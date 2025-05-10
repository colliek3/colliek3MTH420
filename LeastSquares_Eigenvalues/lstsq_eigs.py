# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Katherine Collier>
<MTH420>
<05/09/2025>
"""

import numpy as np
from cmath import sqrt
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    # creates qr factorization
    q, r = la.qr(A, mode='economic')

    # calculates Q^Tb
    qt = np.transpose(q)
    b1 = np.dot(qt, b)

    # solve triangular system for x
    x = la.solve_triangular(r, b1)

    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # define A and b based on housing.npy
    year, price_index = np.load("housing.npy").T

    A = np.column_stack((np.ones_like(year), year))
    b = price_index

    # find least squares
    x = least_squares(A, b)

    intercept, slope = x

    # create scale for axes
    x_fit = np.linspace(0, 16, 100)
    y_fit = intercept + slope * x_fit

    # create plots
    plt.scatter(year, price_index, color='blue', label='Scatter Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Least Squares')
    plt.xlabel('Year (from 0)')
    plt.ylabel('Housing Price index')
    plt.legend()
    plt.title('Least Squares Fit and Scatter Points of Housing Price Index')
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    from scipy import linalg as la

    # Define A and b
    year, data_index = np.load("housing.npy").T
    degrees = [3, 6, 9, 12]
    x_fit = np.linspace(0, 16, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # iterate over each subplot in degrees
    for ax, degree in zip(axes.flatten(), degrees):
        A = np.vander(year, degree+1, increasing=True)

        # calculate least squares accurately
        x = la.lstsq(A, data_index)[0]

        # fit axes properly
        A_fit = np.vander(x_fit, degree+1, increasing=True)
        y_fit = A_fit @ x

        # create each subplot
        ax.scatter(year, data_index, color='blue', label='Data')
        ax.plot(x_fit, y_fit, color='red', label=f'degree {degree}')
        ax.set_title(f'Polynomial Fit (Degree {degree})')
        ax.set_xlabel('Year from 0')
        ax.set_ylabel('Housing Price Index')
        ax.legend()

    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")
