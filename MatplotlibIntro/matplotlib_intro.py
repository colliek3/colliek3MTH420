# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Katherine Collier
MTH 420
2 May 2025 (Updated 6/9/25)
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    my_array = np.random.normal(size=(n, n))

    row_means = np.mean(my_array, axis = 1)

    return np.var(row_means)
    
def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n_vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    variances = [var_of_means(n) for n in n_vals]

    plt.plot(n_vals, variances)
    plt.show()

# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    y_arctan = np.arctan(x)

    plt.plot(x, y_sin, color = 'b')
    plt.plot(x, y_cos, color = 'r')
    plt.plot(x, y_arctan, color = 'g')
    plt.show()

# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 0.99, 100)
    x2 = np.linspace(1.01, 6, 100)

    y1 = 1/(x1-1)
    y2 = 1/(x2-1)

    plt.plot(x1, y1, "m--", linewidth=4)
    plt.plot(x2, y2, "m--", linewidth=4)

    plt.xlim([2, 6])
    plt.ylim([-6, 6])
    plt.show()


# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2*np.pi, 1000)
    y1 = np.sin(x)
    y2 = np.sin(2*x)
    y3 = 2*np.sin(x)
    y4 = 2*np.sin(2*x)

    #plt.subplots(2, 2)

    plt.subplot(221)
    plt.plot(x, y1, 'g-')
    plt.title("sin(x)")

    plt.subplot(222)
    plt.plot(x, y2, 'r--')
    plt.title("sin(2x)")

    plt.subplot(223)
    plt.plot(x, y3, 'b--')
    plt.title("2sin(x)")

    plt.subplot(224)
    plt.plot(x, y4, 'm:')
    plt.title("2sin(2x)")

    plt.axis([0, 2*np.pi, -2, 2])
    plt.suptitle("sin(x) scalings")

    plt.show()


# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.sin(Y)) / (X * Y)

    # heat map
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    # contour map
    plt.subplot(122)
    plt.contour(X, Y, Z, 20, cmap="coolwarm")
    plt.colorbar()

    plt.show()
