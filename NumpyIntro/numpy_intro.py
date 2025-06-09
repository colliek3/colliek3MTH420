# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Katherine Collier
MTH 420
25 April 2025
"""

import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    A = np.array([[3, -1, 4],[1, 5, -9]])
    B = np.array([[2, 6, -5, 3],[5, -8, 9, 7],[9, -3, -2, -3]])

    return np.dot(A, B)
    # raise NotImplementedError("Problem 1 Incomplete")


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    A = np.array([[3, 1, 4],[1, 5, 9],[-5, 3, 1]])
    squared = np.dot(A, A)
    cubed = np.dot(A, squared)

    return -cubed + 9 * squared - 15 * A
    raise NotImplementedError("Problem 2 Incomplete")


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.ones(7)
    A = np.triu(A)
    # A is 7x7 upper triang 1s

    B = np.zeros((7, 7))
    B += np.triu(np.ones((7, 7)) * 5, k = 1)
    B += np.tril(np.ones((7, 7)) * -1, k = 0)
    product = np.dot(np.dot(A, B), A)

    return A.astype(np.int64), B.astype(np.int64)
    # raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    C = np.copy(A)

    C[C < 0] = 0

    return C
    # raise NotImplementedError("Problem 4 Incomplete")


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.array([[-2, 0, 0],[0, -2, 0], [0, 0, -2]])

    column1 = np.vstack([np.zeros((3, 3)), A, B])
    column2 = np.vstack([A.T, np.zeros((2, 2)), np.zeros((3, 2))])
    column3 = np.vstack([np.eye(3), np.zeros((2, 3)), C])

    block = np.hstack([column1, column2, column3])
    return block
    # raise NotImplementedError("Problem 5 Incomplete")


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    row_sums = A.sum(axis = 1, keepdims=True)

    return A / row_sums
    # raise NotImplementedError("Problem 6 Incomplete")


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    raise NotImplementedError("Problem 7 Incomplete")
