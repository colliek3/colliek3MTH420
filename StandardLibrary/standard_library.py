# standard_library.py
"""Python Essentials: The Standard Library.
Katherine Collier
MTH 420
4/18/2025
"""

from math import sqrt
import calculator
from itertools import chain, combinations
  
# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    
    return min(L), max(L), sum(L)/len(L)

    # raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    test_int = 1
    new_int = test_int
    new_int = 2
    if test_int == new_int:
        print("ints are mutable")
    else:
        print("ints are immutable")

    test_string = "hello"
    new_string = test_string
    test_string = "world"
    if test_string == new_string:
        print("strings are mutable")
    else:
        print("strings are immutable")

    test_list = [1, 2, 3]
    new_list = test_list
    new_list[0] = 0
    if test_list == new_list:
        print("lists are mutable")
    else:
        print("lists are immutable")

    test_tuple = ("test", "tuple")
    new_tuple = test_tuple + (1, )
    if test_tuple is new_tuple:
        print("tuples are mutable")
    else:
        print("tuples are immutable")

    test_set = {0, 1, 2, 3}
    new_set = test_set
    new_set | {1}
    if new_set == test_set:
        print("sets are mutable")
    else:
        print("sets are immutable")

    # raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return sqrt(calculator.sum(calculator.product(a, a), calculator.product(b, b))) 
    # my calculator module is in github
    # raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    return [set(comb) for r in range(len(A) + 1) for comb in combinations(A, r)]
    # for whatever reason this works in ipython directly but not when I run it from
    # the file

    #raise NotImplementedError("Problem 4 Incomplete")

import random
import time

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    raise NotImplementedError("Problem 5 Incomplete")
