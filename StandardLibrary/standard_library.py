# standard_library.py
"""Python Essentials: The Standard Library.
Katherine Collier
MTH 420
4/18/2025
"""

from math import sqrt


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
    new_tuple = test_tuple
    new_tuple += 1
    if test_tuple == new_tuple:
        print("tuples are mutable")
    else:
        print("tuples are immutable")

    test_set = {0, 1, 2, 3}
    new_set = test_set
    new_set += 1
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
    import calculator

    return math.sqrt(calculator.sum(calculator.product(a, a), calculator.product(b, b))) 
    # raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    from itertools import combinations

    my_list = list(combinations(A))

    my_list += set(), set(A)

    return my_list

    #raise NotImplementedError("Problem 4 Incomplete")

import random
import time

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    remaining = list[1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = time.time()

    while remaining:
        total_time = time.time - start
        print("You still need to close: ", remaining)
        print("It has been: ", total_time, " seconds")

        if total_time > timelimit:
            print("You Lose :(")
            break

        if sum(remaining) <= 6:
            roll = random.randint(1, 6)
        else:
            roll = random.randint(1, 6) + random.randint(1, 6)

        if not isvalid(roll, remaining):
            print("You Lose :(")

            break
        player_input = input("Enter numbers to flip: ")
        choices = parse_input(player_input, remaining)

        if not choices or sum(choices) != roll:
            print("Try again! ")
            continue

        for number in choices:
            remaining.remove(number)

        if not remaining:
            print("You win!")
            break


    time = time.time() - start
    # raise NotImplementedError("Problem 5 Incomplete")
