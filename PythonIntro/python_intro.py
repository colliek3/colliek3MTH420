# python_intro.py
"""Python Essentials: Introduction to Python.
Katherine Collier
MTH 420
04/11/2025 (Updated 5/25/2025)
"""


# Problem 1 (write code below)
if __name__ == "__main__":
    print("Hello World!")

# Problem 2
def sphere_volume(r):
    """ Returns the volume of the sphere of radius 'r'.
    Uses 3.14159 for pi in computation.
    """
    volume = 4 / 3 * 3.14159 * r ** 3

    return volume
    # raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, "    ", b, "    ", c, d, e)
    # raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    length = len(my_string)
    # floor division, use +1 to include middle
    midpoint = (length + 1) // 2
    return my_string[:midpoint]

    # raise NotImplementedError("Problem 4 Incomplete")

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]
    #raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    my_list = ["bear", "ant", "cat", "dog"]

    my_list.append("eagle")
    my_list[2] = "fox"
    my_list.pop(1)
    # For sorting, we were told that sort existed so I hope I'm allowed to use it
    # I did some outside research for this- checking in in class
    my_list.sort(reverse=True)

    # can't think of a good way to replace a specific value without if statements 
    # and/or for loops, but I know where eagle is?
    
    my_list[1] = "hawk"

    my_list[3] = my_list[3] + "hunter"

    return my_list

    # raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    if word[0] in "aeiouy":
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"

#    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    largest_palindrome = 0
    for i in range(100, 999):
        for j in range(i, 999):
            product = i * j

            if str(product) == str(product)[::-1] and product > largest_palindrome:
                largest_palindrome = product
    return largest_palindrome

 # raise NotImplementedError("Problem 7 Incomplete")

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    my_list = []
    for i in range(1, n + 1):
        val = (-1) ** (i + 1) / i
        my_list.append(val)

    return sum(my_list)
    # raise NotImplementedError("Problem 8 Incomplete")
