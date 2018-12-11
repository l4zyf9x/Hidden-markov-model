import numpy as np


def expand_matrix(matrix, length, value=0):
    """ 
    Return a array with new shape which has all element increasing `length`. And
    fill new variable with `value`

    Examples
    --------
    >>> a=np.zeros((3,4,3))

    >>> expand_matrix(a, 2)

    >>> a.shape # (5,6,5) 
    """
    # Number dimension of matrix
    d = len(matrix.shape)
    # Expand one by one dimesion
    for i in range(d):
        matrix = expand_one_dimesion(matrix, length, i, value)
    return matrix


def expand_one_dimesion(matrix, length, dimension, value=0):
    """ 
    Return a array with new shape which increasing `length` on specify `dimension`. And
    fill new variable with `value`

    Examples
    --------
    >>> x = np.zeros((3, 4, 5))

    >>> x = expand_one_dimesion(x, 3, 1)


    >>> x.shape # (3, 7, 5) 
    """
    d = len(matrix.shape)

    if dimension > d or dimension < 0:
        raise ValueError('dimesion is invalid')
    # Expand one by one dimesion
    extended_shape = tuple([length if(idx == dimension) else old_length for idx, old_length in enumerate(matrix.shape)])
    # Create new extend
    extended_element = np.ones(extended_shape) * value
    return np.concatenate((matrix, extended_element), axis=dimension)
