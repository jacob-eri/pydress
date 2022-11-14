"""Collection of 3-vector operations that are required by the `dress` package.

A 3-vector is represented by an array with shape (3,). Similarly, N vectors
can be represented by an array of shape (3,N)."""

import numpy as np


def check_vectors(*vecs):
    """Check that the given array is a valid 3-vector."""

    for vec in vecs:
        if (vec.ndim != 2) and (vec.shape[0] != 3):
            raise ValueError('A vector must have shape (3,N)')


def dot(v1, v2):
    """Dot product between two vectors.

    Parameters
    ----------
    v1, v2 : array of shape (3,N)
        The vectors to multiply.

    Returns
    -------
    prod : array of shape (N,)
        The dot product of the vectors."""
    
    check_vector(v1, v2)
    prod = np.sum(v1*v2, axis=0)
    
    return prod


def cross(v1, v2):
    """Cross product between two vectors.

    Parameters
    ----------
    v1, v2 : array of shape (3,N)
        The vectors to multiply.

    Returns
    -------
    prod : array of shape (N,)
        The cross product of the vectors."""

    check_vectors(v1, v2)
    prod = np.cross(v1, v2, axis=0)
    
    return prod
    
    
