"""Collection of 3-vector operations that are required by the `dress` package.

A 3-vector is represented by an array with shape (3,1). Similarly, N vectors
can be represented by an array of shape (3,N)."""

import numpy as np


def make_vector(*vecs):
    """Convert given arrays (or array-likes) to dress-vectors, if possible."""
    
    new_vecs = []

    for vec in vecs:
        v = np.atleast_2d(vec)
        
        # If input represented one single vector the array needs to be 
        # transposed in order to have size (3,1)
        if v.shape[0] == 1:
            v = v.T

        new_vecs.append(v)

    # Check that everything worked
    check_vector(*new_vecs)

    if len(new_vecs) == 1:
        return new_vecs[0]
    else:
        return new_vecs


def check_vector(*vecs):
    """Check that the given arrays represent valid 3-vectors."""

    for vec in vecs:
        if type(vec) is not np.ndarray:
            raise TypeError('Vector must be a numpy array with shape (3,N)')

        if not (vec.ndim == 2 and vec.shape[0] == 3):
            msg = f'\n{vec}\n cannot does not represent valid array of three-vectors (shape must be (3,N))'
            raise ValueError(msg)


def dot(v1, v2):
    """Dot product between two vectors.

    Parameters
    ----------
    v1, v2 : array of shape (3,N) or (3,1)
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

    check_vector(v1, v2)
    prod = np.cross(v1, v2, axis=0)
    
    return prod
    
    
def normalize(v, length=1):
    """Normalize vectors to given length.

    Parameters
    ----------
    v : array of shape (3,N)
        The vectors to be normalized.
    length : scalar or array of shape (N,)
        The new lengths of the vectors

    Returns
    -------
    u : array of shape (3,N)
        The renormalized vectors."""
    
    check_vector(v)
    
    N = v.shape[1]
    length = np.atleast_1d(length)
    if length.shape != (1,) and length.shape != (N,):
        raise ValueError(f'`length` must have either 1 element or {N} element')

    old_length = np.sqrt(dot(v,v))
    u = v * length / old_length

    return u
