"""Classes and methods for working with velocity distributions
within the `dress` framework."""


import numpy as np



class VelocityDistribution:
    """A class representing the velocity distribution of an ensemble of particles.

    In order to use it, a user would typically subclass this class and implement
    a method called `_sample(self, n)`, which returns `n` velocity samples from
    the distribution.

    Attributes
    ----------
    density : float
        The density (in particles/m**3) of particles in the distribution.

    v_collective : array of shape (3,)
        Collective velocity (m/s) of all particles in the distribution."""

    
    def __init__(self, density, v_collective=None):
        
        self._density = density
        self._set_collective_velocity(v_collective)


    @property
    def density(self):
        return self._density


    def sample(self,n):
        """Sample `n` velocity vectors from the distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        Returns
        -------
        v : array of shape (3,n)
            The sampled velocity vectors (in m/s)."""

        v = self._sample(n)
        return v

    
    def _sample(self,n):
        """Sampling method to be overloaded by subclasses."""
        pass

    
    def _set_collective_velocity(self, v):
        
        if v is None:
            self.v_collective = 0.0
            return

        v = np.atleast_1d(v)
        
        if v.shape != (3,):
            raise ValueError('Collective velocity must be an array with three components (or None)')
        
        self.v_collective = v
