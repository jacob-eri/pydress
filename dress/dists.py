"""Classes and methods for working with velocity distributions
within the `dress` framework."""


import numpy as np
from scipy.constants import c


class VelocityDistribution:
    """A class representing the velocity distribution of an ensemble of particles.

    In order to use it, a user would typically subclass this class and implement
    a method called `_sample(self, n)`, which returns `n` velocity samples from
    the distribution.

    Attributes
    ----------
    m : float
        Mass of the particles in the distribution (in keV/c**2)

    density : float
        The density (in particles/m**3) of particles in the distribution.

    v_collective : array-like with three elements
        Collective velocity (m/s) of all particles in the distribution.
        Setting `v_collective=None` means no collective motion."""

    
    def __init__(self, m, density, v_collective=None):
        
        self.m = m
        self.density = density
        self._set_collective_velocity(v_collective)


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

        n = int(n)
        v = self._sample(n)
        
        if self.v_collective is not None:
            v = v + self.v_collective[:,None]
        
        return v

    
    def _sample(self,n):
        """Sampling method to be overloaded by subclasses."""
        return np.zeros(3,n)

    
    def _set_collective_velocity(self, v):
        
        if v is None:
            self.v_collective = None
            return

        v = np.atleast_1d(v)
        
        if v.shape != (3,):
            raise ValueError('Collective velocity must be array-like with three components (or None)')
        
        self.v_collective = v


class EnergyDistribution(VelocityDistribution):
    """A velocity distribution characterized mainly by its distribution in energy.

    The speeds are determined from an arbitrary energy distribution and the 
    direction of the corresponding velocity vectors are taken to be uniformly 
    distributed in a given pitch range, where 

        pitch = v_parallel/v

    and v_parallel is the component of the velocity relative to a given 
    reference direction.

    Attributes
    ----------
    pitch_range : array-like with 2 elements
        Pitch values of the particles are taken to be uniformly distributed 
        between the two values in this sequence.

    ref_dir : array with shape (3,)
        Pitch values are given relative to this direction.
    
    For the rest of the attributes see docstring for the parent class(es)."""


    def __init__(self, m, density, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(self, m, density, v_collective=v_collective)
        self.pitch_range = pitch_range
        self.ref_dir = ref_dir


class MaxwellianDistribution(VelocityDistribution):
    """A class representing a Maxwellian velocity distribution.

    Attributes
    ----------    
    T : float
        Temperature (in keV).

    pitch_range : array-like with 2 elements
        Pitch values (v_parallel/v) of the particles are taken to be uniformly 
        distributed between the two values in this sequence.

    ref_dir : array-like with 3 elements
        Pitch values (v_parallel/v) are given relative to this direction.

    For the rest of the attributes see docstring for the parent class(es)."""

    
    def __init__(self, m, T, density, v_collective=None, pitch_range=None, ref_dir=None):

        super().__init__(m, density, v_collective=v_collective)
        self.T = T

        self._spread = np.sqrt(self.T/self.m)*c   # standard deviation of the distribution (m/s)


    def _sample(self,n):
        v = np.random.normal(loc=0.0, scale=self._spread, size=(3,n))        
        return v
        

class MonoEnergeticDistribution(VelocityDistribution):
    """A class representing a mono-energetic velocity distribution.

    Attributes
    ----------
    E : float
        Kinetic energy of the particles (in keV)

    pitch_range : array-like with 2 elements
        Pitch values (v_parallel/v) of the particles are taken to be uniformly 
        distributed between the two values in this sequence.

    ref_dir : array-like with 3 elements
        Pitch values (v_parallel/v) are given relative to this direction.

    (for the rest of the attributes see docstring for the 
    `VelocityDistribution` class)"""

    def __init__(self, m, E, density, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(m, density, v_collective=None)
        self.E = E
        self.pitch_range = pitch_range
        self.ref_dir = np.array(ref_dir)
