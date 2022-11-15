"""Classes and methods for working with velocity distributions
within the `dress` framework."""


import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d

from dress import relkin
from dress import sampler
from dress import vec_ops as vec


class VelocityDistribution:
    """A class representing the velocity distribution of an ensemble of particles.

    In order to use it, a user would typically subclass this class and implement
    a method called `_sample(self, n)`, which returns `n` velocity samples from
    the distribution.

    Attributes
    ----------
    particle : instance of dress.particles.Particle
        The particle that the distribution represent

    density : float
        The density (in particles/m**3) of particles in the distribution.

    v_collective : array-like with three elements
        Collective velocity (m/s) of all particles in the distribution.
        Setting `v_collective=None` means no collective motion."""

    
    def __init__(self, particle, density, v_collective=None):
        
        self.particle = particle
        self.density = density
        self.v_collective = v_collective


    def sample(self, n, **kwargs):
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
        v = self._sample(n, **kwargs)
        
        if self.v_collective is not None:
            v = v + self.v_collective
        
        return v

    def _sample(self, n, **kwargs):
        """Sampling method, to be overloaded by subclasses."""
        return np.zeros(3,n)

    @property
    def v_collective(self):
        return self._v_collective
    
    @v_collective.setter
    def v_collective(self, v):
        
        if v is None:
            self._v_collective = None
            return

        v = vec.make_vector(v)
        
        self._v_collective = v


class EnergyDistribution(VelocityDistribution):
    """A velocity distribution characterized mainly by its distribution in energy.

    The speeds are determined from an arbitrary energy distribution and the 
    direction of the corresponding velocity vectors are taken to be uniformly 
    distributed in a given pitch range, where 

        pitch = v_parallel/v

    and v_parallel is the component of the velocity relative to a given 
    reference direction.

    Typically, a user would subclass this method and override the `sample_energy` method.

    Attributes
    ----------
    pitch_range : array-like with 2 elements
        Pitch values of the particles are taken to be uniformly distributed 
        between the two values in this sequence.

    ref_dir : array with shape (3,1)
        Pitch values are given relative to this direction.
    
    For the rest of the attributes see docstring of the parent class(es)."""


    def __init__(self, m, density, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(m, density, v_collective=v_collective)
        self.pitch_range = pitch_range
        self.ref_dir = ref_dir

        
    @property
    def ref_dir(self):
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, u):
        u = vec.make_vector(u)
        if u.shape != (3,1):
            raise ValueError(f'ref_dir must be a single three-vector.')

        self._ref_dir = vec.normalize(u)

        # Construct coordinate system where one axis aligns with the reference direction
        self._set_local_basis_vectors()

    def _set_local_basis_vectors(self):
        """Construct a rectangular coordinate system with basis vectors 
        e1, e2, e3, where the e2 vector is parallel to self.ref_dir."""

        # ref_dir is already normalized to unity
        e2 = self.ref_dir
        
        # There are infinitely many possible choices for e1 and e2 (we can choose any 
        # of them since we will anyway sample the azimuthal angle uniformly). We start 
        # by finding a vector that is non-parallel to e2.
        u = e2 + vec.make_vector([1,0,0])
        u = vec.normalize(u)
        
        # Now e1 and e3 can be constructed by taking appropriate cross products with e2
        e1 = vec.cross(e2, u)
        e1 = vec.normalize(e1)
        e3 = vec.cross(e1, e2)

        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def _sample(self, n, **kwargs):
        """Sample energy and pitch values and calculate corresponding velocities."""
        E = self.sample_energy(n, **kwargs)
        p = self.sample_pitch(n)

        v = self._calc_velocity(E,p)

        return v

    def sample_energy(self, n, **kwargs):
        """Sample `n` energy values from the distribution, to be overloaded by the sub-classes."""
        return np.zeros(n)

    def sample_pitch(self, n):
        """Sample `n` pitch values uniformly distributed in the range given by self.pitch_range."""
        return sampler.sample_uniform(self.pitch_range, n)

    def _calc_velocity(self, energy, pitch):
        """Calculate velocity corresponding to given values of energy (keV) and pitch."""

        # Velocity components parallel and perpendicular to the reference direction
        v = relkin.get_speed(energy, self.particle.m)       # m/s
        v_par = v*pitch
        v_perp = v*np.sqrt(1 - pitch**2)

        # Parallel velocity is given with respect to the e2 basis vector of the 
        # local coordinate system.
        v_par = v_par*self.e2

        # Azimuthal angle of perpendicular velocity is taken to be uniformly distributed
        phi = sampler.sample_uniform([0,2*np.pi], len(energy))
        v_perp = v_perp*(np.cos(phi)*self.e1 - np.sin(phi)*self.e3)

        return v_par + v_perp
        


class MaxwellianDistribution(EnergyDistribution):
    """A class representing a Maxwellian velocity distribution.

    Attributes
    ----------    
    T : float
        Temperature (in keV).

    For the rest of the attributes see docstring of the parent class(es)."""

    
    def __init__(self, T, m, density, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(m, density, v_collective=v_collective, 
                         pitch_range=pitch_range, ref_dir=ref_dir)
        
        self.T = T
        self._spread = np.sqrt(self.T/self.particle.m)*c   # standard deviation of the distribution (m/s)


    def sample_energy(self, n, **kwargs):
        """Sample energies (in keV) from the Maxwellian distribution."""
        
        # The (kinetic) energy is distributed as a chi2 variable with 3 degrees of freedom
        E = np.random.chisquare(3, size=n) * 0.5 * self.T
        
        return E
        

class MonoEnergeticDistribution(EnergyDistribution):
    """A class representing a mono-energetic velocity distribution.

    Attributes
    ----------
    E0 : float
        Kinetic energy of the particles (in keV)

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, E0, m, density, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(m, density, v_collective=None, 
                         pitch_range=pitch_range, ref_dir=ref_dir)
        
        self.E0 = E0

    
    def sample_energy(self, n, **kwargs):
        """Sample energies (in keV) from the mono-energetic distribution."""
        return sampler.sample_mono(self.E0, n)



class TabulatedEnergyDistribution(EnergyDistribution):
    """A class representing a velocity distribution where the energy distribution
    is given by tabulated values of f vs E.

    Attributes
    ----------
    E_axis : array of shape (N,)
        The energy values (keV) of the tabulated distribution.
    
    energy_dist : array of shape (N,)
        The distribution value at the tabulated energies. The absolute normalization 
        of the distribution is not important, but the values f(E)*dE should be propoprtional
        to the number of particles in the energy interval dE.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, E_axis, energy_dist, m, density, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(m, density, v_collective=None, 
                         pitch_range=pitch_range, ref_dir=ref_dir)

        self.E_axis = np.array(E_axis)
        self.energy_dist = np.array(energy_dist)

    
    def sample_energy(self, n, **kwargs):
        """Sample energies (in keV) from tabulated energy distribution."""
        
        n = int(n)
        method = kwargs.get('method', 'acc_rej')

        if method == 'acc_rej':
            # Function to sample from
            f = interp1d(self.E_axis, self.energy_dist, bounds_error=False, fill_value=0)
            
            # Sampling space
            lims = ([self.E_axis[0]], [self.E_axis[-1]])
            fmax = self.energy_dist.max()
            
            # Sample energies
            s = sampler.sample_acc_rej(f, lims, fmax, n)
            E = np.squeeze(s)
            
        elif method == 'inv_trans':
            E = sampler.sample_inv_trans(self.energy_dist, self.E_axis, n)

        else:
            raise ValueError(f'Unknown sampling method: {method}')

        return E
