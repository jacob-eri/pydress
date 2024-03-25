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

    In order to use it, one would typically subclass this class and implement
    a method called `_sample(self, n, index=0)`, which returns `n` velocity samples from
    the distribution at the given spatial index.

    Attributes
    ----------
    particle : instance of dress.reactions.Particle
        The particle that the distribution represent

    density : array of shape (N,)
        The density (in particles/m**3) of particles in the distribution,
        at each of the N spatial locations.

    v_collective : array with shape (3,N)
        Collective velocity (m/s) of all particles in the distribution, at each spatial location.
        `v_collective=None` means no collective motion.

    n_spatial : int
        The number of spatial points the distribution is given at. This is typically inferred 
        from the shapes of the distribution data supplied to the constructors of the various 
        sub-classes below.

    pos : tuple of arrays
        Coordinates of each spatial point, e.g. (R,Z) or (X,Y,Z).
        Not necessary for any of the sampling routines, but could be useful
        for the user when plotting etc."""

    def __init__(self, particle, density=None, v_collective=None):
        """Create velocity distribution.

        Parameters
        ----------
        particle : instance of dress.particles.Particle
            The particle that the distribution represent

        density : float or array-like
            The density (in particles/m**3) of particles in the distribution,
            at each of the N spatial locations.

        v_collective : array-like with shape (3,N)
            Collective velocity (m/s) of all particles in the distribution, at each spatial location.
            `v_collective=None` means no collective motion."""
        
        self.particle = particle
        self.density = np.atleast_1d(density)
        self.v_collective = v_collective

        self.pos = None

    def sample(self, n, index=0):
        """Sample `n` velocity vectors from the distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.
        index : int
            The spatial index of the `dist` array to sample from

        Returns
        -------
        v : array of shape (3,n)
            The sampled velocity vectors (in m/s)."""

        n = int(n)
        v = self._sample(n, index=index)
        
        if self.v_collective is not None:
            v_coll = vec.make_vector(self.v_collective[:,index])
            v = v + v_coll
        
        return v

    def _sample(self, n, index=0):
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
        
        if v.shape[1] != len(self.density):
            v = vec.repeat(v, len(self.density))

        self._v_collective = v

    def _set_1d_dist(self, dist):
        """Set the `dist` attribute from the given input array."""

        dist = np.atleast_1d(dist)
        
        if dist.ndim == 1:
            dist = dist[np.newaxis, :]   # dummy spatial axis

        self.dist = dist
        self.n_spatial = len(dist)

    def _set_2d_dist(self, dist):
        """Set the `dist` attribute from the given input array."""

        dist = np.atleast_2d(dist)
        
        if dist.ndim == 2:
            dist = dist[np.newaxis, :, :]   # dummy spatial axis

        self.dist = dist
        self.n_spatial = len(dist)
        

class VparVperpDistribution(VelocityDistribution):
    """A velocity distribution given in terms of the velocity components parallel and
    perpendicular relative to a given reference direction.

    Typically, a user would sub-class this class and override the `sample_vpar_vperp` method.

    Attributes
    ----------
    ref_dir : array with shape (3,N)
        The parallel and perpendicular speeds are given relative to this direction.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, particle, density=None, v_collective=None, ref_dir=[0,1,0]):
        
        super().__init__(particle, density=density, v_collective=v_collective)
        self.ref_dir = ref_dir

    @property
    def ref_dir(self):
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, u):
        u = vec.make_vector(u)

        if u.shape[1] != len(self.density):
            u = vec.repeat(u, len(self.density))

        self._ref_dir = vec.normalize(u)

        # Construct coordinate system where one axis aligns with the reference direction
        self._set_local_basis_vectors()

    def _sample(self, n, index=0):
        """Sample paralell and perpendicular speeds and calculate corresponding velocities."""
        
        v_par, v_perp = self.sample_vpar_vperp(n, index=index)
        v = self._calc_velocity(v_par, v_perp, index=index)
        
        return v

    def sample_vpar_vperp(self, n, index=0):
        """Sample paralell and perpendicular speeds (m/s). To be overridden by sub-classes."""
        n = int(n)
        return np.zeros(n), np.zeros(n)

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

    def _calc_velocity(self, v_par, v_perp, index=0):
        """Calculate velocity corresponding to given values of paralell 
        and perpendicular speeds (m/s)."""

        n_samples = len(v_par)

        e1 = vec.make_vector(self.e1[:,index])
        e2 = vec.make_vector(self.e2[:,index])
        e3 = vec.make_vector(self.e3[:,index])

        # Parallel velocity is given with respect to the e2 basis vector of the 
        # local coordinate system.
        v_par = v_par*e2

        # Azimuthal angle of perpendicular velocity is taken to be uniformly distributed
        phi = sampler.sample_uniform([0,2*np.pi], n_samples)
        v_perp = v_perp*(np.cos(phi)*e1 - np.sin(phi)*e3)

        return v_par + v_perp


class EnergyPitchDistribution(VparVperpDistribution):
    """A velocity distribution characterized by its distribution in energy and pitch.

    The speed is determined from the energy and the direction of the velocity vector
    is determined from the pitch, defined as

        pitch = v_parallel/v

    where v_parallel is the component of the velocity relative to a given 
    reference direction.

    Typically, a user would sub-class this class and override the 
    `sample_energy_pitch` methods.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, particle, density=None, v_collective=None, ref_dir=[0,1,0]):

        super().__init__(particle, density=density, v_collective=v_collective, ref_dir=ref_dir)
        self.ref_dir = ref_dir

    def sample_vpar_vperp(self, n, index=0):
        """Sample (energy,pitch) and evaluate the corresponding (v_par,v_perp), in m/s."""
        E, pitch = self.sample_energy_pitch(n, index=index)

        v = relkin.get_speed(E, self.particle.m)       # m/s
        v_par = v*pitch
        v_perp = v*np.sqrt(1 - pitch**2)

        return v_par, v_perp

    def sample_energy_pitch(self, n, index=0):
        """Sample `n` energy and pitch values from the distribution, to be overloaded by the sub-classes."""
        n = int(n)
        return np.zeros(n), np.zeros(n)


class SpeedDistribution(VparVperpDistribution):
    """A velocity distribution characterized mainly by its distribution in speed.

    The speeds are determined from an arbitrary speed distribution and the 
    direction of the corresponding velocity vectors are taken to be uniformly 
    distributed in a given pitch range, where 

        pitch = v_parallel/v

    and v_parallel is the component of the velocity relative to a given 
    reference direction.

    Typically, a user would subclass this class and override the `sample_speed` method.

    Attributes
    ----------
    pitch_range : array-like with 2 elements
        Pitch values of the particles are taken to be uniformly distributed 
        between the two values in this sequence.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, particle, density=None, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(particle, density=density, v_collective=v_collective, ref_dir=ref_dir)
        self.pitch_range = pitch_range

    def sample_vpar_vperp(self, n, index=0):
        v = self.sample_speed(n, index=index)
        pitch = self.sample_pitch(n)

        v_par = v*pitch
        v_perp = v*np.sqrt(1 - pitch**2)

        return v_par, v_perp

    def sample_speed(self, n, index=0):
        """Sample `n` speed values from the distribution, to be overloaded by the sub-classes."""
        n = int(n)
        return np.zeros(n)

    def sample_pitch(self, n):
        """Sample `n` pitch values uniformly distributed in the range given by self.pitch_range."""
        return sampler.sample_uniform(self.pitch_range, n)


class EnergyDistribution(EnergyPitchDistribution):
    """A velocity distribution characterized mainly by its distribution in energy.

    The speeds are determined from an arbitrary energy distribution and the 
    direction of the corresponding velocity vectors are taken to be uniformly 
    distributed in a given pitch range, where 

        pitch = v_parallel/v

    and v_parallel is the component of the velocity relative to a given 
    reference direction.

    Typically, a user would subclass this class and override the `sample_energy` method.

    Attributes
    ----------
    pitch_range : array-like with 2 elements
        Pitch values of the particles are taken to be uniformly distributed 
        between the two values in this sequence.

    For the rest of the attributes see docstring of the parent class(es)."""


    def __init__(self, particle, density=None, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(particle, density=density, v_collective=v_collective, ref_dir=ref_dir)
        self.pitch_range = pitch_range

    def sample_energy_pitch(self, n, index=0):
        E = self.sample_energy(n, index=index)
        p = self.sample_pitch(n)

        return E, p

    def sample_energy(self, n, index=0):
        """Sample `n` energy values from the distribution, to be overloaded by the sub-classes."""
        n = int(n)
        return np.zeros(n)

    def sample_pitch(self, n):
        """Sample `n` pitch values uniformly distributed in the range given by self.pitch_range."""
        return sampler.sample_uniform(self.pitch_range, n)
        

class MaxwellianDistribution(EnergyDistribution):
    """A class representing a Maxwellian velocity distribution.

    Attributes
    ----------    
    T : float or array of shape (N,)
        Temperature (in keV).

    For the rest of the attributes see docstring of the parent class(es)."""

    
    def __init__(self, T, particle, density=None, v_collective=None, pitch_range=[-1,1], ref_dir=[0,1,0]):

        super().__init__(particle, density=density, v_collective=v_collective, 
                         pitch_range=pitch_range, ref_dir=ref_dir)
        
        self.T = np.atleast_1d(T)
        self.n_spatial = len(self.T)

    def sample_energy(self, n, index=0):
        """Sample energies (in keV) from the Maxwellian distribution."""
        
        # The (kinetic) energy is distributed as a chi2 variable with 3 degrees of freedom
        E = np.random.chisquare(3, size=n) * 0.5 * self.T[index]
        
        return E
        

class MonoEnergeticDistribution(EnergyDistribution):
    """A class representing a mono-energetic velocity distribution.

    Attributes
    ----------
    E0 : float or array of length N
        Kinetic energy of the particles (in keV)

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, E0, particle, density=None, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(particle, density=density, v_collective=None, 
                         pitch_range=pitch_range, ref_dir=ref_dir)
        
        self.E0 = np.atleast_1d(E0)

    def sample_energy(self, n, index=0):
        """Sample energies (in keV) from the mono-energetic distribution."""
        return sampler.sample_mono(self.E0[index], n)


class TabulatedEnergyDistribution(EnergyDistribution):
    """A class representing a velocity distribution where the energy distribution
    is given by tabulated values of f vs E.

    Attributes
    ----------
    E_axis : array of shape (N,)
        The energy values (keV) of the tabulated distribution.
    
    dist : array of shape (N,)
        The distribution value at the tabulated energies. The absolute normalization 
        of the distribution is not important, but the values f(E)*dE should be proportional
        to the number of particles in the energy interval dE.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, E_axis, dist, particle, density=None, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(particle, density=density, v_collective=None, 
                         pitch_range=pitch_range, ref_dir=ref_dir)

        self.E_axis = np.array(E_axis)
        self._set_1d_dist(dist)
    
    def sample_energy(self, n, index=0):
        """Sample energies (in keV) from tabulated energy distribution."""
        
        n = int(n)
        E = sampler.sample_tab(self.dist[index], self.E_axis, n_samples=n)

        return E


class TabulatedEnergyPitchDistribution(EnergyPitchDistribution):
    """A class representing a velocity distribution where the energy-pitch distribution
    is given by tabulated values of f(E,pitch).

    Attributes
    ----------
    E_axis : array of shape (NE,)
        The energy values (keV) of the tabulated distribution.

    pitch_axis : array of shape (NP,)
        The pitch values of the tabulated distribution

    dist : array of shape (NE,NP)
        The distribution value at the tabulated (energy,pitch)-values. The absolute normalization 
        of the distribution is not important, but the values f(E,p)*dE*dp should be proportional
        to the number of particles in the phase space volume dE*dp.

    dE : array of shape (NE,)
        Width of the energy bins (keV). If not given, the width will be inferred from `E_axis`
        when sampling the distribution.

    d_pitch : array of shape (NP,)
        Width of the pitch bins. If not given, the width will be inferred from `E_axis`
        when sampling the distribution.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, E_axis, pitch_axis, dist, particle, density=None, dE=None, d_pitch=None, ref_dir=[0,1,0]):
        """Create tabulated energy-pitch distributions from given data.

        Parameters
        ----------
        E_axis : array-like of length NE
            The energy values (keV) of the tabulated distribution.

        pitch_axis : array-like of length NP
            The pitch values of the tabulated distribution

        dist : array-like of shape (NE,NP)
            The distribution value at the tabulated (energy,pitch)-values. The absolute normalization 
            of the distribution is not important, but the values f(E,p)*dE*dp should be proportional
            to the number of particles in the phase space volume dE*dp.

        dE : array of shape (NE,)
            Width of the energy bins (keV). If not given, the width will be inferred from `E_axis`
            when sampling the distribution.

        d_pitch : array of shape (NP,)
            Width of the pitch bins. If not given, the width will be inferred from `pitch_axis`
            when sampling the distribution."""

        super().__init__(particle, density=density, v_collective=None, ref_dir=ref_dir)
        
        self.E_axis = np.array(E_axis)
        self.pitch_axis = np.array(pitch_axis)
        self._set_2d_dist(dist)

        self.dE = dE
        self.d_pitch = d_pitch

    def sample_energy_pitch(self, n, index=0):
        """Sample energies (keV) and pitch values from the tabluated distribution."""
        
        sample = sampler.sample_tab(self.dist[index], self.E_axis, self.pitch_axis, 
                                    dx=[self.dE, self.d_pitch], n_samples=n)

        E = sample[0]
        pitch = sample[1]

        return E, pitch


class TabulatedVparVperpDistribution(VparVperpDistribution):
    """Velocity distribution represented by tabulated values of f(v_par, v_perp).

    Attributes
    ----------
    v_par_axis : array of shape (Nvpa,)
        The v_par axis of the tabulated distribution (m/s).

    v_perp_axis : array of shape (Nvpe,)
        The v_perp values of the tabulated distribution (m/s).

    dist : array of shape (Nvpa,Nvpe) or (N,Nvpa,Nvpe)
        The distribution value at the tabulated (v_par,v_perp)-values. The absolute normalization 
        of the distribution is not important, but the values f(v_par,v_perp)*dv_par*dv_perp should be proportional
        to the number of particles in the phase space volume dv_par*dv_perp.

    dv_par : array of shape (Nvpa,)
        Width of the v_par bins (m/s).

    dv_perp : array of shape (Nvpe,)
        Width of the v_perp bins (m/s).

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, v_par_axis, v_perp_axis, dist, particle, density=None, dv_par=None, dv_perp=None, ref_dir=[0,1,0]):

        super().__init__(particle, density=density, v_collective=None, ref_dir=ref_dir)
        
        self.v_par_axis = np.array(v_par_axis)
        self.v_perp_axis = np.array(v_perp_axis)
        self._set_2d_dist(dist)

        self.dv_par = dv_par
        self.dv_perp = dv_perp

    def sample_vpar_vperp(self, n, index=0):
        """Sample parallel and perpendicular speeds (m/s) from the tabluated distribution."""
        
        sample = sampler.sample_tab(self.dist[index], self.v_par_axis, self.v_perp_axis, 
                                    dx=[self.dv_par, self.dv_perp], n_samples=n)

        v_par = sample[0]
        v_perp = sample[1]
        
        return v_par, v_perp


class TabulatedSpeedDistribution(SpeedDistribution):
    """A class representing a velocity distribution where the speed distribution
    is given by tabulated values of f vs v.

    Attributes
    ----------
    v_axis : array of shape (N,)
        The speed values (m/s) of the tabulated distribution.
    
    dist : array of shape (N,)
        The distribution value at the tabulated speeds. The absolute normalization 
        of the distribution is not important, but the values f(v)*dv should be proportional
        to the number of particles in the energy interval dE.

    For the rest of the attributes see docstring of the parent class(es)."""

    def __init__(self, v_axis, dist, particle, density=None, pitch_range=[-1,1], ref_dir=[0,1,0]):
        
        super().__init__(particle, density=density, v_collective=None, 
                         pitch_range=pitch_range, ref_dir=ref_dir)

        self.v_axis = np.array(v_axis)
        self._set_1d_dist(dist)
    
    def sample_speed(self, n, index=0):
        """Sample speeds (in m/s) from tabulated speed distribution."""
        
        n = int(n)
        v = sampler.sample_tab(self.dist[index], self.v_axis, n_samples=n)

        return v
