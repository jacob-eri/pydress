"""Tools for calculating spatially resolved or volume integrated 
spectra with the `dress` framework."""

import numpy as np

import dress.vec_ops as vec

class VolumeElements:
    """A class representing volume elements from which we want to 
    compute reacion product spectra.

    Parameters
    ----------
    dV : array
        Volumes of each volume element (m**3)

    pos : tuple of arrays
        Spatial coordinates of each volume element, e.g. (R,Z) or (X,Y,Z).
        Not necessary for the spectrum calculations to work but could be useful
        for the user when plotting etc.

    ems_dir : array of shape (N,3) or None
        The emission direction along which to evaluate the spectrum. 
        `None` means to sample emission directions randomly in 4*pi.

    ref_dir : array of shape (N,3)
        If calculating spectra in 2D (i.e. resolved in both energy and emission direction), 
        the emission angle is given relative to this direction.

    solid_angle : array of shape (N,)
        Solid angle (in steradians) in which particles are emitted, for each volume element.

    dist_a, dist_b : instances of dress.dists.VelocityDistribution
        Distribution of the first and second reactants, respectively. Needs to be spatially
        resolved on the same grid as the volume elements, i.e. sampling the dists with the 
        keywords `index=i` should return a sample from volume element `i`."""

    def __init__(self, dV):
        """Create volume elements with given volumes `dV` (m**3)."""

        self.dV = np.atleast_1d(dV)
        self.nvols = len(dV)
        
        # Default attributes
        self.pos = None
        self.ems_dir = None
        self.ref_dir = [0,1,0]
        self.solid_angle = 4*np.pi
        self.dist_a = None
        self.dist_b = None

    @property
    def ems_dir(self):
        return self._ems_dir

    @ems_dir.setter
    def ems_dir(self, u):      

        if u is None or np.all(u == None):
            # Isotropic emission from all volume elements
            self._ems_dir = np.repeat(None, self.nvols)

        else:
            self._ems_dir = self._prep_vector_attribute(u)

    @property
    def ref_dir(self):
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, u):
        self._ref_dir = self._prep_vector_attribute(u)
        
    def _prep_vector_attribute(self, v):
        """Put vector attribute (such as `ems_dir` and `ref_dir`) into the appropriate formats."""
        v = vec.make_vector(v)
            
        if v.shape[1] == 1:
            # Same vector for all volume elements
            v = vec.repeat(v, self.nvols)

        if v.shape[1] != self.nvols:
            raise ValueError('Number of vectors and volume elements do not match')

        # Transpose the array so that v[i] is the vector for element i
        v = v.T

        return v

    def _prep_scalar_attribute(self, x):
        """Put scalar attribute (such as `solid_angle`) into the appropriate format."""

        x = np.atleast_1d(x)
        
        if len(x) == 1:
            # Same value for all volume elements
            x = np.repeat(x, self.nvols)

        if len(x) != self.nvols:
            raise ValueError('Number values and volume elements do not match')

        return x

    @property
    def solid_angle(self):
        return self._solid_angle

    @solid_angle.setter
    def solid_angle(self, omega):
        self._solid_angle = self._prep_scalar_attribute(omega)


    @property
    def dist_a(self):
        return self._dist_a

    @dist_a.setter
    def dist_a(self, dist):
        
        if dist is not None and dist.n_spatial != self.nvols:
            raise ValueError('Wrong number of spatial positions for `dist_a`')

        self._dist_a = dist

    @property
    def dist_b(self):
        return self._dist_b

    @dist_b.setter
    def dist_b(self, dist):
        
        if dist is not None and dist.n_spatial != self.nvols:
            raise ValueError('Wrong number of spatial positions for `dist_b')

        self._dist_b = dist
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    import dress
    import dists
    import vec_ops as vec

    # Create spectrum calculator
    dt = dress.reactions.DTNHe4Reaction()
    scalc = dress.SpectrumCalculator(dt)

    # Create a couple of volume elements
    nvols = 5
    dV = 1e-6*np.ones(nvols)
    
    vols = VolumeElements(dV)
    vols.dist_a = dists.MonoEnergeticDistribution(200.0*np.ones(nvols), scalc.reactant_a, 1e19*np.ones(nvols), pitch_range=[-0.1, 0.1])
    vols.dist_b = dists.MaxwellianDistribution(10*np.ones(nvols), scalc.reactant_b, 1e19*mp.ones(nvols))
    #vol.dist_b.v_collective = [0,1e6,0]

    vols.ems_dir = vec.repeat([0,0,1], nvols)
    vols.ref_dir = vec.repeat([0,1,0], nvols)

    # Calculate neutron spectrum
    E_bins = np.arange(11e3, 18e3, 50.0)
    A_bins = np.linspace(-1,1,25)
    #bins = [E_bins, A_bins]
    bins = E_bins
    spec = calc_vols(vols, scalc, bins, integrate=True)
