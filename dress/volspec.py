"""Tools for calculating spatially resolved or volume integrated 
spectra with the `dress` framework."""

import numpy as np


class VolumeElements:
    """A class representing a volume elements from which we want to 
    compute reacion product spectra.

    Parameters
    ----------
    dV : array
        Volumes of each volume element (m**3)

    pos : array of shape (3,N)
        Spatial coordinates of each volume element, e.g. (R,Z) or (X,Y,Z).
        Not necessary for the spectrum calculations to work but could be useful
        for the user when plotting etc.

    ems_dir : array of shape (3,N) or None
        The emission direction along which to evaluate the spectrum. 
        `None` means to sample emission directions randomly in 4*pi.

    ref_dir : array of shape(3,N)
        If calculating spectra in 2D, the emission angle is given relative to this direction.

    solid_angle : array of shape (N,)
        Solid angle (in steradians) in which particles are emitted.

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


def calc_vols(vols, spec_calc, bins, integrate=True, quiet=True, **kwargs):
    """Calculate spectrum from a number of volume elements.

    Parameters
    ----------
    vols : instance of dress.volspec.VolumeElements
        The volume elements to calculate the spectra from.

    spec_calc : instance of dress.spec.SpectrumCalculator
        The spectrum calculator to apply to each volume element.

    bins : array or [array, array]
        Bin edges in energy and (optionally) in pitch of emission direction. 

    integrate : bool
        Whether to calculate the volume integrated spectrum (units of particles/bin/s)
        or a spatially resolved spectrum (units of particles/bin/m**3/s).

    All additional keyword arguments are passed to `spectrum_calculator.__call__`.

    Returns
    -------
    spec : array
        If integrate = True this will be a 1D array representing the volume integrated 
        spectrum histogram. If integrate = False this will be a 2D array with N rows, 
        such that spec[i] is the spectrum from vols[i]."""

    spec = _get_empty_spec(vols, bins, integrate)

    for i in range(vols.nvols):

        if not quiet:
            print(f'Progress: {100*i/vols.nvols:.2f}%', end='\r')

        # Spectrum from current volume element (particles/bin/m**3/s)
        s = calc_single_vol(vols, i, spec_calc, bins=bins, **kwargs)

        if integrate:
            # Compute volume integrated spectrum (with units particles/bin/s)
            spec = spec + s*vols.dV[i]
        else:
            # Compute spatially resolved spectrum (with units particles/bin/m**3/s)
            spec[i] = s

    return spec

def calc_single_vol(vols, index, spec_calc, **kwargs):
    """Calculate spectrum from a given volume element.

    Parameters
    ----------
    vols : instance of dress.volspec.VolumeElements
        The volume elements to calculate the spectrum from.

    index : int
        The index of the volume element to calculate ths spectrm from.

    spec_calc : instance of dress.spec.SpectrumCalculator
        The spectrum calculator to apply to the volume element.

    All additional keyword arguments are passed to `spectrum_calculator.__call__`.

    Returns
    -------
    spec : array
        The calculated spectrum (units are particles/bin/m**3/s)."""
    
    if (spec_calc.reaction.a != vols.dist_a.particle or 
        spec_calc.reaction.b != vols.dist_b.particle):
        raise ValueError('Reactants and distribution species do not match')

    # Sample reactant distributions
    spec_calc.reactant_a.v = vols.dist_a.sample(spec_calc.n_samples, index=index)
    spec_calc.reactant_b.v = vols.dist_b.sample(spec_calc.n_samples, index=index)
    
    # Calculate spectrum along the requested emission direction
    if vols.ems_dir is None:
        spec_calc.u = None
    else:
        spec_calc.u = vols.ems_dir[:,index]
    
    spec_calc.ref_dir = vols.ref_dir[:,index]

    na = vols.dist_a.density[index]
    nb = vols.dist_b.density[index]
    ???? = vols.solid_angle[index]
    ??ab = get_delta(vols.dist_a, vols.dist_b)

    n_samples = spec_calc.n_samples
    
    spec_calc.weights = na*nb*????/(1 + ??ab) * np.ones(n_samples) / n_samples

    spec = spec_calc(**kwargs)

    return spec

def _get_empty_spec(vols, bins, integrate):
    """Create empty spectrum array of the appropriate size."""
    
    # Make `bins` into 1-element lists, if necessary
    if type(bins) is not list:
        bins = [bins]

    # Check if we should have spatial resolution or not
    if integrate:
        nvols = 1
    else:
        nvols = vols.nvols
    
    # Number of energy bins
    nE = len(bins[0]) - 1
    
    # Check if spectrum should be resolved in emission direction as well
    if len(bins) == 2:
        nA = len(bins[1]) - 1
    else:
        nA = 1
    
    # Create empty spectrum array of correct shape
    spec = np.zeros((nvols,nE,nA))
    
    # Remove unecessary dimensions
    spec = np.squeeze(spec)

    return spec

def get_delta(dist_a, dist_b):
    
    if dist_a == dist_b:
        delta = 1
    else:
        delta = 0

    return delta
    

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
