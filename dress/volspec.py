"""Tools for calculating spatially resolved or or volume integrated 
spectra with the `dress` framework."""

import numpy as np


class VolumeElement:
    """A class representing a volume element from which we want to 
    compute reacion product spectra.

    Parameters
    ----------
    dV : float
        Volume of the volume element (m**3)

    pos : tuple of floats
        Spatial coordinates of the volume element, e.g. (R,Z) or (X,Y,Z).
        Not necessary for the spectrum calculations to work but could be useful
        for the user when plotting etc.

    ems_dir : array of shape (3,1) or None
        The emission direction along which to evaluate the spectrum. 
        `None` means to sample emission directions randomly in 4*pi.

    ref_dir : array of shape(3,1)
        If calculating spectra in 2D, the emission angle is given relative to this direction.

    solid_angle : float
        Solid angle (in steradians) in which particles are emitted.

    dist_a, dist_b : instances of dress.dists.VelocityDistribution
        Distribution of the first and second reactants, respectively."""

    def __init__(self, dV):
        """Create volume element with given volume `dV` (m**3)."""

        self.dV = dV
        
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
    vols : list of N instances of dress.volspec.VolumeElement
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
    n_vols = len(vols)

    for i,vol in enumerate(vols):

        if not quiet:
            print(f'Progress: {100*i/n_vols:.2f}%', end='\r')

        # Spectrum from current volume element (particles/bin/m**3/s)
        s = calc_single_vol(vol, spec_calc, bins=bins, **kwargs)

        if integrate:
            # Compute volume integrated spectrum (with units particles/bin/s)
            spec = spec + s*vol.dV
        else:
            # Compute spatially resolved spectrum (with units particles/bin/m**3/s)
            spec[i] = s

    return spec

        

def calc_single_vol(vol, spec_calc, **kwargs):
    """Calculate spectrum from a given volume element.

    Parameters
    ----------
    vol : nstances of dress.volspec.VolumeElement
        The volume element to calculate the spectrum from.

    spec_calc : instance of dress.spec.SpectrumCalculator
        The spectrum calculator to apply to the volume element.

    All additional keyword arguments are passed to `spectrum_calculator.__call__`.

    Returns
    -------
    spec : array
        The calculated spectrum (units are particles/bin/m**3/s)."""

    
    if (spec_calc.reaction.a != vol.dist_a.particle or 
        spec_calc.reaction.b != vol.dist_b.particle):
        raise ValueError('Reactants and distribution species do not match')


    # Sample reactant distributions
    spec_calc.reactant_a.v = vol.dist_a.sample(spec_calc.n_samples)
    spec_calc.reactant_b.v = vol.dist_b.sample(spec_calc.n_samples)
    
    # Calculate spectrum along the requested emission direction
    spec_calc.u = vol.ems_dir
    spec_calc.ref_dir = vol.ref_dir
    
    na = vol.dist_a.density
    nb = vol.dist_b.density
    δab = get_delta(vol.dist_a, vol.dist_b)
    ΔΩ = vol.solid_angle

    n_samples = spec_calc.n_samples
        
    spec_calc.weights = na*nb*ΔΩ/(1 + δab) * np.ones(n_samples) / n_samples

    spec = spec_calc(**kwargs)

    return spec


def get_delta(dist_a, dist_b):
    
    if dist_a == dist_b:
        delta = 1
    else:
        delta = 0

    return delta


def _get_empty_spec(vols, bins, integrate):
    """Create empty spectrum array of the appropriate size."""
    
    # Make `vols` and `bins` into 1-element lists, if necessary
    if type(vols) is not list:
        vols = [vols]
        
    if type(bins) is not list:
        bins = [bins]

    # Check if we should have spatial resolution or not
    if integrate:
        nvols = 1
    else:
        nvols = len(vols)
    
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
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    import dress
    import dists

    # Create spectrum calculator
    dt = dress.reactions.DTNHe4Reaction()
    scalc = dress.SpectrumCalculator(dt)

    # Create a couple of volume elements
    dV = 1e-6
    vols = []
    
    for i in range(5):
        vol = VolumeElement(dV)
        vol.dist_a = dists.MonoEnergeticDistribution(200.0, scalc.reactant_a, 1e19, pitch_range=[-0.1, 0.1])
        vol.dist_b = dists.MaxwellianDistribution(10, scalc.reactant_b, 1e19)
        #vol.dist_b.v_collective = [0,1e6,0]

        vol.ems_dir = [0,0,1]
        vol.ref_dir = [0,1,0]
        vols.append(vol)

    # Calculate neutron spectrum
    E_bins = np.arange(11e3, 18e3, 50.0)
    A_bins = np.linspace(-1,1,25)
    #bins = [E_bins, A_bins]
    bins = E_bins
    spec = calc_vols(vols, scalc, bins, integrate=False)
