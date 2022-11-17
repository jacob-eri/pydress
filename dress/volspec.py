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
        self.solid_angle = 4*np.pi
        self.dist_a = None
        self.dist_b = None


def calc(vols, spec_calc, bins, integrate=True, **kwargs):
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

    All additional keyword arguments are passed to the spectrum_calculator__call__.

    Returns
    -------
    spec : array
        If integrate = True this will be a 1D array representing the volume integrated 
        spectrum histogram. If integrate = False this will be a 2D array with N rows, 
        such that spec[i] is the spectrum from vols[i]."""

    if 
    for vol in vols:
        
