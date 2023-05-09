"""Useful methods for performing `dress` calculations."""

import numpy as np

from dress import dists
from dress import volspec


def make_dist(dist_type, NP, **kwargs):
    """Create distribution from the given input data.

    Parameters
    ----------
    dist_type : str
        The type of dist. Should be one of ('maxwellian', 'mono-energetic', 'energy', 
        'speed', 'energy-pitch', 'vpar-vperp')

    particle_name : str
        Name of the particle (e.g. 'd', 't, 'he4' etc). See `dress.reactions.particle.py`
        for all valid particle names.

    NP : int
        Number of spatial points covered by the distribution


    General keyword arguments
    -------------------------
    density : array of shape (NP,)
        Density (particles/m**3) at each of the NP spatial points
    
    v_collective : array-like of shape (NP,3)
        Collective velocity (m/s) at each of the NP spatial points. 
        Default is no collective velocity.


    Keyword arguments for `dist_type = 'maxwellian'`
    ---------------------------------------------
    temperature : array-like of shape (NP,)
        Temperature (keV) at each of the NP spatial points

    pitch_range : array-like of shape (NP,2)
        Range of (uniformly distributed) pitch values at each of the NP spatial points.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].


    Keyword arguments for `dist_type = 'mono-energetic'`
    ----------------------------------------------------
    beam_energy : array of shape (NP,)
        Beam energy at each of the NP spatial points

    pitch_range : array-like of shape (NP,2)
        Range of (uniformly distributed) pitch values at each of the NP spatial points.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].


    Keyword arguments for `dist_type = 'energy'`
    --------------------------------------------
    energy_axis : array-like of shape (NE)
        Energy axis of the tabulated distribution

    dist : array of shape (NP,NE)
        Tabulated energy distribution at each of the NP spatial points.
    
    pitch_range : array-like of shape (NP,2)
        Range of (uniformly distributed) pitch values at each of the NP spatial points.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].


    Keyword arguments for `dist_type = 'speed'`
    -------------------------------------------
    speed_axis : array-like of shape (NP,NV)
        Speed axis of the tabulated distribution at each of the NP spatial points

    dist : array of shape (NP,NV)
        Tabulated speed distribution at each of the NP spatial points.
    
    pitch_range : array-like of shape (NP,2)
        Range of (uniformly distributed) pitch values at each of the NP spatial points


    Keyword arguments for `dist_type = 'energy-pitch'`
    --------------------------------------------------
    energy_axis : array-like of shape (NE,)
        Energy axis of the tabulated distribution

    pitch_axis : array-like of shape (Npitch,)
        Pitch axis of the tabulated distribution

    dist : array of shape (NP,NE,Npitch)
        Tabulated energy-pitch distribution at each of the NP spatial points.


    Keyword arguments for `dist_type = 'vpar-vperp'`
    --------------------------------------------------
    vpar_axis : array-like of shape (Nvpar,)
        v_parallel axis of the tabulated distribution

    vperp_axis : array-like of shape (Nvperp,)
        v_perpendicular axis of the tabulated distribution

    dist : array of shape (NP,Nvpar,Nvperp)
        Tabulated vpar-vperp distribution at each of the NP spatial points.


    Returns
    -------
    dist : dress.VelocityDistribution
        The resulting distribution in `dress` format."""

    particle = 
    NP = int(NP)
    density = np.atleast_1d(kwargs.get('density'))
    v_collective = np.atleast_1d(kwargs.get('v_collective', np.zeros((NP,3)) ) )

    dist_type = dist_type.lower()

    if dist_type == 'maxwellian':
        T = np.atleast_1d(kwargs.get('temperature'))

        if not (NP,) == T.shape == density.shape:
            raise ValueError('Input array(s) incompatible with the given number of spatial points')

        
