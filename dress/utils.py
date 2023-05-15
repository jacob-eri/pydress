"""Useful methods for performing `dress` calculations."""

import numpy as np

from dress import dists
from dress import volspec
from dress.reactions.particle import Particle

def make_dist(dist_type, particle_name, NP, density, **kwargs):
    """Create distribution from the given input data.

    Parameters
    ----------
    dist_type : str
        The type of dist. Should be one of ('maxwellian', 'energy', 'speed', 
        'energy-pitch', 'vpar-vperp')

    particle_name : str
        Name of the particle (e.g. 'd', 't, 'he4' etc). See `dress.reactions.particle.py`
        for all valid particle names.

    NP : int
        Number of spatial points covered by the distribution

    density : array of shape (NP,)
        Density (particles/m**3) at each of the NP spatial points

    All distributions require the above input. Additionally, depending on the requested distribution type,
    some of the following keyword arguments should be supplied:

    General keyword arguments
    -------------------------    
    v_collective : array-like of shape (NP,3)
        Collective velocity (m/s) at each of the NP spatial points. 
        Default is no collective velocity.

    ref_dir : array-like with shape (NP,3)
        Reference direction at each of the NP spatial points.
        Pitch values of the particles are given relative to this direction.
        Does not have to be normalized to unity.
        Default is [0,1,0].


    Keyword arguments for `dist_type = 'maxwellian'`
    ---------------------------------------------
    temperature : array-like of shape (NP,)
        Temperature (keV) at each of the NP spatial points

    pitch_range : array-like of shape (2,)
        Range of (uniformly distributed) pitch values.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].


    Keyword arguments for `dist_type = 'energy'`
    --------------------------------------------
    energy_axis : array-like of shape (NE)
        Energy axis of the tabulated distribution

    distvals : array of shape (NP,NE)
        Tabulated energy distribution at each of the NP spatial points.
    
    pitch_range : array-like of shape (2,)
        Range of (uniformly distributed) pitch values.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].

    
    Keyword arguments for `dist_type = 'speed'`
    --------------------------------------------
    speed_axis : array-like of shape (Nv)
        Speed axis of the tabulated distribution

    distvals : array of shape (NP,Nv)
        Tabulated speed distribution at each of the NP spatial points.
    
    pitch_range : array-like of shape (2,)
        Range of (uniformly distributed) pitch values.
        Default is isotropically distributed velocities, i.e. pitch_range = [-1,1].


    Keyword arguments for `dist_type = 'energy-pitch'`
    --------------------------------------------------
    energy_axis : array-like of shape (NE,)
        Energy axis of the tabulated distribution

    pitch_axis : array-like of shape (Npitch,)
        Pitch axis of the tabulated distribution

    distvals : array of shape (NP,NE,Npitch)
        Tabulated energy-pitch distribution at each of the NP spatial points.


    Keyword arguments for `dist_type = 'vpar-vperp'`
    --------------------------------------------------
    vpar_axis : array-like of shape (Nvpar,)
        v_parallel axis of the tabulated distribution

    vperp_axis : array-like of shape (Nvperp,)
        v_perpendicular axis of the tabulated distribution

    distvals : array of shape (NP,Nvpar,Nvperp)
        Tabulated vpar-vperp distribution at each of the NP spatial points.


    Returns
    -------
    dist : dress.dists.VelocityDistribution
        The resulting distribution in `dress` format."""

    
    # Massage the input a bit
    dist_type = dist_type.lower()
    particle = Particle(particle_name)
    NP = int(NP)
    density = np.atleast_1d(density)
    if not density.shape == (NP,): raise ValueError(f'Wrong shape of `density`, should be ({NP},)')

    # Get keyword arguments
    default_v_collective = np.zeros((NP,3))
    v_collective = _get_dist_kwarg(kwargs, 'v_collective', (NP,3), default_v_collective)
    v_collective = v_collective.T     # make correct shape for dress

    default_ref_dir = np.repeat(np.atleast_2d([0,1,0]), NP, axis=0)
    ref_dir = _get_dist_kwarg(kwargs, 'ref_dir', (NP,3), default_ref_dir)
    ref_dir = ref_dir.T     # make correct shape for dress

    default_pitch_range = [-1,1]
    pitch_range = _get_dist_kwarg(kwargs, 'pitch_range', (2,), default_pitch_range)

    temperature = _get_dist_kwarg(kwargs, 'temperature', (NP,), None)
    energy_axis = _get_dist_kwarg(kwargs, 'energy_axis', None, None)
    speed_axis = _get_dist_kwarg(kwargs, 'speed_axis', None, None)
    pitch_axis = _get_dist_kwarg(kwargs, 'pitch_axis', None, None)
    vpar_axis = _get_dist_kwarg(kwargs, 'vpar_axis', None, None)
    vperp_axis = _get_dist_kwarg(kwargs, 'vpar_axis', None, None)

    # Create the requested distribution
    if dist_type == 'maxwellian':
        dist = dists.MaxwellianDistribution(temperature, particle, density=density, v_collective=v_collective,
                                            pitch_range=pitch_range, ref_dir=ref_dir)

    elif dist_type == 'energy':
        if energy_axis is None: raise ValueError('Must provide `energy_axis` keyword')
        NE = len(energy_axis)

        distvals = _get_dist_kwarg(kwargs, 'distvals', (NP,NE), None)
        if distvals is None: raise ValueError('Must provide `distvals` keyword')
        
        dist = dists.TabulatedEnergyDistribution(energy_axis, distvals, particle, density=density,
                                                 pitch_range=pitch_range, ref_dir=ref_dir)

    elif dist_type == 'speed':
        if speed_axis is None: raise ValueError('Must provide `speed_axis` keyword')
        Nv = len(speed_axis)

        distvals = _get_dist_kwarg(kwargs, 'distvals', (NP,Nv), None)
        if distvals is None: raise ValueError('Must provide `distvals` keyword')
        
        dist = dists.TabulatedSpeedDistribution(speed_axis, distvals, particle, density=density,
                                                pitch_range=pitch_range, ref_dir=ref_dir)

    elif dist_type == 'energy-pitch':
        if energy_axis is None: raise ValueError('Must provide `energy_axis` keyword')
        NE = len(energy_axis)

        if pitch_axis is None: raise ValueError('Must provide `pitch_axis` keyword')
        Npitch = len(pitch_axis)

        distvals = _get_dist_kwarg(kwargs, 'distvals', (NP,NE,Npitch), None)
        if distvals is None: raise ValueError('Must provide `distvals` keyword')
        
        dist = dists.TabulatedEnergyPitchDistribution(energy_axis, pitch_axis, distvals, particle, 
                                                      density=density, ref_dir=ref_dir)

    elif dist_type == 'vpar-vperp':
        if vpar_axis is None: raise ValueError('Must provide `vpar_axis` keyword')
        Nvpar = len(vpar_axis)

        if vperp_axis is None: raise ValueError('Must provide `vperp_axis` keyword')
        Nvperp = len(vperp_axis)

        distvals = _get_dist_kwarg(kwargs, 'distvals', (NP,Nvpar,Nvperp), None)
        if distvals is None: raise ValueError('Must provide `distvals` keyword')
        
        dist = dists.TabulatedVparVperpDistribution(vpar_axis, vperp_axis, distvals, particle, 
                                                    density=density, ref_dir=ref_dir)

    else:
        raise ValueError(f'dist_type = {dist_type} is not a valid option')


    return dist


def make_vols(dV, dist_a, dist_b, solid_angle, emission_dir=None, ref_dir=None, pos=None):
    """Make volume elements for dress.

    Make a collection of volume elements from which volume intergated or 
    spatially resolved spectrum calculations - between reactants from the 
    distributions `dist_a` and `dist_b` - can be performed.

    Parameters
    ----------
    dV : array-like of shape (NP,)
        Element volumes (in m**3).

    dist_a, dist_b : dress.dists.VelocityDistribution
        Reactant distributions to sample from when computing spectra.

    solid_angle : array-lilke of shape (NP,)
        Solid angle that the reaction products of interest are emitted into
        (typically defined by the viewing geometry, collimation etc).

    emission_dir : array-like of shape (NP,3) or None (default)
        Reaction product emission direction for each volume element. Does not 
        have to be normalized to unity. emission_dir = None (default) means 
        isotropic emission in 4*pi.

    ref_dir : array-like of shape (NP,3)
        Reference direction for the emitted reaction products. Only needs to be
        provided if the spectrum should be resolved with respect to the emission angle.

    pos : array or tuple of arrays
        Spatial coordinates of each volume element, e.g. (x, y, z), (R, Z) or similar.
        Not required for the dress calculations, but can be handy for plotting etc.

    
    Returns
    -------
    vols : dress.volspec.VolumeElement
        The resulting volume elements."""

    # Check the input
    NP = dist_a.n_spatial

    if dist_b.n_spatial != NP: 
        raise ValueError('Number of spatial points in `dist_a` and `dist_b` do not match')

    dV = np.atleast_1d(dV)
    if dV.shape != (NP,): 
        raise ValueError('`dV` is incompatible with the number of spatial points in the distributions')

    solid_angle = np.atleast_1d(solid_angle)
    if solid_angle.shape != (NP,): 
        raise ValueError('`solid_angle` is incompatible with the number of spatial points in the distributions')
    
    if emission_dir is not None:
        emission_dir = np.atleast_2d(emission_dir)
        
        if emission_dir.shape != (NP,3):
            raise ValueError('Wrong shape of `emission_dir`')

    if ref_dir is None:
        # User does not intend to resolve spectra with respect to 
        # emission direction -> ref_dir is arbitrary
        ref_dir = np.array([0,1,0])
        ref_dir = np.repeat(ref_dir, NP, axis=0)

    else:
        ref_dir = np.atleast_2d(ref_dir)
        
    if ref_dir.shape != (NP,3):
        raise ValueError('Wrong shape of `ref_dir`')

    # Create volume elements
    vols = volspec.VolumeElements(dV)
    vols.dist_a = dist_a
    vols.dist_b = dist_b
    vols.solid_angle = solid_angle
    vols.ems_dir = emission_dir
    vols.ref_dir = ref_dir
    vols.pos = pos

    return vols


def _get_dist_kwarg(kwargs, kw, required_shape, default_val):
    var = kwargs.get(kw, default_val)
    
    if var is not None:
        var = np.atleast_1d(var)

        if (required_shape is not None) and (var.shape != required_shape):
            raise ValueError(f'Shape of {kw} is {var.shape}, but it should be {required_shape}')

    return var


if __name__ == '__main__':
    
    NP = 1
    density = np.repeat(1e19, NP)
    particle_name = 'd'

    # Make a Maxwellian distribution
    T = np.repeat(10, NP)
    maxwellian_dist = make_dist('maxwellian', particle_name, NP, density, temperature=T)

    # Make tabulated energy distribution
    Eaxis = np.linspace(0, 1000, 500)
    Edist = np.zeros_like(Eaxis)
    Edist[:250] = Eaxis[:250]
    Edist[250:] = Eaxis[-1] - Eaxis[250:]
    Edist = np.repeat(np.atleast_2d(Edist), NP, axis=0)
    energy_dist = make_dist('energy', particle_name, NP, density, energy_axis=Eaxis,
                            distvals=Edist, pitch_range=[-0.5,0.5])

    # Make tabulated speed distribution
    v_axis = np.linspace(0, 1e7, 500)
    v_dist = np.zeros_like(v_axis)
    v_dist[:250] = v_axis[:250]
    v_dist[250:] = v_axis[-1] - v_axis[250:]
    v_dist = np.repeat(np.atleast_2d(v_dist), NP, axis=0)
    speed_dist = make_dist('speed', particle_name, NP, density, speed_axis=v_axis,
                            distvals=v_dist, pitch_range=[-0.5,0.5])
