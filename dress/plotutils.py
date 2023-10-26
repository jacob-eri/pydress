"""Tools for plotting spectra and distributions in the `dress` framework."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

from dress import relkin, vec_ops

E_conversion_factors = {}
E_conversion_factors['keV'] = 1.0
E_conversion_factors['MeV'] = 1e-3


def convert_energy(E, unit):
    try:
        cf = E_conversion_factors[unit]
    except KeyError:
        raise ValueError(f'Conversion to {unit} currently not supported')

    return E*cf


def plot_spec(spec, *bin_edges, **kwargs):
    """Plot a given spectrum.

    Parameters
    ----------
    spec : spectrum array
        Spectrum produced by a SpectrumCalculator.

    bin_edges: arrays
        Energy bins and (optionally) pitch bins for the product particle.

    *** Possible keyword arguments: ***

    E_label : str
        Label for the energy axis. Default is "E".

    A_label : str
        Label for the pitch axis, if present. Default is "cos(θ)".

    spec_label : str
        Label for the spectrum axis. Default is "spectrum (events/bin/s)".

    label : str
        Label for the legend. Default is None (no legend will be drawn).

    E_unit : str
        Which energy units to use. Default is "MeV".
    
    convert_E : bool
        Whether to convert the energy axis according to the value of `E_unit`, under 
        the assumption that the input energy axis is in keV (the default units of dress). 
        Default is True.

    erase : bool
        Whether to erase existing spectra from the figure. Default is `True`.
    """

    # Get keyword arguments
    E_label = kwargs.get('E_label', 'E')
    A_label = kwargs.get('A_label', 'cos(θ)')
    spec_label = kwargs.get('spec_label', 'events/bin/s')
    label = kwargs.get('label', None)
    E_unit = kwargs.get('E_unit', 'MeV')
    convert_E = kwargs.get('convert_E', True)
    erase = kwargs.get('erase', True)

    # Convert energy scale, if necessary
    E_bins = bin_edges[0]
    if convert_E:
        E_bins = convert_energy(E_bins, E_unit)

    # Add energy unit to the energy label
    E_label += f' ({E_unit})'

    if len(bin_edges) == 1:
        plt.figure('DRESS energy spectrum')
        if erase: plt.clf()
        plt.step(E_bins[1:], spec, where='pre', label=label)
        plt.xlabel(E_label)
        plt.ylabel(spec_label)
        plt.legend()
        
    if len(bin_edges) == 2:
        plt.figure('DRESS energy-pitch spectrum')
        if erase: plt.clf()
        A_bins = bin_edges[1]
        plt.pcolor(E_bins, A_bins, spec.T)
        plt.xlabel(E_label)
        plt.ylabel(A_label)
        plt.colorbar(label=spec_label)
        
    
def plot_emissivity(vols, spec, *bin_edges, **kwargs):
    """Plot 2D spatial emissivity from given volume elements.

    Parameters
    ----------
    vols : list of instances of dress.volspec.VolumeElement
        The volume elements for which the spectra has been calculated.
    
    spec : spectrum array
        This should be a spectrum of the kind calculated with dress.volspec.calc_vols, 
        using the `integrate=False` option.

    bin_edges : arrays
        Energy bins and (optionally) pitch bins for the product particle.

    x_label, y_label : str
        Label for the spatial x and y- axes.

    ems_label : str
        Label for the emissivity axis. Default is "events/m³/s"."""

    if len(vols.pos) != 2:
        raise ValueError('Emissivity plot currently only works with 2D profiles')

    # Keyword arguments
    x_label = kwargs.get('x_label', 'x')
    y_label = kwargs.get('y_label', 'y')
    ems_label = kwargs.get('ems_label', 'events/m³/s')

    # Extract data to plot
    x = np.atleast_1d(vols.pos[0])
    y = np.atleast_1d(vols.pos[1])

    if spec.ndim == 2:
        sum_axis = 1
    elif spec.ndim == 3:
        sum_axis = (1,2)

    ems = spec.sum(axis=sum_axis)

    plt.figure('DRESS emissivity')
    plt.clf()
    plt.tripcolor(x, y, ems)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(label=ems_label)
    plt.axis('equal')
    

def plot_dist_point(dist, i_spatial=1, n_samples=100_000, plot_type='energy-pitch'):
    """Plot distribution at a given spatial location.

    The plotting is done by sampling `n_samples` velocities and making a histogram,
    thus generating a plot that exactly represent the distribution as 'seen' by
    the spectrum calculation routines in `dress`.

    Parameters
    ----------
    dist : dress.dists.VelocityDistribution
        The distribution to plot.

    i_spatial : int
        The spatial point for which the distribution is to be plotted.

    n_samples : int
        Number of samples to draw for the histogram.

    plot_type : str
        How to visualize the distribution. Can be one of 
            - `energy-pitch`
            - `vpar-vperp`
            - `energy`
            - `speed`"""
    
    # Sample velocities from the distribution
    vel = dist.sample(n_samples, index=i_spatial)

    # Convert to the appropriate coordinates
    m = dist.particle.m     # mass i keV/c**2

    if plot_type == 'energy-pitch':
        E = relkin.get_energy(vel, m)
        ref_dir = dist.ref_dir[:,i_spatial]
        v_par = vel[0]*ref_dir[0] + vel[1]*ref_dir[1] + vel[2]*ref_dir[2]
        v = np.sqrt(vec_ops.dot(vel, vel))
        pitch = v_par / v
        
        plt.figure('energy-pitch dist')
        plt.clf()
        plt.hist2d(E, pitch, bins=50)
        plt.ylim(-1,1)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Pitch')

    elif plot_type == 'energy':
        E = relkin.get_energy(vel, m)

        plt.figure('energy dist')
        plt.clf()
        plt.hist(E, bins=50)
        plt.xlabel('Energy (keV)')

    elif plot_type == 'vpar-vperp':
        ref_dir = dist.ref_dir[:,i_spatial]
        v_par = vel[0]*ref_dir[0] + vel[1]*ref_dir[1] + vel[2]*ref_dir[2]
        vel_par = ref_dir[:,None] * v_par[None,:]
        vel_perp = vel - vel_par
        v_perp = np.sqrt(vec_ops.dot(vel_perp, vel_perp))

        plt.figure('vpar-vperp dist')
        plt.clf()
        plt.hist2d(v_par, v_perp, bins=50)
        plt.xlabel('v$_{\parallel}$ (m/s)')
        plt.ylabel('v$_{\perp}$ (m/s)')

    elif plot_type == 'speed':
        v = np.sqrt(vec_ops.dot(vel, vel))

        plt.figure('speed dist')
        plt.clf()
        plt.hist(v, bins=50)
        plt.xlabel('Speed (m/s)')
