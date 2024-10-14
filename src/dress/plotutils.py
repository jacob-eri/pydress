"""Tools for plotting spectra and distributions in the `dress` framework."""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

    dV : array
        Volume elements, if spec is spatially resolved (e.g. produced with 
        `dress.utils.calc_vols` using `integrate=False`)

    dynamic_range : int
        Number of orders of magnitude spanned by the spectrum axis. Default is 5.

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
        the assumption that the input energy axis is in keV (the default units of pydress). 
        Default is True.

    erase : bool
        Whether to erase existing spectra from the figure. Default is `True`.
    """

    # Get keyword arguments
    dV = kwargs.get('dV', None)
    dyn_range = kwargs.get('dynamic_range', 5)
    E_label = kwargs.get('E_label', 'E')
    A_label = kwargs.get('A_label', 'cos(θ)')
    spec_label = kwargs.get('spec_label', 'events/bin/s')
    label = kwargs.get('label', None)
    E_unit = kwargs.get('E_unit', 'MeV')
    convert_E = kwargs.get('convert_E', True)
    erase = kwargs.get('erase', True)

    # Integrate over volume, if necessary
    if dV is not None:
        spec = np.sum(spec*dV[:,None], axis=0)

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
        plt.ylim(bottom=spec.max()/10**dyn_range)
        plt.legend()
        
    if len(bin_edges) == 2:
        plt.figure('DRESS energy-pitch spectrum')
        if erase: plt.clf()
        A_bins = bin_edges[1]
        plt.pcolor(E_bins, A_bins, spec.T)
        plt.xlabel(E_label)
        plt.ylabel(A_label)
        plt.clim(vmin=spec.max()/10**dyn_range)
        plt.colorbar(label=spec_label)
        
    
def plot_emissivity(pos, spec, *bin_edges, **kwargs):
    """Plot 2D spatial emissivity from given volume elements.

    Parameters
    ----------
    pos : tuple
        The position coordinates (e.g. (R,Z)) where the spectrum is given.
    
    spec : spectrum array
        This should be a spectrum of the kind calculated with dress.volspec.calc_vols, 
        using the `integrate=False` option.

    bin_edges : arrays
        Energy bins and (optionally) pitch bins for the product particle.

    x_label, y_label : str
        Label for the spatial x and y- axes.

    ems_label : str
        Label for the emissivity axis. Default is "events/m³/s"."""

    if len(pos) != 2:
        raise ValueError('Emissivity plot currently only works with 2D profiles')

    # Keyword arguments
    x_label = kwargs.get('x_label', 'x')
    y_label = kwargs.get('y_label', 'y')
    ems_label = kwargs.get('ems_label', 'events/m³/s')

    # Extract data to plot
    x = np.atleast_1d(pos[0])
    y = np.atleast_1d(pos[1])

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
    

def plot_dist_point(dist, i_spatial=0, n_samples=100_000, dist_type='energy-pitch', 
                    log_dist= False, bins=50):
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

    dist_type : str
        How to visualize the distribution. Can be one of 
            - `energy-pitch`
            - `vpar-vperp`
            - `energy`
            - `speed`

    log_dist : bool
        Whether to plot dist in log scale (default is False).

    bins : various
        Bins specification as expected by np.histogram and np.histogram2d."""

    d = _extract_dist(dist, i_spatial=i_spatial, n_samples=n_samples, 
                      dist_type=dist_type, bins=bins)

    # Plot settings
    if log_dist:
        norm = LogNorm()
    else:
        norm = None

    if dist_type == 'energy-pitch':
        plt.figure('energy-pitch dist')
        plt.clf()
        plt.pcolor(d[1], d[2], d[0].T, norm=norm)
        plt.ylim(-1,1)
        plt.xlabel('Energy (keV)')
        plt.ylabel('v$_{\parallel}$/v (m/s)')
        plt.colorbar()

    elif dist_type == 'energy':
        plt.figure('energy dist')
        plt.clf()
        x = 0.5*(d[1][1:] + d[1][:-1])
        w = np.diff(d[1])
        plt.bar(x, d[0], width=w, log=log_dist)
        plt.xlabel('Energy (keV)')

    elif dist_type == 'vpar-vperp':
        plt.figure('vpar-vperp dist')
        plt.clf()
        plt.pcolor(d[1], d[2], d[0].T, norm=norm)
        plt.xlabel('v$_{\parallel}$ (m/s)')
        plt.ylabel('v$_{\perp}$ (m/s)')
        plt.ylim(bottom=0)
        plt.axis('equal')
        plt.colorbar()

    elif dist_type == 'speed':
        plt.figure('speed dist')
        plt.clf()
        x = 0.5*(d[1][1:] + d[1][:-1])
        w = np.diff(d[1])
        plt.bar(x, d[0], width=w, log=log_dist)
        plt.xlabel('Speed (m/s)')


def _extract_dist(dist, i_spatial=1, n_samples=100_000, dist_type='energy-pitch', bins=50):
    """Sample distribution at the given spatial point 
    and generate histogram in the requested variables.

    Possible choices for `dist_type` are
        - `energy-pitch`
        - `vpar-vperp`
        - `energy`
        - `speed`"""

    # Sample velocities from the distribution
    vel = dist.sample(n_samples, index=i_spatial)
    density = dist.density[i_spatial]

    # Convert to the appropriate coordinates
    m = dist.particle.m     # mass i keV/c**2

    if dist_type == 'energy-pitch':
        E = relkin.get_energy(vel, m)
        ref_dir = dist.ref_dir[:,i_spatial]
        v_par = np.sum(vel*ref_dir[:,None], axis=0)
        v = np.sqrt(vec_ops.dot(vel, vel))
        pitch = v_par / v
        h = np.histogram2d(E, pitch, density=True, bins=bins)
        
    elif dist_type == 'energy':
        E = relkin.get_energy(vel, m)
        h = np.histogram(E, density=True, bins=bins)
        
    elif dist_type == 'vpar-vperp':
        ref_dir = dist.ref_dir[:,i_spatial]
        v_par = np.sum(vel*ref_dir[:,None], axis=0)
        vel_par = ref_dir[:,None] * v_par[None,:]
        vel_perp = vel - vel_par
        v_perp = np.sqrt(vec_ops.dot(vel_perp, vel_perp))
        h = np.histogram2d(v_par, v_perp, density=True, bins=bins)

    elif dist_type == 'speed':
        v = np.sqrt(vec_ops.dot(vel, vel))
        h = np.histogram(v, density=True, bins=bins)

    d = h[0] * density
    
    return (d, *h[1:])

        
def explore_dist(dist, **kwargs):
    """Interactive distribution plots.

    The particle density as a function of posistion is plotted.
    By clocking in this figure the user can plot the velocity distribution
    at the different points.

    Keyword arguments are passed to plot_dist_point."""

    if dist.pos is None:
        raise ValueError('Must set pos attribute')

    if len(dist.pos) > 2:
        print('Only 2D plotting supported so far. Plotting density projected on first two dimensions.')

    # Plot density
    density_fig = plt.figure('density')
    plt.tripcolor(dist.pos[0], dist.pos[1], dist.density)
    plt.axis('equal')

    click_fun = lambda event: _density_click_fun(event, dist, **kwargs)
    cid = density_fig.canvas.mpl_connect('button_press_event', click_fun)    # connect to event manager

    return density_fig, click_fun
   
 
def _density_click_fun(event, dist, **kwargs):
    """Determine what happens after clicking in the density plot."""

    # Find the spatial point closest to the click
    x_click = event.xdata
    y_click = event.ydata

    distance = np.sqrt( (x_click - dist.pos[0])**2 + (y_click - dist.pos[1])**2 )

    i_click = np.argmin(distance)

    plot_dist_point(dist, i_spatial=i_click, **kwargs)
    plt.draw()
