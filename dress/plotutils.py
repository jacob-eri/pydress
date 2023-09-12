"""Tools for plotting spectra and distributions in the `dress` framework."""

import matplotlib.pyplot as plt
import numpy as np


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
    

def plot_dist_point(dist, i_spatial=1, **kwargs):
    """Plot distribution at given spatial location."""
    pass
