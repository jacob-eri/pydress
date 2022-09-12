"""Classes for describing scattering reaction between alphas and d or t ions."""

import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

from dress.reactions.config import cross_section_path
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


def load_tab_cross_section(file_path):
    """Load tabulated differential cross section for alpha scattering."""

    f = open(file_path)

    # Dimensions of the data
    dims = tuple(int(n) for n in f.readline().split('\t'))

    cos_theta = np.zeros(dims[1])
    sigma_diff = np.zeros(dims)

    # Read energy axis
    E = np.array([float(e) for e in f.readline().strip().split('\t')])
    
    # Read cross section table
    for i,line in enumerate(f):
        vals = [float(val) for val in line.split('\t')]
        cos_theta[i] = vals[0]
        sigma_diff[:,i] = vals[1:]

    # Convert to the correct units
    E = E*1e3            # keV
    sigma_diff = sigma_diff*1e-31   # m**2 sr**-1

    # Calculate total cross section
    sigma_tot = 2*np.pi*np.trapz(sigma_diff, x=cos_theta, axis=1)

    return E, cos_theta, sigma_diff, sigma_tot


class AlphaScattering(Reaction):
    """A class representing alpha scattering on D or T."""

    def __init__(self, d_or_t, cross_section_file):
        
        super().__init__(d_or_t, 'alpha', d_or_t, 'alpha', None)

        # Load tabulated differential cross section 
        # (same as the one used in ControlRoom)
        self.cx_path = os.path.join(cross_section_path, cross_section_file)
        E, cos_theta, sigma_diff, sigma_tot = load_tab_cross_section(self.cx_path)

        sigma_tab = {}
        sigma_tab['E'] = E
        sigma_tab['cos_theta'] = cos_theta
        sigma_tab['sigma_diff'] = sigma_diff
        sigma_tab['sigma_tot'] = sigma_tot

        self.sigma_tab = sigma_tab
        self.sigma_diff_interp = RectBivariateSpline(E, cos_theta, sigma_diff)


    def _calc_sigma_tot(self, E):
        return np.interp(E, self.sigma_tab['E'], self.sigma_tab['sigma_tot'])

    def _calc_sigma_diff(self, E, cos_theta):
        return self.sigma_diff_interp(E, cos_theta, grid=False)


class DAlphaScattering(AlphaScattering):
    """A class representing D + alpha -> D + alpha scattering."""

    def __init__(self):
        super().__init__('d', 'alphad-cross-section.txt')


class TAlphaScattering(AlphaScattering):
    """A class representing T + alpha -> T + alpha scattering."""

    def __init__(self):
        super().__init__('t', 'alphat-cross-section.txt')
