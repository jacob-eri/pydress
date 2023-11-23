import os

import numpy as np

from dress.reactions.masses import md, m9Be
from dress.config import cross_section_dir
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class DBe9NB10Reaction(Reaction):
    """Class representing the d + Be9 -> n + B10 fusion reaction."""

    __slots__ = ('tab_cross_section', 'fill_val_high_energy')

    def __init__(self):

        super().__init__('d', 'be9', 'n', 'b10', None)

        # Load tabulated cross section (from ENDF)
        cx_path = os.path.join(cross_section_dir,'Be9(d,n)B10-cross-section_jsi.txt')
        tab_cross_section = np.loadtxt(cx_path, usecols=[0,1], comments='#')
        
        tab_cross_section[:,0] /= 1000.0                   # convert from eV to keV
        tab_cross_section[:,0] *= m9Be / (md + m9Be)       # convert from LAB to CMS
        tab_cross_section[:,1] *= 1e-28                    # convert from barn to m**2

        self.tab_cross_section = tab_cross_section
        self.fill_val_high_energy = 0.0

    
    def _calc_sigma_tot(self, E):
        sigma =  np.interp(E, self.tab_cross_section[:,0], self.tab_cross_section[:,1],
                           left=0.0, right=self.fill_val_high_energy)

        return sigma

    def _calc_sigma_diff(self, E, costheta):
        E = np.atleast_1d(E)
        costheta = np.atleast_1d(costheta)

        # Assume isotropic cross section
        sigma = self.calc_sigma_tot(E)/(4*np.pi)    # m**2/sr

        return sigma
