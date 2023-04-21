import os

import numpy as np

from dress.reactions.masses import mp, m9Be
from dress.config import cross_section_dir
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class PBe9DBe8Reaction(Reaction):
    """Class representing the p + Be9 -> d + 8Be fusion reaction."""

    def __init__(self):

        super().__init__('p', 'be9', 'd', 'be8', None)

        # Load tabulated cross section (from ENDF)
        cx_path = os.path.join(cross_section_dir,'Be9(p,d)Be8-cross-section_jsi.txt')
        tab_cross_section = np.loadtxt(cx_path, usecols=[0,1], comments='#')
        
        tab_cross_section[:,0] /= 1000.0                   # convert from eV to keV
        tab_cross_section[:,0] *= m9Be / (mp + m9Be)       # convert from LAB to CMS
        tab_cross_section[:,1] *= 1e-28                    # convert from barn to m**2

        self.tab_cross_section = tab_cross_section

    
    def _calc_sigma_tot(self, E):
        sigma =  np.interp(E, self.tab_cross_section[:,0], self.tab_cross_section[:,1],
                           left=0.0, right=0.0)

        return sigma

    def _calc_sigma_diff(self, E, costheta):
        E = np.atleast_1d(E)
        costheta = np.atleast_1d(costheta)

        # Assume isotropic cross section
        sigma = self.calc_sigma_tot(E)/(4*np.pi)    # m**2/sr

        return sigma
