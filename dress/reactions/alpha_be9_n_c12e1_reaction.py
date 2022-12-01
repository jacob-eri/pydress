import os

import numpy as np

from dress.reactions.masses import m4He, m9Be
from dress.config import cross_section_dir
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class AlphaBe9NC12E1Reaction(Reaction):
    """Class representing the He4 + Be9 -> n + 12C(E1) fusion reaction."""

    def __init__(self):

        super().__init__('he4', 'be9', 'n', 'c12(e1)', None)

        # Load tabulated cross section (from ENDF)
        cx_path = os.path.join(cross_section_dir,'Be9(alpha,n)C12E1-cross-section.txt')
        tab_cross_section = np.loadtxt(cx_path, usecols=[0,1], comments='#')

        tab_cross_section[:,0] /= 1000.0                   # convert from eV to keV
        tab_cross_section[:,0] *= m9Be / (m4He + m9Be)     # convert from LAB to CMS
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
