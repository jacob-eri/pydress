import os

import numpy as np

from dress.config import cross_section_dir
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class PTNHe3Reaction(Reaction):
    """Class representing the p + t -> n + he3 fusion reaction."""

    __slots__ = ('tab_cross_section')

    def __init__(self):

        super().__init__('p', 't', 'n', 'he3', None)

        # Load tabulated cross section (from ENDF)
        cx_path = os.path.join(cross_section_dir,'T(t,2n)He4-cross-section.txt')
        tab_cross_section = np.loadtxt(cx_path, usecols=[0,1], comments='#')
        
        tab_cross_section[:,0] /= 2000.0   # convert from lab to CM, and from eV to keV
        tab_cross_section[:,1] *= 1e-28    # convert from barn to m**2

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
