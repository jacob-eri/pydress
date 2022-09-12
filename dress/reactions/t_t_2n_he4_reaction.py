import os

import numpy as np

from dress.reactions.config import cross_section_path
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class TT2NHe4Reaction(Reaction):
    """Class representing the t + t -> n + n + he4 fusion reaction."""

    def __init__(self):

        super().__init__('t', 't', 'n', 'n', 'he4')

        # Load tabulated cross section (from ENDF)
        cx_path = os.path.join(cross_section_path,'T(t,2n)He4-cross-section.txt')
        tab_cross_section = np.loadtxt(cx_path, usecols=[0,1], comments='#')
        
        tab_cross_section[:,0] /= 2000.0   # convert from lab to CM, and from eV to keV
        tab_cross_section[:,1] *= 1e-28    # convert from barn to m**2

        self.tab_cross_section = tab_cross_section
        
    
    def calc_sigma_tot(self, E):
        """Evaluate total cross section.

        Arguments
        ----------
        E : array-like
            Reactant energies in the center of mass frame (keV).

        Returns
        -------
        array
            The cross section (in m**2) for each energy value.
        """
        
        sigma =  np.interp(E, self.tab_cross_section[:,0], self.tab_cross_section[:,1],
                           left=0.0, right=0.0)

        return sigma
        

    def calc_sigma_diff(self, E, costheta):
        """Evaluate the angular differential cross section.

        Arguments
        ---------
        E : array-like
            Reactant energies in the center of mass frame (keV).
        costheta : array-like
            Cosine of the emission angle of the neutron 
            (with respect to the reactant relative velocity)

        Returns
        -------
        array
            The cross section (in m**2 sr**-1) for each energy value.
        """

        E = np.atleast_1d(E)
        costheta = np.atleast_1d(costheta)

        # Assume isotropic cross section
        sigma = self.calc_sigma_tot(E)/(4*np.pi)    # m**2/sr

        return sigma
