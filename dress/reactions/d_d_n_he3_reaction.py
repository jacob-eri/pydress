import os

import numpy as np
import scipy.interpolate as interp
from scipy.special import legendre

from dress.reactions.config import cross_section_path
from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


# Load Legendre coefficients for DD differential cross section
c_endf = np.loadtxt(os.path.join(cross_section_path,'ddn3he_legendre_endf.txt')).T

# Convert from LAB to COM (equal masses) and energies to keV
c_endf[0,:] = c_endf[0,:]/2000.0

# Number of coefficients in the expansion
n_endf = c_endf.shape[0] - 1


class DDNHe3Reaction(Reaction):
    """Class representing the d + d -> n + he3 fusion reaction."""

    def __init__(self):

        super().__init__('d', 'd', 'n', 'he3', None)

    
    def _calc_sigma_tot(self, E):
        E = np.atleast_1d(E).astype('d')
        sigma = np.zeros_like(E)
        
        # Bosch-Hale parameterization of the D(d,n)3He reaction
        B_G = 31.3970
            
        A1  = 5.3701e4
        A2  = 3.3027e2
        A3  = -1.2706e-1
        A4  = 2.9327e-5
        A5  = -2.5151e-9
        
        # Calculate the S factor
        S = A1+E*(A2+E*(A3+E*(A4+E*A5)))

        # Calculate the cross section
        nonzero = E > 0
        sigma[nonzero] = S[nonzero]/(E[nonzero]*np.exp(B_G/np.sqrt(E[nonzero])))     # mb
        
        return sigma * 1e-31     # m**2


    def _calc_sigma_diff(self, E, costheta):
        E = np.atleast_1d(E)
        costheta = np.atleast_1d(costheta)

        # Compute the angular dependence of the DD reaction
        A     = interp.interp1d(c_endf[0],c_endf[1:], fill_value=0, bounds_error=False)(E)
        prob  = 0.5*np.ones_like(E)
        
        for i in range(n_endf):
            l = i+1
            P = legendre(l)(costheta)
            prob += (2*l+1)/2 * A[i] * P

        sigma = prob*self.calc_sigma_tot(E)/(2*np.pi)    # m**2/sr

        return sigma
