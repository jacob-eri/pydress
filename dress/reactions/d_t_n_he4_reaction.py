import numpy as np

from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class DTNHe4Reaction(Reaction):
    """Class representing the d + t -> n + he4 fusion reaction."""

    def __init__(self, reactant_a='d', reactant_b='t'):

        super().__init__('d', 't', 'n', '4he', None)

    
    def _calc_sigma_tot(self, E):        
        E = np.atleast_1d(E).astype('d')
        sigma = np.zeros_like(E)
        
        # Bosch-Hale parameterization of the T(d,n)4He reaction 
        B_G  = 34.3827
        A1_l = 6.927e4
        A2_l = 7.454e8
        A3_l = 2.05e6
        A4_l = 5.2002e4
        A5_l = 0.0
        B1_l = 6.38e1
        B2_l = -9.95e-1
        B3_l = 6.981e-5
        B4_l = 1.728e-4
        A1_h = -1.4714e6
        A2_h = 0.0
        A3_h = 0.0
        A4_h = 0.0
        A5_h = 0.0
        B1_h = -8.4127e-3
        B2_h = 4.7983e-6
        B3_h = -1.0748e-9
        B4_h = 8.5184e-14

        I_l = E<=550.0 
        I_h = E>550.0
        El = E[I_l]
        Eh = E[I_h]
        
        # Calculate the S factor
        S = np.zeros(np.shape(E))
        S[I_l] = (A1_l + El*(A2_l + El*(A3_l + El*(A4_l + El*A5_l))))/\
                (1.0 + El*(B1_l + El*(B2_l + El*(B3_l + El*B4_l))))
        S[I_h] = (A1_h + Eh*(A2_h + Eh*(A3_h + Eh*(A4_h + Eh*A5_h))))/\
                (1.0 + Eh*(B1_h + Eh*(B2_h + Eh*(B3_h + Eh*B4_h))))
        
        # Calculate the cross section
        nonzero = E > 0
        sigma[nonzero] = S[nonzero]/(E[nonzero]*np.exp(B_G/np.sqrt(E[nonzero])))     # mb
        
        return sigma * 1e-31     # m**2

    def _calc_sigma_diff(self, E, costheta):
        E = np.atleast_1d(E)
        costheta = np.atleast_1d(costheta)

        # Assume isotropic cross section
        sigma = self.calc_sigma_tot(E)/(4*np.pi)    # m**2/sr

        return sigma
