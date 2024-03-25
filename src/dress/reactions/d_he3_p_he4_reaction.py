import numpy as np

from dress.reactions.particle import Particle
from dress.reactions.reaction import Reaction


class DHe3PHe4Reaction(Reaction):
    """Class representing the d + he3 -> p + he4 fusion reaction."""

    __slots__ = ()

    def __init__(self):

        super().__init__('d', 'he3', 'p', '4he', None)
    
    def _calc_sigma_tot(self, E):
        E = np.atleast_1d(E).astype('d')
        sigma = np.zeros_like(E)
        
        # Bosch-Hale parameterization of the 3He(d,p)4He reaction 
        # (low and high energy parametrization)
        B_G  = 68.7508
        A1_l = 5.7501e6
        A2_l = 2.5226e3
        A3_l = 4.5566e1
        A4_l = 0.0
        A5_l = 0.0
        B1_l = -3.1995e-3
        B2_l = -8.5530e-6
        B3_l = 5.9014e-8
        B4_l = 0.0
        A1_h = -8.3993e5
        A2_h = 0.0
        A3_h = 0.0
        A4_h = 0.0
        A5_h = 0.0
        B1_h = -2.6830e-3
        B2_h = 1.1633e-6
        B3_h = -2.1332e-10
        B4_h = 1.4250e-14

        I_l = E<=900.0 
        I_h = E>900.0
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
