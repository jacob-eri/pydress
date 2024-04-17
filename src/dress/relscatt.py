"""Module for relativistic scattering calculations."""

import numpy as np
from scipy.constants import c       # speed of light

from dress import relkin

try:
    import sys
    sys.path.append('/home/jeriks/Utils/jspec-res/')

    import jspecf
    CAN_DO_THREEBODY = True

except ModuleNotFoundError:
    CAN_DO_THREEBODY = False


class ReactantData:
    """Helper class for calculating and holding kinematic data about reactants."""

    def __init__(self, Pa, Pb, ma, mb):
        
        # Basic quanteties
        self.Pa = Pa
        self.Pb = Pb
        self.ma = ma
        self.mb = mb

        self.Ptot = Pa + Pb
        self.Mtot = np.sqrt(relkin.mult_four_vectors(self.Ptot, self.Ptot))      # total invariant mass

        self.Tcm = self.Mtot - ma - mb    # kinetic energy in the CMS
        
        # Gamma and beta factors for the CMS
        self.gamma_cm = self.Ptot[0] / self.Mtot
        self.beta_cm =  self.Ptot[1:] / self.Ptot[0]

        # Relative velocity
        self.vrel = calc_vrel(Pa, Pb)
        self.vrel_mag = np.linalg.norm(self.vrel, axis=0)


class ProductData:
    """Helper class for calculating and holding kinematic data about reaction products."""

    def __init__(self, P, beta_cm):

        # Lab frame
        self.P = P
        self.pmag = np.linalg.norm(P[1:], axis=0)

        # CMS
        self.Pcm = relkin.boost(P, beta_cm)
        self.pcm_mag = np.linalg.norm(self.Pcm[1:], axis=0)


def two_body_event(reactant_data, m1, m2, u1):
    """
    Calculate a possible 4-vector (along direction 'u1') of particle 1 in
    the two-body event

           a + b --> 1 + 2

    'Pa' and 'Pb' are the incoming 4-momenta (keV/c).
    'm1' and 'm2' are the masses of the products (keV/c**2).

    All vector input should be 2D, i.e. shape (4,N) for four-vectors
    and (3,N) for three-vectors, where N is either equal to 1 or to
    the number of events.
    """

    # Normalize emission directions
    u1 = u1 / np.linalg.norm(u1, axis=0)

    # Calculate magnitude of 3-momentum
    p1_mag = invariant_to_momentum(m2**2, reactant_data, m1, u1)

    # Energy
    E1 = np.sqrt(p1_mag**2 + m1**2)

    # 4-vector
    P1 = relkin.four_vector(E1, p1_mag*u1)

    return P1


def three_body_event(reactant_data, m1, m2, m3, u1):
    """
    Calculate a possible 4-vector (direction 'u1') of particle 1 in
    the three-body event

         a + b --> 1 + 2 + 3

    'pa' and 'pb' are the incoming 4-momenta (keV/c). 'm1', 'm2' and 'm3' are
    the masses of the products (keV/c**2).

    All vector input should be 2D, i.e. shape (4,N) for four-vectors
    and (3,N) for three-vectors, where N is either equal to 1 or to
    the number of events.
    """
    
    if not CAN_DO_THREEBODY:
        raise NotImplementedError('Three-body calculations not available')

    # Normalize emission directions
    u1 = u1 / np.linalg.norm(u1, axis=0)

    # Total invariant mass
    Mtot = reactant_data.Mtot

    # Sample available phase space (Dalitz plot)
    m23_sq = np.zeros_like(Mtot)
    for i,mtot in enumerate(Mtot):
        m12_sq, m23_sq[i] = jspecf.relevents.dalitzsample(mtot, m1, m2, m3) 

    # Calculate magnitude of 3-momentum
    p1_mag = invariant_to_momentum(m23_sq, reactant_data, m1, u1)

    # Energy
    E1 = np.sqrt(p1_mag**2 + m1**2)

    # 4-vector
    P1 = relkin.four_vector(E1, p1_mag*u1)

    return P1


def invariant_to_momentum(inv, reactant_data, m1, u1):
    """Solution to the four-momentum conservation equation."""

    Ptot = reactant_data.Ptot
    Mtot = reactant_data.Mtot

    Cm  = Mtot**2 + m1**2 - inv
    p0u = np.sum(Ptot[1:] * u1, axis=0)    # projection of CM momentum on emission direction
    E0  = Ptot[0]

    first_term = Cm * p0u

    sq_term1 = Cm**2 * E0**2
    sq_term2 = 4*m1**2 * E0**2 * (E0**2 - p0u**2)
    second_term = np.sqrt(sq_term1 - sq_term2)

    p1_plus = (first_term + second_term) / (2 * (E0**2 - p0u**2))   # magnitude of three-momentum (1st solution)
    p1_minus = (first_term - second_term) / (2 * (E0**2 - p0u**2))   # magnitude of three-momentum (2nd solution)

    # Check which solution (if any) that is valid
    valid_plus = p1_plus >= 0.0
    valid_minus = p1_minus >= 0.0

    p1 = np.repeat(np.nan, len(p1_plus))
    p1[valid_minus] = p1_minus[valid_minus]
    p1[valid_plus] = p1_plus[valid_plus]

    has_two_solutions = valid_plus & valid_minus
    if np.any(has_two_solutions):
        n = np.sum(has_two_solutions)
        ntot = len(has_two_solutions)
        print(f'NOTE: two kinematically allowed solutions found in {n}/{ntot} cases. Using the one with highest energy.')

    return p1


def calc_vrel(Pa, Pb):
    """Calculate relative velocity between particles with four-momenta ´Pa´ and ´Pb´."""
    
    beta_b = Pb[1:] / Pb[0]
    Pa_b = relkin.boost(Pa, beta_b)   # boost particle a into the rest frame of particle b
    vrel = Pa_b[1:] / Pa_b[0]
    
    return vrel


def calc_jacobian_cms_lab(reactant_data, product_data):
    """
    Jacobian for transforming between CMS and LAB emission solid angles,
    as given by Eq. (III.4.16) in Byckling & Kajantie.
    """

    P = product_data.P
    pmag = product_data.pmag
    pcm_mag = product_data.pcm_mag

    beta_cm = reactant_data.beta_cm
    gamma_cm = reactant_data.gamma_cm

    u_dot_beta = np.sum(P[1:]*beta_cm, axis=0) / pmag
    jacobian = pmag**2 / (gamma_cm * pcm_mag * (pmag - P[0]*u_dot_beta))

    return jacobian


def calc_costheta_cm(reactant_data, product_data):
    """Calculate cosine of emission angle of a given reaction product.

    The emission angle is the angle between the emission direction and the
    reactant relative velocity, in the CMS.
    """

    vrel = reactant_data.vrel
    vrel_mag = reactant_data.vrel_mag
    
    nonzero = vrel_mag > 0
    u_rel = np.zeros_like(vrel)
    u_rel[:,nonzero] = vrel[:,nonzero] / vrel_mag[nonzero]   # unit vector along relative motion of reactants

    P1_cm = product_data.Pcm
    p1cm_mag = product_data.pcm_mag
    costheta = np.sum(P1_cm[1:]*u_rel, axis=0) / p1cm_mag

    return costheta
    

def get_reactivity(reactant_data, P1, reaction):
    """
    Calculate the reactivity (m**3/sr/s) for a reaction with
    given reactant data and the product of interest with four-momentum ´P1´.

    All vector input should be 2D, i.e. shape (4,N) for four-vectors
    and (3,N) for three-vectors, where N is either equal to 1 or to
    the number of events.
    """

    # Eval P1 in the CMS
    P1_data = ProductData(P1, reactant_data.beta_cm)

    # Jacobian for transforming between CMS and LAB
    jacobian = calc_jacobian_cms_lab(reactant_data, P1_data)

    # Angular differential cross section in the CMS
    costheta = calc_costheta_cm(reactant_data, P1_data)
    sigma = reaction.calc_sigma_diff(reactant_data.Tcm, costheta)

    # Reactivity (m**3/sr/s)
    vrel_mag = reactant_data.vrel_mag * c      # m/s
    sigmav = sigma * vrel_mag * jacobian

    return sigmav
