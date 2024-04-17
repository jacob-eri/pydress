from dress.reactions.particle import Particle


class Reaction:
    """Base class for representing nuclear reactions with two or three particles in the final state."""

    __slots__ = ('name', 'a', 'b', 'p1', 'p2', 'p3')

    def __init__(self, a, b, p1, p2, p3):
        
        self.name = f'{a}-{b}'

        # Reactants
        self.a = Particle(a)
        self.b = Particle(b)

        # Products
        self.p1 = Particle(p1)
        self.p2 = Particle(p2)
        if p3 is None:
            self.p3 = None
        else:
            self.p3 = Particle(p3)

    def __repr__(self):
        return f'Reaction: {self.formula}'

    @property
    def formula(self):
        """A string representation of the reaction formula, like 'a + b -> 1 + 2'"""
        
        formula_string = f'{self.a.name} + {self.b.name} -> {self.p1.name} + {self.p2.name}'
        
        if self.p3 is not None:
            formula_string += f' + {self.p3.name}'            

        return formula_string

    @property
    def Q(self):
        m_reactants = self.a.m + self.b.m
        m_products = self.p1.m + self.p2.m

        if self.p3 is not None:
            m_products += self.p3.m

        return m_reactants - m_products

    def calc_sigma_tot(self, E):
        """Evaluate total cross section.

        Arguments
        ----------
        E : array-like
            Reactant energies in the center-of-momentum frame (keV).

        Returns
        -------
        array
            The cross section (in m**2) for each energy value.
        """
        return self._calc_sigma_tot(E)

    def calc_sigma_diff(self, E, costheta):
        """Evaluate angular differential cross section in the center-of-momentum frame.

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
            The cross section (in m**2 sr**-1) for each energy value."""
        return self._calc_sigma_diff(E, costheta)

        
    # Methods to be overloaded by sub-classes
    def _calc_sigma_tot(self,E):
        pass

    def _calc_sigma_tot(self,E, costheta):
        pass
