import numpy as np

from dress.reactions import masses


class Particle:
    """A class for holding particle info."""

    __slots__ = ('name', 'long_name', 'u', 'excitation_energy', 'm')

    def __init__(self, name):

        name = name.lower()

        self.excitation_energy = 0.0

        if name == 'n':
            self.name = 'n'
            self.long_name = 'neutron'
            self.u = masses.mn

        elif name in ['p', 'h']:
            self.name = 'p'
            self.long_name = 'proton'
            self.u = masses.mp

        elif name == 'd':
            self.name = 'd'
            self.long_name = 'deuteron'
            self.u = masses.md

        elif name == 't':
            self.name = 't'
            self.long_name = 'triton'
            self.u = masses.mt

        elif name in ['3he', 'he3']:
            self.name = 'he3'
            self.long_name = 'helium-3'
            self.u = masses.m3He

        elif name in ['4he', 'he4', 'alpha']:
            self.name = 'he4'
            self.long_name = 'helium-4'
            self.u = masses.m4He

        elif name in ['6li', 'li6']:
            self.name = 'li6'
            self.long_name = 'lithium-6'
            self.u = masses.m6Li

        elif name in ['8be', 'be8']:
            self.name = 'be8'
            self.long_name = 'beryllium-8'
            self.u = masses.m8Be

        elif name in ['9be', 'be9']:
            self.name = 'be9'
            self.long_name = 'beryllium-9'
            self.u = masses.m9Be

        elif name in ['9b', 'b9']:
            self.name = 'b9'
            self.long_name = 'boron-9'
            self.u = masses.m9B

        elif name in ['10b', 'b10']:
            self.name = 'b10'
            self.long_name = 'boron-10'
            self.u = masses.m10B

        elif name in ['12c', 'c12']:
            self.name = 'c12'
            self.long_name = 'carbon-12'
            self.u = masses.m12C

        elif name in ['12c(e1)', 'c12(e1)']:
            self.name = 'c12(e1)'
            self.long_name = 'carbon-12 (1st excited state)'
            self.u = masses.m12C
            
            # 1st excited state according to nndc.bnl.gov
            self.excitation_energy = 4439.8    # keV
            
        else:
            raise ValueError(f'Invalid particle name: {name}')

        self.m = self.u * masses.u_keV + self.excitation_energy


    def __repr__(self):
        return 'Particle: {}'.format(self.long_name)

        
    def __eq__(self, other):
        
        if isinstance(other, Particle):
            if self.long_name == other.long_name:
                return True
            else:
                return False

        else:
            return False
