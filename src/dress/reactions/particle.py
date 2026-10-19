import numpy as np

from dress.reactions import masses

# Dictionary of particles known to DRESS and different name used to represent them.

PARTICLE_DICT = {}

PARTICLE_DICT['n'] = {'long name':'neutron',
                      'Z':0, 'A':1, 'u':masses.mn,
                      'name variations':('n', 'neutron'),
                      'excitation energy':0.0}
PARTICLE_DICT['p'] = {'long name':'proton',
                      'Z':1, 'A':1, 'u':masses.mp,
                      'name variations':('p', 'h', 'proton'),
                      'excitation energy':0.0}
PARTICLE_DICT['d'] = {'long name':'deuteron',
                      'Z':1, 'A':2, 'u':masses.md,
                      'name variations':('d', 'h2', '2h', 'deuteron'),
                      'excitation energy':0.0}
PARTICLE_DICT['t'] = {'long name':'triton',
                      'Z':1, 'A':3, 'u':masses.mt,
                      'name variations':('t', 'h3', '3h', 'triton'),
                      'excitation energy':0.0}
PARTICLE_DICT['he3'] = {'long name':'helium-3',
                        'Z':2, 'A':3, 'u':masses.m3He,
                        'name variations':('he3', '3he', 'helium-3'),
                        'excitation energy':0.0}
PARTICLE_DICT['he4'] = {'long name':'helium-4',
                        'Z':2, 'A':4, 'u':masses.m4He,
                        'name variations':('he4', '4he', 'alpha', 'helium-4'),
                        'excitation energy':0.0}
PARTICLE_DICT['li6'] = {'long name':'lithium-6',
                        'Z':3, 'A':6, 'u':masses.m6Li,
                        'name variations':('li6', '6li', 'lithium-6'),
                        'excitation energy':0.0}
PARTICLE_DICT['be8'] = {'long name':'beryllium-8',
                        'Z':4, 'A':8, 'u':masses.m8Be,
                        'name variations':('be8', '8be', 'beryllium-8'),
                        'excitation energy':0.0}
PARTICLE_DICT['be9'] = {'long name':'beryllium-9',
                        'Z':4, 'A':8, 'u':masses.m9Be,
                        'name variations':('be9', '9be', 'beryllium-9'),
                        'excitation energy':0.0}
PARTICLE_DICT['b9'] = {'long name':'boron-9',
                       'Z':5, 'A':9, 'u':masses.m9B,
                       'name variations':('b9', '9b', 'boron-9'),
                       'excitation energy':0.0}
PARTICLE_DICT['b10'] = {'long name':'boron-10',
                        'Z':5, 'A':10, 'u':masses.m10B,
                        'name variations':('b10', '10b', 'boron-10'),
                        'excitation energy':0.0}
PARTICLE_DICT['c12'] = {'long name':'carbon-12',
                        'Z':6, 'A':12, 'u':masses.m12C,
                        'name variations':('c12', '12c', 'carbon-12'),
                        'excitation energy':0.0}
PARTICLE_DICT['c12(e1)'] = {'long name':'carbon-12 (1st excited state)',
                            'Z':6, 'A':12, 'u':masses.m12C,
                            'name variations':('12c(e1)', 'c12(e1)'),
                            'excitation energy':4439.8}      # 1st excited state according to nndc.bnl.gov

# Useful functions
def _get_standard_names(input_name):
    """Get the adopted short and long names for a given particle name or abbreviation."""

    input_name = input_name.lower()
    
    for name, info in PARTICLE_DICT.items():
        if input_name in info['name variations']:
            # We have found the correct particle
            standard_name = name 
            standard_long_name = info['long name']
            return standard_name, standard_long_name

    raise ValueError(f'Name "{input_name}" does not represent a particle known to DRESS')
    
def get_name(A, Z):
    """ Get particle name for given mass number A and atomic number Z."""
    
    for name, info in PARTICLE_DICT.items():
        if (A == info['A']) and (Z == info['Z']):
            return name

    raise ValueError(f'A={A} and Z={Z} does not represent a particle known to DRESS')


# Particle class
class Particle:
    """A class for holding particle info."""

    __slots__ = ('name', 'long_name', 'A', 'Z', 'u', 'excitation_energy', 'm')

    def __init__(self, name):

        self.name, self.long_name = _get_standard_names(name)
        self.A, self.Z = PARTICLE_DICT[self.name]['A'], PARTICLE_DICT[self.name]['Z']
        self.u = PARTICLE_DICT[self.name]['u']
        self.excitation_energy = PARTICLE_DICT[self.name]['excitation energy']

        self.m = self.u * masses.u_keV + self.excitation_energy


    @classmethod
    def from_AZ(cls, A, Z):
        name = get_name(A, Z)
        return cls(name)
    
        
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
