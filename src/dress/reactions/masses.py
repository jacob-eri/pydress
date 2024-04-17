"""Particle masses used by the 'reactions' module."""

from scipy.constants import physical_constants as const


# Atomic mass unit
u_keV = const['atomic mass constant energy equivalent in MeV'][0] * 1000.0      # keV
u_kg  = const['atomic mass constant'][0]   # kg

# Electron mass
me = const['electron mass in u'][0]

# Neutron mass
mn = const['neutron mass in u'][0]

# Proton mass
mp = const['proton mass in u'][0]

# Deuteron mass
md = const['deuteron mass in u'][0]

# Triton mass
mt = const['triton mass in u'][0]

# 3-He mass
# (from https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m3He = (3*u_keV + 14931.21888) / u_keV
m3He = m3He - 2*me        # u

# 4-He mass
m4He = const['alpha particle mass in u'][0]

# 6-Li mass
# (from https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m6Li = (6*u_keV + 14086.8804) / u_keV
m6Li = m6Li - 3*me        # u

# 8-Be mass
# (from https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m8Be = (8*u_keV + 4941.67) / u_keV
m8Be = m8Be - 4*me        # u

# 9-Be mass
# (from https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m9Be = (9*u_keV + 11348.45) / u_keV
m9Be = m9Be - 4*me        # u

# 9-B mass
# (https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m9B = (9*u_keV + 12416.5) / u_keV
m9B = m9B - 5*me          # u

# 10-B mass
# (https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m10B = (10*u_keV + 12050.611) / u_keV
m10B = m10B - 5*me          # u

# 12-C mass
# (from https://www.nndc.bnl.gov/nudat3/. Atomic binding energy neglected)
m12C = 12*u_keV / u_keV
m12C = m12C - 6*me        # u

