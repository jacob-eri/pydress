"""Useful tools for performing reaction product spectrum calculations 
for tokamak plasmas."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, interp1d

import dress.utils


class FluxSurfaceMap:
    """Mapping between the spatial coordinates (R,Z) and the 
    flux surface label rho."""

    def __init__(self, R, Z, rho, rho_lim=1.0, rho_outside=2.0):
        """Construct a flux surface map rho = f(R,Z) from given tabulated data.

        Parameters
        ----------
        R : array-like of shape (NP,)
            R coordinates of the tabulated flux surfaces.

        Z : array-like of shape (NP,)
            Z coordinates of the tabulated flux surfaces.
        
        rho : array-like of shape (NP,)
            Flux coordinate value at each of the given (R,Z) points.

        rho_lim : float
            Flux surface label of the last closed flux surface.
            Default is 1.0.

        rho_outside : float
            Fill value to use outside the last closed flux surface.
            Default is 2.0."""

        R = np.atleast_1d(R)
        Z = np.atleast_1d(Z)
        rho = np.atleast_1d(rho)

        if not (1 == R.ndim == Z.ndim == rho.ndim):
            raise ValueError('Input data must be 1D arrays of tabulated flux values')

        if not (len(R) == len(Z) == len(rho)):
            raise ValueError('Input data must have the same length')

        self.Rtab = R
        self.Ztab = Z
        self.rho_tab = rho

        self.rho_lim = rho_lim
        self.rho_outside = rho_outside


    def get_rho(self, R, Z):
        """Evaluate rho for given (R,Z) values.

        Parameters
        ----------
        R, Z : 1D array-likes of same length
            The (R,Z) values where rho is to be evaluated.

        Returns
        -------
        rho : 1D array
            The flux surface label at the requested points."""

        rho = griddata((self.Rtab, self.Ztab), self.rho_tab, (R, Z), method='linear', 
                       fill_value=self.rho_outside)

        return rho


    def get_closest_gridpoint(self, R, Z):
        """Evaluate closest grid point of the given (R,Z) position.

        Parameters
        ----------
        R, Z : 1D array-likes of same length
            The (R,Z) values for which to find the closest grid points.

        Returns
        -------
        grid_index : array of int's
            Indices of the closest grid points."""

        index_values = np.arange(len(self.rho_tab))
        rho_index = griddata((self.Rtab, self.Ztab), index_values, (R, Z), method='nearest')

        return rho_index.astype('int')
        

class FluxSurfaceQuantity:
    """A class for representing flux surface quantities in tokamaks."""

    def __init__(self, rho, val, flux_surface_map):
        """Construct a flux surface quantity from tabulated values.

        Parameters
        ----------
        rho, val : 1D array-likes
            The flux surface values and corresponding values of the tabulated profile.

        flux_surface_map : FluxSurfaceMap
            Mapping between (R,Z) and rho."""

        
        rho = np.atleast_1d(rho)
        val = np.atleast_1d(val)

        if not (1 == rho.ndim == val.ndim):
            raise ValueError('Input data must be 1D arrays of tabulated values')

        if not (len(rho) == len(val)):
            raise ValueError('Input data must have the same length')

        if not isinstance(flux_surface_map, FluxSurfaceMap):
            raise ValueError(f'`flux_surface_map` must be an instance of {FluxSurfaceMap}')

        self.flux_surface_map = flux_surface_map
        self.rho = rho
        self.val = val

        self.eval_from_rho = interp1d(rho, val, kind='linear', fill_value=(val[0],0), bounds_error=False)
        

    def eval_from_RZ(self, R, Z):
        
        rho = self.flux_surface_map.get_rho(R, Z)
        val = self.eval_from_rho(rho)

        return val


def get_toroidal_velocity(angular_freq, R, Z, toroidal_unit_vector=[0,1,0]):
    """Evaluate toroidal velocity at the given (R,Z) points.

    Parameters
    ----------
    angular_freq : instance of dress.tokamak.utils.FluxSurfaceQuantity
        The angular frequency profile (radians/sec).

    R, Z : array-likes of shape (NP,)
        The positions where the toroidal rotation is to be evaluated.

    toroidal_unit_vector : array-like of shape (3,) of (NP,3)
        Components of the toroidal unit vector in the reference frame of interest.
        If 1D array, it is assumed to be the same for all the requested points.

    Returns
    -------
    v_tor : array of shape (NP,3)
        Toroidal velocities at each of the requested points."""

    R, Z = np.atleast_1d(R, Z)
    angf = angular_freq.eval_from_RZ(R, Z)
    
    toroidal_speed = angf*R     # m/s
    
    toroidal_unit_vector = np.atleast_2d(toroidal_unit_vector)
    NP = len(toroidal_speed)

    if toroidal_unit_vector.shape == (1,3):
        toroidal_unit_vector = np.repeat(toroidal_unit_vector, NP, axis=0)

    v_tor = toroidal_speed[:,None] * toroidal_unit_vector

    return v_tor
