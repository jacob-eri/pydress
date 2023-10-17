"""A module for holding and processing ion distribution data for
tokamaks, where the distribution is typically given as functions of 
two spatial coordinates (R,Z)."""


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class TokaDistData:
    """A class for holding distribution data.

    The distribution data is held in an array `F`, organized such that
    `F[i]` gives the distribution data at spatial point with index `i`.

    The distribution data at a given spatial point  could be a scalar 
    (e.g. temperature) or an array representing e.g. f(v), f(E,pitch) etc."""

    def __init__(self, dist_data, density_data, axes, spatial_index_fun):
        """Initialize a `TokaDistData` instance.

        Parameters
        -----------

        dist_data : array
            The distribution data. The length of the first dimension 
            should match the number of spatial points `NP`.

        density : array of shape (NP,)
            The density at each spatial point.

        axes : tuple of arrays
            The axis along each dimension (e.g. speed, pitch , energy,...).
            The spatial dimension is handeled separately (by the 
            `spatial_index_fun` argument) and is NOT included here.

        spatial_index_fun : callable
            Callable that should have the property that `spatial_index_fun(R,Z)`
            returns the spatial index(indices) corresponding to the given (R,Z)
            point(s), where `R` and `Z` can be arrays (of equal length). Points
            outside the distribution domain should return -1."""


        self.F = dist_data
        self.density = np.atleast_1d(density_data)
        self.X = axes
        self._get_spatial_index = spatial_index_fun


    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, dist_data):
        dist_data = np.atleast2d(dist_data)
        
        # Add one last point with only zeros
        dist_shape = list(dist_data.shape)
        dist_shape[0] = dist_shape[0] + 1

        F = np.zeros(dist_shape)
        F[:-1] = dist_data

        # Set attribute
        self._F = F


    def map_dist(self, R, Z):
        """Return distributions and densities at the given (R,Z) points)."""

        i_spatial = self._get_spatial_index(R, Z, **kwargs)
        F = self.F[i_spatial]
        density = self.density[i_spatial]

        return F, density

        
# Helper functions for creating the spatial index function
# --------------------------------------------------------

class SpatialIndexFun:
    """Helper class for storing data for a spatial index function."""
    
    def __call__(self, R, Z):
        R, Z = np.atleast_1d(R, Z)
        return self.get_spatial_index(R, Z)


def indfun_for_rho_dist(rho_axis, flux_map):
    """Creat a spatial index function for a rho = sqrt(psi_pol) grid.

    Parameters
    ----------

    rho_axis : array
        The rho axis used by the distribution.

    flux_map : dress.tokamak.utils.FluxSurfaceMap
        Mapping between (R,Z) and rho."""

    rho_indices = np.arange(len(rho_axis))
    ind_fun = interp1d(rho_axis, rho_indices, kind='nearest', 
                       bounds_error=False, fill_value=-1)

    fun = SpatialIndexFun()
    fun.flux_map = flux_map

    def 
