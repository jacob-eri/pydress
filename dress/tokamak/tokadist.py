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

    def __init__(self, dist_data, density_data, axes):
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
            `spatial_index_fun` argument) and is NOT included here."""


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


    def _get_spatial_index(self, R, Z):
        """Return the spatial index for the given (R,Z) values."""
        pass         # to be over-ridden by subclasses


    def map_dist(self, R, Z):
        """Return distributions and densities at the given (R,Z) points."""

        R, Z = np.atleast_1d(R, Z)
        i_spatial = self._get_spatial_index(R, Z)
        F = self.F[i_spatial]
        density = self.density[i_spatial]

        return F, density

        
# Subclasses for common special cases
# -----------------------------------

class RhoDistData(TokaDistData):
    """Class for holding distribution data where spatial variations 
    depend on a flux surface label `rho` only."""

    def __init__(self, *args, rho_axis, flux_map):
        """Initialize dist which is a function of rho only.

        Parameters
        ----------

        rho_axis : array
            The rho axis used by the distribution.

        flux_map : dress.tokamak.utils.FluxSurfaceMap
            Mapping between (R,Z) and rho.""" 
        
        super().__init__(*args)
        self.rho = rho_axis

        rho_indices = np.arange(len(rho_axis))
        ind_fun = interp1d(rho_axis, rho_indices, kind='nearest', 
                           bounds_error=False, fill_value=-1)

        self._ind_fun = ind_fun
        self.flux_map = flux_map


    def _get_spatial_index(self, R, Z):
        rho_RZ = self.flux_map.get_rho(R, Z)
        ind_RZ = self.ind_fun(rho_RZ)

        return ind_RZ.astype('int')
