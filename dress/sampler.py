"""Routines for sampling different types of distributions."""

import numpy as np
import scipy.integrate as integrate


def sample_mono(val, n_samples):
    """ Sample delta function. """

    s = val * np.ones(n_samples)

    return s

def sample_uniform(val_range, n_samples):
    """ Sample values from uniform distribution. """

    s = np.random.uniform(val_range[0], val_range[1], size=n_samples)

    return s

def sample_discrete(vals, n_samples, weights=None):
    """ Sample values from a 1D discrete distribution. """

    vals = np.atleast_1d(vals)

    if weights is not None:
        weights = np.atleast_1d(weights)
        weights = weights / weights.sum()

    s = np.random.choice(vals, p=weights, size=n_samples)

    return s

def sample_inv_trans(f, x, n_samples):
    """ Randomly sample from a tabulated distribution representing f(x), by
    means of inverse transform sampling. """

    if any(f<0):
        raise ValueError('Distribution must be non-negative!')

    # Inverse transform sampling
    cdf = np.zeros_like(f)
    cdf[1:] = integrate.cumtrapz(f, x=x)

    r = np.random.rand(n_samples) * cdf[-1]
    s = np.interp(r, cdf, x)

    return s

def sample_acc_rej(f, lims, fmax, n_samples, quiet=True):
    """ Randomly sample from a distribution f, using the acceptance-rejection method.
    'f' should be callable, taking the dependent variables input. The dimensionality of f
    is inferred from the tuple 'lims', which should contain the ranges to sample from, i.e.

         lims = ([x0_min,x1_min, ...], [x0_max,x1_max, ...])

    'fmax' is the maximum number that f can take. """

    # Sample points xp and fp in the hypercube bounded by 'lims' and 'fmax', and keep only
    # the points that fall below f(*xp). Loop until we have collected enough samples.
    ndims = len(lims[0])

    s = np.zeros((n_samples, ndims))
    n_to_sample = n_samples
    n_sampled = 0

    while n_to_sample > 0:

        # Sample points
        xp = np.random.uniform(low=lims[0], high=lims[1], size=(n_to_sample,ndims))
        fp = np.random.uniform(low=0, high=fmax, size=n_to_sample)

        # Accept/reject
        x_acc = xp[fp < f(*xp.T)]

        # Save accepted points
        i0 = n_sampled

        n_sampled += len(x_acc)
        i1 = n_sampled

        s[i0:i1] = x_acc

        # How many points to sample next time
        n_to_sample -= len(x_acc)

        if not quiet:
            print('{:.2f}%'.format(n_sampled/n_samples*100))

    return s

def sample_sphere(n_samples):
    """Sample points that are uniformly distributed over the surface
    of a unit sphere. Output points are given in a rectangular coordinate 
    system and is given as an array with shape (3, n_samples). """

    # Uniformly sample cosine of polar angle
    u = np.random.uniform(-1, 1, size=n_samples)

    # Uniformly sample the azimuthal angle
    theta = np.random.uniform(0, 2*np.pi, size=n_samples)

    # Coordinates in a rectangular system
    x = np.sqrt(1-u**2) * np.cos(theta)
    y = np.sqrt(1-u**2) * np.sin(theta)
    z = u

    # Return array
    v = np.vstack((x,y,z))

    return v

def sample_mcmc(F, x0, dx, n_samples):
    """
    Sample F(x) by means of Markov Chain Monte Carlo sampling. 
    'F' should be callable, accepting a vector 'x' as input. 
    'x0' is the starting point for the Markov chain.
    'dx' is a suitable step size.
    """

    # Parameters for controlling how to step to new points
    mean_step = np.zeros_like(x0)
    cov_step = dx * np.eye(len(x0))
    
    # Array to store the resulting Markov chain
    chain = np.zeros((n_samples,len(x0)))

    for i in range(n_samples):
        
        # Evaluate function at the present point
        f0 = F(x0)
        
        # Step to next point
        x1 = x0 + np.random.multivariate_normal(mean_step, cov_step)

        # Evaluate function at the new point
        f1 = F(x1)

        # Calculate the Markov ratio
        r = f1/f0

        # Accept new point unconditionally if r >= 1,
        # but if r < 1, accept it with probability r
        if r < 1:
            u = np.random.rand()

            if u > r:
                x1 = x0

        # Store new point and roll values for next step
        chain[i] = x1
        x0 = x1

    return chain

def sample_tab(dist, *axes, n_samples=1e6, dx=None, var_type='continuous'):
    """Sample a tabulated probability density function (pdf).

    Parameters
    ----------
    dist : array-like of shape (N0, N1, N2, ...)
        The tabulated probability density function. Does not have to be normalized.
    axes : 1D-arrays
        The coordinate axes (bin centers). The number of axes should match the number of 
        dimensions of `dist`. The length of each axis should match the number of elements 
        along the corresponding dimension of `dims`, i.e. len(axes[0]) = N0, 
        len(axes[1]) = N1 etc.
    n_samples : int
        Number of samples to draw.
    dx : list of 1D arrays
        The widths of the bins represented by the axes. If `None` (default) an attempt is
        made to reconstruct the bin widths from the axes arrays.
    var_type : str
        Can be either 'continuous', in which case the sampled values are distributed uniformly
        within each bin, of 'discrete', in which case onely the exact tabulated values are sampled.

    Returns
    -------
    sample : array of shape (n_samples, N0, N1, N2, ...)
        Coordinates of the random samples."""

    n_samples = int(n_samples)
    dist = np.atleast_1d(dist)
    axes = [np.atleast_1d(x) for x in axes]
    axes_mgrid = np.meshgrid(*axes, indexing='ij')

    # Input checks
    if len(axes) != dist.ndim:
        raise ValueError(f'Number of axes must match the number of dimenstion in `dist`')

    for i,x in enumerate(axes):
        if x.shape != (dist.shape[i],):
            raise ValueError(f'Axis {i} with shape {x.shape} incompatible with dist with shape {dist.shape}')

        if dx is not None:
            if dx[i].shape != (dist.shape[i],):
                raise ValueError(f'Bin widths for axis {i} with shape {dx[i].shape} incompatible with dist with shape {dist.shape}')

    # Determine the volume elements
    if dx is None:
        dx = _reconstruct_bin_widths(axes)
        
    dx = np.meshgrid(*dx, indexing='ij')
    dv = 1.0
    for dx_ in dx:
        dv = dv*dx_

    # Flatten the distribution to a 1D pdf    
    pdf_1d = dist.flatten()
    x_1d = [x.flatten() for x in axes_mgrid]
    dx_1d = [dx_.flatten() for dx_ in dx]
    dv_1d = dv.flatten()

    # Multiply pdf and dv to get a probability distribution (rather than a density)
    P = pdf_1d * dv_1d
    P = P/P.sum()     # normalize to unity
    
    # Sample from the probability distribution
    i_sample = np.random.choice(len(P), p=P, size=n_samples)
    sample = np.array([x[i_sample] for x in x_1d])

    if var_type == 'continuous':
        # Redistribute each sample uniformly within its respective bin
        jitter = (-0.5 + np.random.random_sample(sample.shape))     # random variable from [-0.5,0.5]
        dx_sample = np.array([dx[i_sample] for dx in dx_1d])        # bin width for each sample
        sample = sample + jitter*dx_sample
        
    elif var_type != 'discrete':
        raise ValueError('`var_type` must be either "continuous" or "discrete"')
        
    
    return sample
    
def _reconstruct_bin_widths(bin_centers):
    """Determine the bin widths for a list of bin_centers.

    For bins with variable bin spacing this is a bit ambigous;
    here we make the assumption that the bin widths are everywhere
    equal to the distance between successive bin centers, and that
    the last bin width is equal to the second last bin width."""

    widths = []

    for i,b in enumerate(bin_centers):
        w = np.zeros_like(b)
        w[:-1] = np.diff(b)
        w[-1] = w[-2]
        widths.append(w)

    return widths
