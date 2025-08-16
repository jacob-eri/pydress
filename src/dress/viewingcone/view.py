"""Module for mapping out simple viewing cone geometries (simplified python version of LINE21)."""

import numpy as np


# Class definitions
# -----------------

class Line:
    """Vector representation of a line."""

    def __init__(self, p0, p1):
        """Initialize Line instance from the two points ´p0´ and ´p1´."""

        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.l = self.p1 - self.p0   # vector in the direction of the line

    def __call__(self, t):
        """Evaluate xyz coordinates along the line. t=0 corresponds to p0 
        and t=1 corresponds to p1."""

        return self.p0[None,:] + t[:,None]*self.l[None,:]
    
    def get_distance(self, p):
        """Calculate the perpendicular distance between array of points `p` and the line."""

        u = p - self.p0[None,:]
        d = get_norm(np.cross(self.l[None,:], u)) / get_norm(self.l)

        return d


class Plane:
    """Vector representation of a plane."""

    def __init__(self, p0, n):
        """Initialize Plane instance from the point ´p0´ on the plane
        and the vector ´n´ normal to the plane."""

        self.p0 = np.array(p0)
        n = np.array(n)
        self.n = n / get_norm(n)

        # Construct two orthogonal unit vectors, u and v, that lie in the plane.
        self.u = get_perp_vec(self.n)
        self.v = np.cross(self.n, self.u)
    
    
    def contains_point(self, p):
        """Check if a given point lies on the plane."""
        dot_prod = np.dot(p-self.p0, self.n)
        return np.isclose(dot_prod, 0.0)

    
    def eval_uv(self, u, v):
        """Evaluate xyz coordinates of given parameters ´u´ and ´v´."""
        return self.p0 + u*self.u + v*self.v
        

class Collimator:
    """Representation of a cylindrical collimator."""

    def __init__(self, front_point, back_point, radius, back_radius=None):
        """Initialize a cylindrical collimator."""

        front_point = np.array(front_point)
        back_point = np.array(back_point)

        self.front_point = front_point
        self.back_point = back_point
        self.radius = radius

        if back_radius is None:
            self.back_radius = radius
        else:
            self.back_radius = back_radius
        
        # Line along the center
        mid_vec = back_point - front_point
        self.mid_line = Line(front_point, back_point)
        
        self.length = get_norm(mid_vec)

        # Planes defining the front and back surfaces, respectively.
        self.front_plane = Plane(front_point, mid_vec)
        self.back_plane = Plane(back_point, mid_vec)


    def get_solid_angle(self, p0, dx=0.0, dy=0.0, dz=0.0, n_samples=100):
        """Determine the solid angle of the detector surface seen from voxel centered at ´p0´."""

        intensity = self.get_intensity(p0, dx=dx, dy=dy, dz=dz, n_samples=n_samples)
        omega = intensity*4*np.pi
        
        return omega


    def get_intensity(self, p0, dx=0.0, dy=0.0, dz=0.0, n_samples=100):
        """Determine the fraction of particles emitted from voxelel centered at ´p0´ that hits the detector.

        Parameters
        ----------
        p0 : array, shape (3,)
            x,y,x coordinates of the center of the volume from which particles are emitted.

        dx, dy, dz : float
            Widths of the rectangular block from which particles are emitted.

        Returns
        -------
        intensity : float
            The fraction of particles emitted from ´p´ that reach the detector.

        """
        
        # Sample points on the detector surface
        U,V = sample_circle(n_samples, radius=self.back_radius)

        # Sample points from the emission volume
        P = np.random.uniform(low=(p0[0]-dx/2.0, p0[1]-dy/2.0, p0[2]-dz/2.0),
                              high=(p0[0]+dx/2.0, p0[1]+dy/2.0, p0[2]+dz/2.0),
                              size=(n_samples, 3))

        # Compute average particle intensity incident on the detector
        intensity = 0.0
        for u,v,p in zip(U,V,P):
            q = self.back_plane.eval_uv(u,v)
            flux = self._get_flux(p,q)
            intensity += np.dot(flux,self.back_plane.n)

        # Normalize and make units particles/s
        A = np.pi*self.back_radius**2
        intensity = intensity*A/n_samples

        return intensity
        

    def _get_flux(self, p, q):
        """Determine the vector flux at a given point.

        Determine what flux a unit particle flux at the point ´p´ gives rise 
        to at the point ´q´.

        Parameters
        ----------
        p : array, shape (3,)
            x,y,z coordinates of the source point.
        q : array, shape (3,)
            x,y,z coordinates of the point where we want to evaluate the intensity.

        Returns
        -------
        I : array, shape (3,)
            The vector flux (1/m^2/s) at the point ´q´.
        """

        # Check if any particles can reach ´q´, or if they will be stopped by 
        # the front plane of the collimator.
        view_line = Line(p, q)
        x = find_intersection(view_line, self.front_plane)
        
        d = get_norm(x-self.front_point)
        if d > self.radius:
            # ´q´ is not visible from ´p´.
            return np.zeros(3)

        else:
            # ´q´ is visible from ´p´. Evaluate the vector flux.
            r_vec = q - p
            r = get_norm(r_vec)
            return r_vec/(4*np.pi*r**3)


# Functions
# -----------
def find_intersection(line, plane):
    """ Find the intersection between a line and a plane."""

    l_dot_n = np.dot(line.l, plane.n)
    diff_dot_n = np.dot(plane.p0-line.p0, plane.n)

    if l_dot_n != 0.0:
        # One unique intersection
        q = line.p0 + line.l * diff_dot_n / l_dot_n
        return q
    else:
        print('The line is parallel to the plane!')
        return None


def gen_viewing_cone(coll, x_range, y_range, z_range, dx, dy, dz, max_distance):
    """Generate viewing cone specification.
    
    Parameters
    ----------
    
    coll : viewingcone.Collimator instance
        Collimator specification
    
    x_range, y_range, z_range : length-3 tuples
        Specification of the domain where the viewing cone is generated. 
        Each tuple should contain the info (x_min, x_max).
    
    dx, dy, dz : float
        Voxel dimensions.
    
    max_distance : float
        Only points within this distance of the center of the sightline will be mapped.
        Default is None, which means that all points will be mapped.
         
    Returns
    -------
    
    vc : dict
        Dictionary containing the viewing cone specification."""
    
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range

    # Create grid covering the entire domain of interest
    x_vals = np.arange(xmin, xmax+dx, dx)
    y_vals = np.arange(ymin, ymax+dy, dy)
    z_vals = np.arange(zmin, zmax+dz, dz)

    x, y, z = np.meshgrid(x_vals,y_vals,z_vals)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    if max_distance is not None:
        # Only consider points in the vicinity of the sightline
        points = np.column_stack((x, y, z))
        d = coll.mid_line.get_distance(points)
        include = d < max_distance

        x = x[include]
        y = y[include]
        z = z[include]

    n_voxels = len(x)

    dv = dx*dy*dz * np.ones(n_voxels)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    # Loop over voxels and calculate voxel weights and emission directions
    omega = np.zeros(n_voxels)
    u_cyl = np.zeros((n_voxels,3))
    u_xyz = np.zeros((n_voxels,3))

    for i in range(n_voxels):
        # Solid angle
        p0 = np.array([x[i], y[i], z[i]])
        omega[i] = coll.get_solid_angle(p0, dx=dx, dy=dy, dz=dz, n_samples=100)

        # Emission direction (Cartesian coordinates)
        u = coll.back_point - p0
        u_xyz[i] = u / get_norm(u)

    # Convert emission direction to cylindrical coordinates
    ux = u_xyz[:,0]
    uy = u_xyz[:,1]
    uz = u_xyz[:,2]

    u_cyl[:,0] = ux*np.cos(phi) + uy*np.sin(phi)
    u_cyl[:,1] = -ux*np.sin(phi) + uy*np.cos(phi)
    u_cyl[:,2] = uz

    # Remove voxels outside the field of view
    inside = omega > 0.0 

    # Put everything into a dictionary and return it
    vc = {}
    vc['r'] = r[inside]
    vc['z'] = z[inside]
    vc['phi'] = phi[inside]
    vc['x'] = x[inside]
    vc['y'] = y[inside]
    vc['u_xyz'] = u_xyz[inside]
    vc['u_cyl'] = u_cyl[inside]
    vc['dv'] = dv[inside]
    vc['omega'] = omega[inside]
    vc['n_voxels'] = len(vc['omega'])

    return vc


def get_norm_squared(v):
    """Square of the vector ´v´."""
    return np.sum(v**2, axis=-1)


def get_norm(v):
    """2-norm of the vector ´v´."""
    return np.sqrt(get_norm_squared(v))


def get_perp_vec(v):
    """Return a vector that is perpendicular to the input vector ´v´, with the same lenght."""
    
    # First we need any vector which is non-parallell to v. We can obtain
    # this by adding an arbitrary number to the v-component with samllest
    # absolute value.
    v_abs = get_norm(v)
    i = np.argmin(np.abs(v))
    w = v.copy()
    w[i] += v_abs
    
    # Now take the cross product to obtain a perpendicular vector.
    u = np.cross(v,w)

    # Finally, make the output array the same length as the input.
    u = u * v_abs / get_norm(u)

    return u


def sample_circle(n_samples, radius=1.0):
    """Sample points inside a circle, in rectilinear coordinates."""
    r = np.sqrt(np.random.rand(n_samples))
    theta = 2*np.pi*np.random.rand(n_samples)

    u = radius*r*np.cos(theta)
    v = radius*r*np.sin(theta)

    return u,v


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    front_point = [10.0,0,0]
    back_point = [12.0,0,0]
    radius = 0.05
    coll = Collimator(front_point, back_point, radius)

    # Evaluate relative intensity in each point of a given xyz grid
    x = np.linspace(0,4,25)
    y = np.linspace(-0.5,0.5,20)
    z = np.linspace(-0.5,0.5,19)
    
    X,Y,Z = np.meshgrid(x,y,z)
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()

    c = np.zeros_like(x)
    for i in range(len(x)):
        print(i, end='\r')
        p = np.array([x[i], y[i], z[i]])
        c[i] = coll.get_intensity(p)

    # Plot
    plt.hist2d(x, y, weights=c, bins=(25,20))
    plt.axis('equal')
