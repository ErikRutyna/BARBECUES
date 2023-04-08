import copy
import numpy as np
from scipy import spatial as sp
from scipy import sparse as spa
from numba import njit


def distmesh2d(anon_sdf, h, bound_box, fixed_points, k=0.2, Fscale=1.5, ttol=1e-3, dptol=1e-3):
    """Runs my version of DistMesh in 2D that is implemented for the generic shapes domain. Code is meant to be similar
    in architecture/algorithm to actual DistMesh for debugging and usability purposes. Only difference is that each
    DistMesh function has its own created distance function as I don't know how to do MATLAB style inline functions in
    Python. We also make a small change to the code provided online in that this code always assumes a uniform
    distribution of cells - no local area refinement as the solver as AMR.

    :param anon_sdf: Anonymous SDF function - but in Python this is a lambda function (anon_sdf = lambda p: dpoly(...)
    :param h: Desired edge length in unitless length
    :param bound_box: Bounding box of domain (edges of the computational domain) [xmin, xmax, ymin, ymax]
    :param fixed_points: Fixed points of domain that are to not be moved
    :param k: Bar stiffness coefficient, defaults to 0.2
    :param Fscale: Scaling factor for bar length force adjustments, defaults to 1.5
    :param dptol: Re-triangulation update tolerance, defaults to 1e-3
    :param ttol: RMS of bar forces residual, must be less than this to exit mesh generation, defaults to 1e-3

    :return: V, T (Vertices [N, 2] x-y coordinate pairs, and triangles [N, 3] of V indices)
    """
    # Termination/Control numbers
    geps = 1e-3 * h
    deps = np.sqrt(np.spacing(1)) * h

    # Bounding box information
    xmin, xmax, ymin, ymax = bound_box

    # Part 1. - Creating initial point distribution using equilateral triangles
    x, y = np.mgrid[(xmin):(xmax+h):h, (ymin+h):(ymax+h*np.sqrt(3)/2):(h*np.sqrt(3)/2)]

    # Shift every other row to help with point distribution
    x[:, 1::2] += h/2

    # Create [N, 2] arrays of points
    V = np.vstack((x.flat, y.flat)).T

    # Part 2. - Removing out of bounds points and adding in the fixed points
    V = V[anon_sdf(V) < geps]

    # Removed the fixed points if they're in V
    V = setdiff_rows(V, fixed_points)

    # Add the fixed points back into V
    V = np.vstack((fixed_points, V))

    # Also count number of fixed points we have - need to know index to avoid moving the fixed points later on
    n_fixed = fixed_points.shape[0]

    # Part 3. - Triangulation via Delaunay and spring-force projection
    oldV = np.Inf
    counter = 0
    while True:
        counter += 1
        dist = lambda Vnew, Vold: np.sqrt(((Vnew - Vold)**2).sum(1))
        # Check for "large" movement by checking relative change of nodes from one cycle to next
        if (dist(V, oldV)/h).max() > dptol:
            oldV = copy.deepcopy(V)
            # Compute our new triangles
            T = (sp.Delaunay(V)).simplices

            # Reject triangles that have centroids outside of the domain
            Tmid = V[T].sum(1) / 3
            T = T[anon_sdf(Tmid) < - geps]

            # Part 4. - Create edges of each triangle, make them unique, and then sort them
            edges = np.vstack((T[:, 0:2], T[:, 1::], T[:, 0::2]))
            edges = np.unique(np.squeeze(edges).reshape(-1, np.squeeze(edges).shape[-1]), axis=0)
            edges.sort(axis=1)
            # Further cleaning to remove duplicates where [1, 2] == [2, 1]
            cleaned_edges = np.empty((0, 2), dtype=int)
            for edge in edges:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(np.logical_and(np.equal(cleaned_edges[:, 0], edge[0]), np.equal(cleaned_edges[:, 1], edge[1])))\
                or np.any(np.logical_and(np.equal(cleaned_edges[:, 0], rev_edge[0]), np.equal(cleaned_edges[:, 1], rev_edge[1]))): continue
                cleaned_edges = np.vstack((cleaned_edges, edge))
        # Part 5. - Move the mesh points based on their bar lengths and forces
        # Current edge information
        edge_vec = V[cleaned_edges[:, 0], :] - V[cleaned_edges[:, 1], :]
        edge_L = np.sqrt(np.sum(edge_vec ** 2, axis=1))

        # Desired length for each triangle
        edge_L_W = np.ones((cleaned_edges.shape[0])) * Fscale * \
                   np.sqrt(np.sum(np.power(edge_L, 2)) / cleaned_edges.shape[0])

        # Force to move each edge
        F = edge_L_W - edge_L
        F[F < 0] = 0
        F_vectorized = (F[:, None] / edge_L[:, None]).dot([[1, 1]]) * edge_vec

        I = cleaned_edges[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], cleaned_edges.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()
        Ftot = dense(I, J, S, shape=(V.shape[0], 2))
        # Account for the fixed nodes that don't have forces applied
        Ftot[0:n_fixed] = 0

        # Move the nodes by a scaled version of the force
        V += k * Ftot

        # Part 6. - Projecting points outside the boundary to the boundary/inside
        p = anon_sdf(V)
        if (p>0).any():
            gradx = (anon_sdf(V[p>1e-6]+[deps, 0]) - p[p>1e-6]) / deps
            grady = (anon_sdf(V[p>1e-6]+[0, deps]) - p[p>1e-6]) / deps
            grad_tot = gradx ** 2 + grady ** 2
            V[p>1e-6] -= (p[p>1e-6] * np.vstack((gradx, grady))/grad_tot).T

        if counter % 10 == 0:
            print('Mesh Iteration #: {0} \t Meshing Energy Residual {1}'.format(counter, (np.sqrt((k * Ftot[p<-geps]**2).sum(1))/h).max()))

        # Part 7. - Termination condition: All nodes that had to be moved didn't move that much
        # TODO: Plot this residual as a function of iteration # and have title show inputs/nodes #/cells #
        # TODO: Add the counter/residual amount as an exit condition
        if counter > 50 or (np.sqrt((k * Ftot[p<-geps]**2).sum(1))/h).max() < ttol:
            break
    print('Number of iterations to make a good mesh: {0}'.format(counter))
    return V, T


# TODO: Combine wnpoly and isLeft into a single function for speed
@njit(cache=True)
def wn_PnPoly(V, p):
    """ Performs the wining test for a point in a polygon using the Dan Sunday
    algorithm:
    https://web.archive.org/web/20130126163405/ ...
    http://geomalgorithms.com/a03-_inclusion.html#wn_PnPoly()

    :param V: [N, 2] array of x-y points that make up the polygon
    :param p: x-y point tested if it is in the polygon denoted by vertices V
    :return: wn_counter - if wn != 0 then the point p exists in the polygon
    """
    v_stack = np.empty((1,2))
    v_stack[0, 0] = V[0, 0]
    v_stack[0, 1] = V[0, 1]
    closed_V = np.vstack((V, v_stack))

    wn = 0
    for i in range(V.shape[0]):
        if closed_V[i, 1] <= p[1]:
            if closed_V[i+1, 1] > p[1]:
                if isLeft(closed_V[i], closed_V[i+1], p) >= 0:
                    wn += 1
        else:
            if closed_V[i+1, 1] <= p[1]:
                if isLeft(closed_V[i], closed_V[i+1], p) < 0:
                    wn -= 1
    return wn


@njit(cache=True)
def isLeft(P0, P1, P2):
    """ Checks whether or not point x-y point p is left or right of a line
    that passes through x-y points V1 -> V2

    :param P0: First point of line in x-y form
    :param P1: Second point of line in x-y form
    :param P2: point to be checked in x-y coordinate pair
    :return: True if point is left, False if it is now
    """
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - \
           (P2[0] - P0[0]) * (P1[1] - P0[1])



@njit(cache=True)
def dpoly(V, p):
    """Signed distance function for "p" points with respect to a generic polygon made up by connecting "V" vertices in a
     closed path

    :param V: Vertices of the polygon given [N, 2] x-y coordinate pair array
    :param p: Points of the domain given [N, 2] x-y coordinate pair array
    """
    # Compute all the line segment pairs
    a = np.zeros((V.shape[0], 2))
    b = np.zeros((V.shape[0], 2))
    for i in range(V.shape[0]):
        # If we're at the end - set the last segment to loop to the first
        if i == (V.shape[0] - 1):
            if np.abs(V[-1, :] - V[0, :]).sum() < 1e-8:
                V[-1, :] += 1e-10
            a[i, :] = V[-1, :]
            b[i, :] = V[0, :]
        # Otherwise the line segment is made up of its node and next node
        else:
            if np.abs(V[i, :] - V[i + 1, :]).sum() < 1e-8:
                V[i, :] += 1e-10
            a[i, :] = V[i, :]
            b[i, :] = V[i + 1, :]

    # TODO: See comment about combining to remove this loop
    booleanmatrix = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        booleanmatrix[i] = wn_PnPoly(V, p[i])

    poly_sdf = (-1) ** booleanmatrix * sdf_segment(a, b, p)
    return poly_sdf

@njit(cache=True)
def sdf_segment(a, b, p):
    """Signed distance function for a line segments made up of points [a, b]. It computes the distances

    :param a: Starting position of each line segment, [x, y]
    :param b: Ending position of each line segment, [x, y]
    :param p: All points in the domain [N, 2], x-y coordinate pair array
    """
    sdf_min = np.zeros(p.shape[0])
    # Now for every line segment and every point in the domain, compute the signed distance function
    for i in range(p.shape[0]):
        sdf = np.zeros(a.shape[0])
        for j in range(a.shape[0]):
            h = min(1, max(0, np.dot(p[i] - a[j], b[j] - p[i]) / np.linalg.norm(b[j] - a[j])))
            sdf[j] = np.linalg.norm(p[i] - a[j] - h * (b[j] - a[j]))
        sdf_min[i] = np.min(sdf)
    return sdf_min

def ddiff(d1, d2):
    """ Signed distance function to find the difference between two regions described by the specific distance functions
    d1 and d2.

    :param d1: Distance function for the encompassing polygon
    :param d2: Distance function for the internal polygon
    """
    diff = np.vstack((d1, -d2)).max(0)
    return diff


# MATLAB compatability utility borrowed from:
# https://github.com/bfroehle/pydistmesh/blob/master/distmesh/mlcompat.py
def dense(I, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a dense array.
    Usage
    -----
    shape = (m, n)
    A = dense(I, J, S, shape, dtype)
    """

    # Advanced usage: allow J and S to be scalars.
    if np.isscalar(J):
        x = J
        J = np.empty(I.shape, dtype=int)
        J.fill(x)
    if np.isscalar(S):
        x = S
        S = np.empty(I.shape)
        S.fill(x)

    # Turn these into 1-d arrays for processing.
    S = S.flat; I = I.flat; J = J.flat
    return spa.coo_matrix((S, (I, J)), shape, dtype).toarray()


def setdiff_rows(A, B, return_index=False):
    """
    Similar to MATLAB's setdiff(A, B, 'rows'), this returns C, I
    where C are the row of A that are not in B and I satisfies
    C = A[I,:].
    Returns I if return_index is True.
    """
    A = np.require(A, requirements='C')
    B = np.require(B, requirements='C')

    assert A.ndim == 2, "array must be 2-dim'l"
    assert B.ndim == 2, "array must be 2-dim'l"
    assert A.shape[1] == B.shape[1], \
           "arrays must have the same number of columns"
    assert A.dtype == B.dtype, \
           "arrays must have the same data type"

    # NumPy provides setdiff1d, which operates only on one dimensional
    # arrays. To make the array one-dimensional, we interpret each row
    # as being a string of characters of the appropriate length.
    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize*ncolumns))
    C = np.setdiff1d(A.view(dtype), B.view(dtype)) \
        .view(A.dtype) \
        .reshape((-1, ncolumns), order='C')
    if return_index:
        raise NotImplementedError
    else:
        return C


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize*ncolumns))
    B, I, J = np.unique(A.view(dtype),
                        return_index=True,
                        return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order='C')

    # There must be a better way to do this:
    if (return_index):
        if (return_inverse):
            return B, I, J
        else:
            return B, I
    else:
        if (return_inverse):
            return B, J
        else:
            return B
