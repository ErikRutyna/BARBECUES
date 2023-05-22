import geometryGeneration as geomGen
import matplotlib.pyplot as plt
from scipy import spatial as sp
from scipy import sparse as spa
plt.switch_backend('agg')
from numba import njit
import numpy as np
import copy
import os


def distmesh2d(sdf, h, bound_box, fixed_points, k=0.095, Fscale=1.588, maxIterations=100):
    """Runs a modified version of the DistMesh level-set meshing algorithm. This version of DistMesh has been optimized
    with PySwarms in an effort to minimize the number of iterations to produce a high quality mesh for any given 2D
    geometry.
    
    :param sdf: Anonymous SDF function - this is a lambda function (sdf = lambda p: dpoly(...))
    :param h: Desired edge length in unitless length
    :param bound_box: Bounding box of domain (edges of the computational domain) [xmin, xmax, ymin, ymax]
    :param fixed_points: Fixed points of domain that are to not be moved
    :param k: Bar stiffness coefficient, defaults to 0.2
    :param Fscale: Scaling factor for bar length force adjustments, defaults to 1.5
    :param maxIterations: Maximum number of iterations to produce a high quality mesh, defaults to 1000

    :returns: V, T ([:, 2] x-y coordinate pairs, [:, 3] array of indices in V)
    """
    # Termination/Control numbers
    geps = 1e-3 * h
    dptol = 1e-3
    deps = np.sqrt(np.spacing(1)) * h

    # Bounding box information
    xmin, xmax, ymin, ymax = bound_box

    # Part 1. - Creating initial point distribution using equilateral triangles
    x, y = np.mgrid[xmin:(xmax+h):h, (ymin+h):(ymax+h*np.sqrt(3)/2):(h*np.sqrt(3)/2)]

    # Shift every other row to help with point distribution
    x[:, 1::2] += h/2

    # Create [N, 2] arrays of points
    V = np.vstack((x.flat, y.flat)).T

    # Part 2. - Removing out of bounds points and adding in the fixed points
    V = V[sdf(V) < geps]

    # Removed the fixed points if they're in V
    V = setdiff_rows(V, fixed_points)

    # Add the fixed points back into V
    V = np.vstack((fixed_points, V))

    # Also count number of fixed points we have - need to know index to avoid moving the fixed points later on
    n_fixed = fixed_points.shape[0]

    # Part 3. - Triangulation via Delaunay and spring-force projection
    oldV = np.Inf
    energyResiduals = []
    averageCellQuality = []
    for i in range(maxIterations):
        dist = lambda Vnew, Vold: np.sqrt(((Vnew - Vold)**2).sum(1))
        # Check for large movements in node pairs by evaluating position changes from one cycle to the next
        if (dist(V, oldV)/h).max() > dptol:
            # If there is a large movement, store the current node positions and re-triangulate the mesh
            oldV = copy.deepcopy(V)

            # Compute our new triangles
            T = (sp.Delaunay(V)).simplices

            # Reject triangles that have centroids outside the domain
            Tmidpoints = V[T].sum(1) / 3
            T = T[sdf(Tmidpoints) < - geps]

            # Part 4. - Create edges of each triangle, make them unique, and then sort them
            edgeNodePairsWithDupes = np.vstack((T[:, 0:2], T[:, 1::], T[:, 0::2]))
            edgeNodePairsWithDupes = np.unique(np.squeeze(edgeNodePairsWithDupes).
                                               reshape(-1, np.squeeze(edgeNodePairsWithDupes).shape[-1]), axis=0)
            edgeNodePairsWithDupes.sort(axis=1)

            # Further cleaning to remove duplicates where [1, 2] == [2, 1]
            edgeNodePairs = np.empty((0, 2), dtype=int)
            for edge in edgeNodePairsWithDupes:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(np.logical_and(np.equal(edgeNodePairs[:, 0], edge[0]),
                                         np.equal(edgeNodePairs[:, 1], edge[1])))\
                or np.any(np.logical_and(np.equal(edgeNodePairs[:, 0], rev_edge[0]),
                                         np.equal(edgeNodePairs[:, 1], rev_edge[1]))):
                    continue
                edgeNodePairs = np.vstack((edgeNodePairs, edge))

        # Part 5. - Move the mesh points based on their bar lengths and forces
        # Current edge information
        edgeVectors = V[edgeNodePairs[:, 0], :] - V[edgeNodePairs[:, 1], :]
        edgeLengths = np.sqrt(np.sum(edgeVectors ** 2, axis=1))

        # Desired length for each triangle
        edgeLengthsWanted = np.ones((edgeNodePairs.shape[0])) * Fscale * \
                   np.sqrt(np.sum(np.power(edgeLengths, 2)) / edgeNodePairs.shape[0])

        # Force to move each edge
        F = edgeLengthsWanted - edgeLengths
        F[F < 0] = 0
        F_vectorized = (F[:, None] / edgeLengths[:, None]).dot([[1, 1]]) * edgeVectors

        I = edgeNodePairs[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], edgeNodePairs.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()
        Ftot = dense(I, J, S, shape=(V.shape[0], 2))
        # Account for the fixed nodes that don't have forces applied
        Ftot[0:n_fixed] = 0

        # Move the nodes by a scaled version of the force
        V += k * Ftot

        # Part 6. - Projecting points outside the boundary to the boundary/inside
        p = sdf(V)
        if (p>0).any():
            gradx = (sdf(V[p > 1e-6] + [deps, 0]) - p[p > 1e-6]) / deps
            grady = (sdf(V[p > 1e-6] + [0, deps]) - p[p > 1e-6]) / deps
            grad_tot = gradx ** 2 + grady ** 2
            V[p>1e-6] -= (p[p>1e-6] * np.vstack((gradx, grady))/grad_tot).T

        # After point projection compute the new edges and their lengths
        updatedEdgeVectors = V[edgeNodePairs[:, 0], :] - V[edgeNodePairs[:, 1], :]
        updatedEdgeLengths = np.sqrt(np.sum(updatedEdgeVectors ** 2, axis=1))
        # Elastic Potential Energy =  1/2 * k * x**2, if this is zero then all edge lengths are at desired length
        energyResidual = (0.5 * k * (updatedEdgeLengths - edgeLengthsWanted) ** 2).sum()
        energyResiduals.append(energyResidual)

        # Track the average cell for cells close to boundaries to ensure that cells that have strong geometry
        # interaction are of high quality
        Tmidpoints = V[T].sum(1) / 3
        averageCellQuality.append(geomGen.cellQualityCalculator(T[np.abs(sdf(Tmidpoints)) < 0.1], V))

        # Printout for observation
        if i % 10 == 0:
            print('Mesh Iteration #: {0} \t Meshing Energy Residual: {1} \t Near Boundary Average Cell Quality: {2}'
                  .format(i, energyResiduals[-1], averageCellQuality[-1]))

        # Part 7. - Termination condition: All nodes that had to be moved didn't move that much
        if energyResidual <= 1e-8:
            break

    # Part 8. - Plot the energy residuals and average cell quality as a function of iteration and the mesh itself
    currentDir = os.getcwd()

    # Add the Output dir if it doesn't exist and swap to it
    if not os.path.isdir(os.path.join(os.getcwd(), '../Output/')):
        os.mkdir(os.path.join(os.getcwd(), '../Output/'))
    os.chdir(os.path.join(os.getcwd(), '../Output/'))

    # Plot the energy residuals and average cell quality at each iteration cycle
    f = plt.figure(figsize=(12, 12))
    plt.title('Energy Residual: {0}, N_V = {1}, N_T = {2}'.
              format(energyResiduals[-1], V.shape[0], T.shape[0]))
    plt.plot(range(i+1), energyResiduals, color='r')
    plt.xlabel('Iteration Number')
    plt.ylabel('Energy Residual')
    plt.ylim(0, max(energyResiduals))
    plt.xlim(0, i)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('energy_residuals.png', bbox_inches='tight')

    plt.clf()

    plt.title('Average Cell Quality: {0}, N_V = {1}, N_T = {2}'.
              format(averageCellQuality[-1], V.shape[0], T.shape[0]))
    plt.plot(range(i+1), averageCellQuality, color='r')
    plt.xlabel('Iteration Number')
    plt.ylabel('Average Cell Quality')
    plt.ylim(0, 1)
    plt.xlim(0, i)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('average_cell_quality.png', bbox_inches='tight')
    plt.close('all')

    # Swap back to original working directory
    os.chdir(currentDir)

    print('Meshing Complete - Please view ../Output/ for information and visualization of the mesh.')
    return V, T


def distmesh2d_optimize(sdf, h, bound_box, fixed_points, k=0.2, Fscale=1.5, maxIterations=1000):
    """A modified version of distmesh that can be used with optimization methods to minimize the number of iterations
    required to make a good mesh by maximizing cell quality and minimizing the energy residual..

    :param sdf: Anonymous SDF function - this is a lambda function (sdf = lambda p: dpoly(...))
    :param h: Desired edge length in unitless length
    :param bound_box: Bounding box of domain (edges of the computational domain) [xmin, xmax, ymin, ymax]
    :param fixed_points: Fixed points of domain that are to not be moved
    :param k: Bar stiffness coefficient, defaults to 0.2
    :param Fscale: Scaling factor for bar length force adjustments, defaults to 1.5
    :param maxIterations: Maximum number of iterations to produce a high quality mesh, defaults to 1000

    :returns: V, T ([:, 2] x-y coordinate pairs, [:, 3] array of indices in V)
    """
    # Termination/Control numbers
    geps = 1e-3 * h
    dptol = 1e-3
    deps = np.sqrt(np.spacing(1)) * h

    # Bounding box information
    xmin, xmax, ymin, ymax = bound_box

    # Part 1. - Creating initial point distribution using equilateral triangles
    x, y = np.mgrid[xmin:(xmax+h):h, (ymin+h):(ymax+h*np.sqrt(3)/2):(h*np.sqrt(3)/2)]

    # Shift every other row to help with point distribution
    x[:, 1::2] += h/2

    # Create [N, 2] arrays of points
    V = np.vstack((x.flat, y.flat)).T

    # Part 2. - Removing out of bounds points and adding in the fixed points
    V = V[sdf(V) < geps]

    # Removed the fixed points if they're in V
    V = setdiff_rows(V, fixed_points)

    # Add the fixed points back into V
    V = np.vstack((fixed_points, V))

    # Also count number of fixed points we have - need to know index to avoid moving the fixed points later on
    n_fixed = fixed_points.shape[0]

    # Part 3. - Triangulation via Delaunay and spring-force projection
    oldV = np.Inf
    energyResiduals = []
    averageCellQuality = []
    for i in range(maxIterations):
        dist = lambda Vnew, Vold: np.sqrt(((Vnew - Vold)**2).sum(1))
        # Check for large movements in node pairs by evaluating position changes from one cycle to the next
        if (dist(V, oldV)/h).max() > dptol:
            # If there is a large movement, store the current node positions and re-triangulate the mesh
            oldV = copy.deepcopy(V)

            # Compute our new triangles
            T = (sp.Delaunay(V)).simplices

            # Reject triangles that have centroids outside the domain
            Tmidpoints = V[T].sum(1) / 3
            T = T[sdf(Tmidpoints) < - geps]

            # Part 4. - Create edges of each triangle, make them unique, and then sort them
            edgeNodePairsWithDupes = np.vstack((T[:, 0:2], T[:, 1::], T[:, 0::2]))
            edgeNodePairsWithDupes = np.unique(np.squeeze(edgeNodePairsWithDupes).
                                               reshape(-1, np.squeeze(edgeNodePairsWithDupes).shape[-1]), axis=0)
            edgeNodePairsWithDupes.sort(axis=1)

            # Further cleaning to remove duplicates where [1, 2] == [2, 1]
            edgeNodePairs = np.empty((0, 2), dtype=int)
            for edge in edgeNodePairsWithDupes:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(np.logical_and(np.equal(edgeNodePairs[:, 0], edge[0]),
                                         np.equal(edgeNodePairs[:, 1], edge[1])))\
                or np.any(np.logical_and(np.equal(edgeNodePairs[:, 0], rev_edge[0]),
                                         np.equal(edgeNodePairs[:, 1], rev_edge[1]))):
                    continue
                edgeNodePairs = np.vstack((edgeNodePairs, edge))

        # Part 5. - Move the mesh points based on their bar lengths and forces
        # Current edge information
        edgeVectors = V[edgeNodePairs[:, 0], :] - V[edgeNodePairs[:, 1], :]
        edgeLengths = np.sqrt(np.sum(edgeVectors ** 2, axis=1))

        # Desired length for each triangle
        edgeLengthsWanted = np.ones((edgeNodePairs.shape[0])) * Fscale * \
                   np.sqrt(np.sum(np.power(edgeLengths, 2)) / edgeNodePairs.shape[0])

        # Force to move each edge
        F = edgeLengthsWanted - edgeLengths
        F[F < 0] = 0
        F_vectorized = (F[:, None] / edgeLengths[:, None]).dot([[1, 1]]) * edgeVectors

        I = edgeNodePairs[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], edgeNodePairs.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()
        Ftot = dense(I, J, S, shape=(V.shape[0], 2))
        # Account for the fixed nodes that don't have forces applied
        Ftot[0:n_fixed] = 0

        # Move the nodes by a scaled version of the force
        V += k * Ftot

        # Part 6. - Projecting points outside the boundary to the boundary/inside
        p = sdf(V)
        if (p>0).any():
            gradx = (sdf(V[p > 1e-6] + [deps, 0]) - p[p > 1e-6]) / deps
            grady = (sdf(V[p > 1e-6] + [0, deps]) - p[p > 1e-6]) / deps
            grad_tot = gradx ** 2 + grady ** 2
            V[p>1e-6] -= (p[p>1e-6] * np.vstack((gradx, grady))/grad_tot).T

        # After point projection compute the new edges and their lengths
        updatedEdgeVectors = V[edgeNodePairs[:, 0], :] - V[edgeNodePairs[:, 1], :]
        updatedEdgeLengths = np.sqrt(np.sum(updatedEdgeVectors ** 2, axis=1))
        # Elastic Potential Energy =  1/2 * k * x**2, if this is zero then all edge lengths are at desired length
        energyResidual = (0.5 * k * (updatedEdgeLengths - edgeLengthsWanted) ** 2).sum()
        energyResiduals.append(energyResidual)

        # Track the average cell quality across the mesh
        averageCellQuality.append(geomGen.cellQualityCalculator(T, V))

        # Part 7. - Termination condition: All nodes that had to be moved didn't move that much
        if energyResidual <= 1e-8:
            break

    return energyResiduals[-1], averageCellQuality[-1], i


@njit(cache=True)
def pnpoly(V, p):
    """Performs the winding test for a point in a polygon using the Dan Sunday algorithm:
    https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html#wn_PnPoly()

    :param V: [:, 2] array of x-y points that are the vertices of the polygon
    :param p: x-y coordinates of point to be tested if it is inside the polygon
    :returns: wn_counter - if wn != 0 then the point p exists in the polygon
    """
    v_stack = np.empty((1,2))
    v_stack[0, 0] = V[0, 0]
    v_stack[0, 1] = V[0, 1]
    closed_V = np.vstack((V, v_stack))

    wn = 0
    for i in range(V.shape[0]):
        if closed_V[i, 1] <= p[1]:
            if closed_V[i+1, 1] > p[1]:
                if isleft(closed_V[i], closed_V[i + 1], p) >= 0:
                    wn += 1
        else:
            if closed_V[i+1, 1] <= p[1]:
                if isleft(closed_V[i], closed_V[i + 1], p) < 0:
                    wn -= 1
    return wn


@njit(cache=True)
def isleft(P0, P1, P2):
    """Checks whether x-y point P2 is left or right of a line that passes through x-y points P0 -> P1

    :param P0: x-y coordinate of start of line segment
    :param P1: x-y coordinate of end of line segment
    :param P2: x-y coordinate of point to be tested
    :returns: True if point is left, False if it is now
    """
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])


@njit(cache=True)
def dpoly(V, p):
    """A signed distance function that for a returns the minimum signed distance value for "p" points for a polygon with
     "V" vertices.

    :param V: [:, 2] x-y coordinate pair array of the vertices of a convex closed polygon
    :param p: [:, 2] x-y coordinate pair array of points in the domain
    :returns: poly_sdf: Minimum value signed distance function value for points "p" with regard to a convex polygon made
    up by vertices "V"
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

    inPoly = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        inPoly[i] = pnpoly(V, p[i])

    poly_sdf = (-1) ** inPoly * sdf_segment(a, b, p)
    return poly_sdf

@njit(cache=True)
def sdf_segment(a, b, p):
    """Signed distance function for line segments made up of points [a, b]. The function returns  the minimum signed
    distance of each point in "p" to any of the possible line segments ([a, b]).

    :param a: [:, 2] x-y coordinate pair array of starting position of line segments
    :param b: [:, 2] x-y coordinate pair array of ending position of line segments
    :param p: [:, 2] x-y coordinate pair array of points in the domain
    :returns: sdf_min: Minimum value of the line segment SDF function at each point p to any line segments ([a, b])
    """
    sdf_min = np.zeros(p.shape[0])
    # Now for every line segment and every point in the domain, compute the signed distance function
    for i in range(p.shape[0]):
        sdf = np.zeros(a.shape[0])
        for j in range(a.shape[0]):
            h = min(1, max(0, np.dot(p[i] - a[j], b[j] - p[i]) / np.sqrt(np.power(b[j] - a[j], 2).sum())))
            sdf[j] = np.sqrt(np.power(p[i] - a[j] - h * (b[j] - a[j]), 2).sum())
        sdf_min[i] = np.min(sdf)
    return sdf_min

def ddiff(d1, d2):
    """This signed distance function returns the boolean intersection between two other signed distance functions. It
    can be used to effectively "cut" one polygon out of another.

    :param d1: Distance function for the encompassing polygon
    :param d2: Distance function for the internal polygon (polygon to be removed)
    :returns: diff: Boolean intersection of the two sdf functions "d1" and "d2"
    """
    diff = np.vstack((d1, -d2)).max(axis=0)
    return diff


# MATLAB compatability utility borrowed from: https://github.com/bfroehle/pydistmesh/blob/master/distmesh/mlcompat.py
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
    S = S.flat
    I = I.flat
    J = J.flat
    return spa.coo_matrix((S, (I, J)), shape, dtype).toarray()


def setdiff_rows(A, B, return_index=False):
    """
    Similar to MATLAB's setdiff(A, B, 'rows'), this returns C, and I were C is the rows of A that are not in B and I
    satisfies C = A[I,:].
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
    if return_index:
        if return_inverse:
            return B, I, J
        else:
            return B, I
    else:
        if return_inverse:
            return B, J
        else:
            return B
