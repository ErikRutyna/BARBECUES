import copy

import numpy as np
import math
from scipy import spatial as sp
from scipy import sparse as spa
import mesh_generation_helper as mgh

def shapes_distmesh(x, y, h, circle, square, rectangle, fixed_points):
    """Runs my version of DistMesh in 2D that is implemented for the generic shapes domain. Code is meant to be similar
    in architecture/algorithm to actual DistMesh for debugging and usability purposes. Only difference is that each
    DistMesh function has its own created distance function as I don't know how to do MATLAB style inline functions in
    Python. We also make a small change to the code provided online in that this code always assumes a uniform
    distribution of cells - no local area refinement as the solver as AMR.

    :return:
    """
    N = np.ceil(math.sqrt(x / h))
    # The set of tolerances/scaling factors
    dptol = 1e-3
    ttol = 1e-1
    Fscale = 1.2
    deltat = 0.2
    geps = 1e-3 * h
    deps = np.sqrt(np.spacing(1)) * h

    # Part 1 - Initial point distribution & shifting of every other row to make nice triangles
    x_nodes, y_nodes = np.meshgrid(np.arange(-x/2, x/2, h), np.arange(-y/2, y/2, math.sqrt(3)/2*h))
    x_nodes[1::2, :] += h/2
    nodes = np.stack((x_nodes.flatten(), y_nodes.flatten()), axis=1)

    # Part 2 - Remvoing points outside the region via the custom distance function
    nodes = np.vstack((fixed_points, nodes[np.squeeze(shapes_distance(x, y, circle, square, rectangle, nodes)< geps), :]))

    old_nodes = np.Inf
    # The loop that runs the bulk of the Delaunay Triangulation - Adjustment - Re-triangulation process
    while True:
        # Part 3 - Delaunay Triangulation via built in functionalities
        ttol_tracker = max(np.sqrt(np.sum(np.power(nodes - old_nodes, 2), axis=1)) / h)
        ttol_tracker =  (np.sqrt((np.power(nodes - old_nodes, 2)).sum(1)) / h ).max()
        if ttol_tracker > ttol:
            old_nodes = copy.deepcopy(nodes)
            triangles = (sp.Delaunay(nodes)).simplices

            centroids = (nodes[triangles[:, 0], :] + nodes[triangles[:, 0], :] + nodes[triangles[:, 2], :])/3

            # Reject triangles with centroids outside the domain
            triangles = triangles[np.squeeze(shapes_distance(x, y, circle, square, rectangle, centroids) < -geps)]

            # Part 4 - Describing edges as a pair of nodes
            edges = np.vstack((triangles[:, 0:2], triangles[:, 1::], triangles[:, 0::2]))
            edges = np.unique(np.squeeze(edges).reshape(-1, np.squeeze(edges).shape[-1]), axis=0)
            edges.sort(axis=1)
            # Fixing the edges to account for duplicates (i.e. [1, 2] == [2, 1])
            cleaned_edges = np.empty((0, 2), dtype=int)
            for edge in edges:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(np.logical_and(np.equal(cleaned_edges[:, 0], edge[0]), np.equal(cleaned_edges[:, 1], edge[1])))\
                or np.any(np.logical_and(np.equal(cleaned_edges[:, 0], rev_edge[0]), np.equal(cleaned_edges[:, 1], rev_edge[1]))): continue
                cleaned_edges = np.vstack((cleaned_edges, edge))

        # Part "5" - Plotting/saving the current mesh for viewing purposes
        #TODO: Setup plotting for the mesh at each iteration of the algorithm

        # Part 6 - Moving meshing points based on lengths and forces "F"
        edge_vectors = nodes[cleaned_edges[:, 0], :] - nodes[cleaned_edges[:, 1], :]
        edge_lengths = np.sqrt(np.sum(edge_vectors ** 2, axis=1))
        # Uniform distribution - scaling by number of edges
        edge_lengths_wanted = np.ones((cleaned_edges.shape[0])) * Fscale * \
                              np.sqrt(np.sum(np.power(edge_lengths, 2)) / cleaned_edges.shape[0])
        F = np.stack((edge_lengths_wanted - edge_lengths, np.zeros((edge_lengths.shape[0]))), axis=1).max(axis=1)

        F_vectorized = (F[:, None]/edge_lengths[:, None]).dot([[1, 1]]) * edge_vectors
        # F_vectorized = np.multiply(np.stack((F_vectorized, F_vectorized), axis=1), edge_vectors)
        # F_vectorized = np.multiply(np.multiply(np.stack((np.divide(F, edge_lengths), np.divide(F, edge_lengths)), axis=1), np.ones((F.shape[0], 2))), edge_vectors)

        I = cleaned_edges[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], cleaned_edges.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()

        Ftot = dense(I, J, S, shape=(nodes.shape[0], 2))
        Ftot[0:fixed_points.shape[0]+1, :] = 0
        nodes += deltat * Ftot

        # Part 7 - Moving exterior points towards the interior
        d = unit_circle_distance(nodes)
        i_out = d > 0

        x_grad_nodes = nodes[i_out] + [deps, 0]
        y_grad_nodes = nodes[i_out] + [0, deps]
        gradient_x = (unit_circle_distance(x_grad_nodes) - d[i_out]) / deps
        gradient_y = (unit_circle_distance(y_grad_nodes) - d[i_out]) / deps
        gradient_mag = gradient_x ** 2 + gradient_y ** 2

        nodes[i_out] -= (d[i_out]*np.vstack((gradient_x, gradient_y))/gradient_mag).T

        # Part 8 - Termination: All interior points move less than dptol then we've made a good quality mesh
        tracker = (np.sqrt((deltat*Ftot[d<-geps]**2).sum(1))/h).max()
        if tracker < dptol:
            break
    return nodes, triangles


def unit_circle_distmesh(x, y, h, circle, square, rectangle, fixed_points):
    """Runs my version of DistMesh in 2D that is implemented for the generic shapes domain. Code is meant to be similar
    in architecture/algorithm to actual DistMesh for debugging and usability purposes. Only difference is that each
    DistMesh function has its own created distance function as I don't know how to do MATLAB style inline functions in
    Python. We also make a small change to the code provided online in that this code always assumes a uniform
    distribution of cells - no local area refinement as the solver as AMR.

    :return:
    """
    N = np.ceil(math.sqrt(x / h))
    # The set of tolerances/scaling factors
    dptol = 1e-3
    ttol = 1e-1
    Fscale = 1.2
    deltat = 0.2
    geps = 1e-3 * h
    deps = np.sqrt(np.spacing(1)) * h

    # Part 1 - Initial point distribution & shifting of every other row to make nice triangles
    x_nodes, y_nodes = np.meshgrid(np.arange(-x/2, x/2, h), np.arange(-y/2, y/2, math.sqrt(3)/2*h))
    x_nodes[1::2, :] += h/2
    nodes = np.stack((x_nodes.flatten(), y_nodes.flatten()), axis=1)

    # Part 2 - Remvoing points outside the region via the custom distance function
    nodes = np.vstack((nodes[unit_circle_distance(nodes) < geps, :]))

    old_nodes = np.Inf
    # The loop that runs the bulk of the Delaunay Triangulation - Adjustment - Re-triangulation process
    while True:
        # Part 3 - Delaunay Triangulation via built in functionalities
        ttol_tracker = max(np.sqrt(np.sum(np.power(nodes - old_nodes, 2), axis=1)) / h)
        ttol_tracker =  (np.sqrt((np.power(nodes - old_nodes, 2)).sum(1)) / h ).max()
        if ttol_tracker > ttol:
            old_nodes = copy.deepcopy(nodes)
            triangles = (sp.Delaunay(nodes)).simplices

            centroids = (nodes[triangles[:, 0], :] + nodes[triangles[:, 0], :] + nodes[triangles[:, 2], :])/3

            # Reject triangles with centroids outside the domain
            triangles = triangles[unit_circle_distance(centroids) < -geps, :]

            # Part 4 - Describing edges as a pair of nodes
            edges = np.vstack((triangles[:, 0:2], triangles[:, 1::], triangles[:, 0::2]))
            edges = np.unique(np.squeeze(edges).reshape(-1, np.squeeze(edges).shape[-1]), axis=0)
            edges.sort(axis=1)
            # Fixing the edges to account for duplicates (i.e. [1, 2] == [2, 1])
            cleaned_edges = np.empty((0, 2), dtype=int)
            for edge in edges:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(np.logical_and(np.equal(cleaned_edges[:, 0], edge[0]), np.equal(cleaned_edges[:, 1], edge[1])))\
                or np.any(np.logical_and(np.equal(cleaned_edges[:, 0], rev_edge[0]), np.equal(cleaned_edges[:, 1], rev_edge[1]))): continue
                cleaned_edges = np.vstack((cleaned_edges, edge))


        # Part "5" - Plotting/saving the current mesh for viewing purposes
        #TODO: Setup plotting for the mesh at each iteration of the algorithm

        # Part 6 - Moving meshing points based on lengths and forces "F"
        edge_vectors = nodes[cleaned_edges[:, 0], :] - nodes[cleaned_edges[:, 1], :]
        edge_lengths = np.sqrt(np.sum(edge_vectors ** 2, axis=1))
        # Uniform distribution - scaling by number of edges
        edge_lengths_wanted = np.ones((cleaned_edges.shape[0])) * Fscale * \
                              np.sqrt(np.sum(np.power(edge_lengths, 2)) / cleaned_edges.shape[0])
        F = np.stack((edge_lengths_wanted - edge_lengths, np.zeros((edge_lengths.shape[0]))), axis=1).max(axis=1)

        F_vectorized = (F[:, None]/edge_lengths[:, None]).dot([[1, 1]]) * edge_vectors
        # F_vectorized = np.multiply(np.stack((F_vectorized, F_vectorized), axis=1), edge_vectors)
        # F_vectorized = np.multiply(np.multiply(np.stack((np.divide(F, edge_lengths), np.divide(F, edge_lengths)), axis=1), np.ones((F.shape[0], 2))), edge_vectors)

        I = cleaned_edges[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], cleaned_edges.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()

        Ftot = dense(I, J, S, shape=(nodes.shape[0], 2))
        # Ftot[0:fixed_points.shape[0]+1, :] = 0
        nodes += deltat * Ftot

        # Part 7 - Moving exterior points towards the interior
        d = unit_circle_distance(nodes)
        i_out = d > 0

        x_grad_nodes = nodes[i_out] + [deps, 0]
        y_grad_nodes = nodes[i_out] + [0, deps]
        gradient_x = (unit_circle_distance(x_grad_nodes) - d[i_out]) / deps
        gradient_y = (unit_circle_distance(y_grad_nodes) - d[i_out]) / deps
        gradient_mag = gradient_x ** 2 + gradient_y ** 2

        nodes[i_out] -= (d[i_out]*np.vstack((gradient_x, gradient_y))/gradient_mag).T

        # Part 8 - Termination: All interior points move less than dptol then we've made a good quality mesh
        tracker = (np.sqrt((deltat*Ftot[d<-geps]**2).sum(1))/h).max()
        if tracker < dptol:
            break
    return nodes, triangles


def flat_plate_distmesh_shape(x, y, h, fixed_points, l , w):
    """Like unit circle distmesh algorithm, but for a flat plate with sharp edges."""

    # Part 0 - tolerances and variance parameters
    dptol = 1e-3
    ttol = 1e-1
    Fscale = 1.2
    deltat = 0.2
    geps = 1e-3 * h
    deps = np.sqrt(np.spacing(1)) * h

    # Part 1 - Initial point distribution & shifting of every other row to make nice triangles
    x_nodes, y_nodes = np.meshgrid(np.arange(-x/2, x/2, h), np.arange(-y/2, y/2, math.sqrt(3)/2*h))
    x_nodes[1::2, :] += h/2
    nodes = np.stack((x_nodes.flatten(), y_nodes.flatten()), axis=1)

    # Part 2 - Remvoing points outside the region via the custom distance function
    nodes = np.vstack((fixed_points, nodes[rectangle_distance(nodes, l, w) < geps, :]))

    old_nodes = np.Inf
    # The loop that runs the bulk of the Delaunay Triangulation - Adjustment - Re-triangulation process
    while True:
        # Part 3 - Delaunay Triangulation via built in functionalities
        ttol_tracker = (np.sqrt((np.power(nodes - old_nodes, 2)).sum(1)) / h).max()
        if ttol_tracker > ttol:
            old_nodes = copy.deepcopy(nodes)
            triangles = (sp.Delaunay(nodes)).simplices

            centroids = (nodes[triangles[:, 0], :] + nodes[triangles[:, 0], :] + nodes[triangles[:, 2], :]) / 3

            # Reject triangles with centroids outside the domain
            triangles = triangles[rectangle_distance(centroids, l, w) < 0, :]

            # Part 4 - Describing edges as a pair of nodes
            edges = np.vstack((triangles[:, 0:2], triangles[:, 1::], triangles[:, 0::2]))
            edges = np.unique(np.squeeze(edges).reshape(-1, np.squeeze(edges).shape[-1]), axis=0)
            edges.sort(axis=1)
            # Fixing the edges to account for duplicates (i.e. [1, 2] == [2, 1])
            cleaned_edges = np.empty((0, 2), dtype=int)
            for edge in edges:
                rev_edge = np.array([edge[1], edge[0]])
                if np.any(
                        np.logical_and(np.equal(cleaned_edges[:, 0], edge[0]), np.equal(cleaned_edges[:, 1], edge[1]))) \
                        or np.any(np.logical_and(np.equal(cleaned_edges[:, 0], rev_edge[0]),
                                                 np.equal(cleaned_edges[:, 1], rev_edge[1]))): continue
                cleaned_edges = np.vstack((cleaned_edges, edge))

        # Part "5" - Plotting/saving the current mesh for viewing purposes
        # TODO: Setup plotting for the mesh at each iteration of the algorithm

        # Part 6 - Moving meshing points based on lengths and forces "F"
        edge_vectors = nodes[cleaned_edges[:, 0], :] - nodes[cleaned_edges[:, 1], :]
        edge_lengths = np.sqrt(np.sum(edge_vectors ** 2, axis=1))
        # Uniform distribution - scaling by number of edges
        edge_lengths_wanted = np.ones((cleaned_edges.shape[0])) * Fscale * \
                              np.sqrt(np.sum(np.power(edge_lengths, 2)) / cleaned_edges.shape[0])
        F = np.stack((edge_lengths_wanted - edge_lengths, np.zeros((edge_lengths.shape[0]))), axis=1).max(axis=1)

        F_vectorized = (F[:, None] / edge_lengths[:, None]).dot([[1, 1]]) * edge_vectors
        # F_vectorized = np.multiply(np.stack((F_vectorized, F_vectorized), axis=1), edge_vectors)
        # F_vectorized = np.multiply(np.multiply(np.stack((np.divide(F, edge_lengths), np.divide(F, edge_lengths)), axis=1), np.ones((F.shape[0], 2))), edge_vectors)

        I = cleaned_edges[:, [0, 0, 1, 1]].flatten()
        J = np.repeat([[0, 1, 0, 1]], cleaned_edges.shape[0], axis=0).flatten()
        S = np.stack((F_vectorized, -F_vectorized), axis=1).flatten()

        Ftot = dense(I, J, S, shape=(nodes.shape[0], 2))
        Ftot[0:fixed_points.shape[0]+1, :] = 0
        nodes += deltat * Ftot

        # Part 7 - Moving exterior points towards the interior
        d = rectangle_distance(nodes, l, w)
        i_out = d > 0

        x_grad_nodes = nodes[i_out] + [deps, 0]
        y_grad_nodes = nodes[i_out] + [0, deps]
        gradient_x = (rectangle_distance(x_grad_nodes, l, w) - d[i_out]) / deps
        gradient_y = (rectangle_distance(y_grad_nodes, l, w) - d[i_out]) / deps
        gradient_mag = gradient_x ** 2 + gradient_y ** 2

        nodes[i_out] -= (d[i_out] * np.vstack((gradient_x, gradient_y)) / gradient_mag).T

        # Part 8 - Termination: All interior points move less than dptol then we've made a good quality mesh
        tracker = (np.sqrt((deltat * Ftot[d < -geps] ** 2).sum(1)) / h).max()
        if tracker < dptol:
            break
    return nodes, triangles

def rectangle_distance(nodes, l, w):
    distance = np.zeros((nodes.shape[0]))
    for i in range(nodes.shape[0]):
        distance[i] = min([l - nodes[i, 0], nodes[i, 0] - l, w - nodes[i, 1], nodes[i, 1] - w])
        if -l/2 < nodes[i, 0] < l/2 and -w/2 < nodes[i, 1] < w/2: distance[i] = abs(distance[i])
    return distance

def unit_circle_distance(nodes):
    distance = np.zeros((nodes.shape[0]))
    for i in range(nodes.shape[0]):
        distance[i] = np.linalg.norm(nodes[i]) - 1
    return distance

def shapes_distance(x, y, circle, square, rectangle, p):
    """This is a "distance" function that returns -1/0/1 for a given point p if the point p is inside/on edge of/outside
    the given domain configuration for the basic shape combination.

    :param x: Size of domain in x-dir x E [-x/2, x/2]
    :param y: Size of domain in y-dir x E [-y/2, y/2]
    :param circle: [x0, y0, r]
    :param square: [x0, y0, l]
    :param rectangle: [x0, y0, l, w]
    :param p: Nx2 array of points p
    :return:
    """
    dist = np.zeros((p.shape[0], 1))

    for i in range(p.shape[0]):
        # Check if the point is outside the domain w/ following if-tree

        # Check x-pos
        if -x/2 > p[i, 0] or p[i, 0] > x/2:
            x_dis = abs(p[i, 0]) - x/2
            # If the y-position is also not inside, find distance
            if -y/2 > p[i, 1] or p[i, 1] > y/2:
                y_dis = abs(p[i, 1]) - y/2
                dist[i] = np.linalg.norm([x_dis, y_dis])
            # If y-position is inside, the closest distance is x-distance
            else:
                dist[i] = x_dis

        # Check y-pos
        if -y/2 > p[i, 1] or p[i, 1] > y/2:
            y_dis = abs(p[i, 1]) - y/2
            # If the y-position is also not inside, find distance
            if -x/2 > p[i, 0] or p[i, 0] > x/2:
                x_dis = abs(p[i, 0]) - x/2
                dist[i] = np.linalg.norm([x_dis, y_dis])
            # If y-position is inside, the closest distance is y-distance
            else:
                dist[i] = y_dis

        # Check circle
        if check_in_circle(circle, p[i, 0], p[i, 1]):
            polar_point = [p[i, 0] - circle[0], p[i, 1] - circle[1]]
            dist[i] = np.linalg.norm(polar_point)

        # Check square
        if check_in_square(square, p[i, 0], p[i, 1]):
            # [x0, y0, l]
            lines = np.array([[abs(square[0] + square[2]/2 - p[i, 0])],
                               [abs(square[0] - square[2]/2 - p[i, 0])],
                               [abs(square[1] + square[2]/2 - p[i, 1])],
                               [abs(square[1] - square[2]/2 - p[i, 1])]])
            dist[i] = min(lines)

        # Check rectangle
        if check_in_rectangle(rectangle, p[i, 0], p[i, 1]):
            # [x0, y0, l, w]
            lines = np.array([[abs(rectangle[0] + rectangle[3]/2 - p[i, 0])],
                               [abs(rectangle[0] - rectangle[3]/2 - p[i, 0])],
                               [abs(rectangle[1] + rectangle[2]/2 - p[i, 1])],
                               [abs(rectangle[1] - rectangle[2]/2 - p[i, 1])]])
            dist[i] = min(lines)

        # Check if point is inside
        if (not check_in_rectangle(rectangle, p[i, 0], p[i, 1]) and not check_in_circle(circle, p[i, 0], p[i, 1]) \
            and not check_in_square(square, p[i, 0], p[i, 1])) and \
                (-x/2 < p[i, 0] < x/2) and (-y/2 < p[i, 1] < y/2):
            dist[i] = -1
        # If neither hit then point is on boundary in which continue
        else: continue

    return dist


def check_in_rectangle(rectangle, xp, yp):
    """Check if the point [xp, yp] is in the given rectangle [x0, y0, l, w]

    :param rectangle: [x0, y0, l, w]
    :param xp: x-coordinate of point
    :param yp: y-coordinate of point
    :return: True/False depending on if point is/isn't inside the rectangle
    """
    if rectangle[0] - rectangle[2] / 2 < xp < rectangle[0] + rectangle[2] / 2 and \
        rectangle[1] - rectangle[3] / 2 < yp < rectangle[1] + rectangle[3] / 2:
        return True
    else:
        return False


def check_in_square(square, xp, yp):
    """Check if the point [xp, yp] is in the given rectangle [x0, y0, l, w]

    :param square: [x0, y0, l]
    :param xp: x-coordinate of point
    :param yp: y-coordinate of point
    :return: True/False depending on if point is/isn't inside the rectangle
    """
    if square[0] - square[2] / 2 < xp < square[0] + square[2] / 2 and \
        square[1] - square[2] / 2 < yp < square[1] + square[2] / 2:
        return True
    else:
        return False


def check_in_circle(circle, xp, yp):
    """Checks if the point [xp, yp] is in the given circle [x0, y0, r]

    :param circle: [x0, y0, r]
    :param xp: x-coordinate of point
    :param yp: y-coordinate of point
    :return: True/False depending on if point is/isn't inside the circle
    """
    polar_point = [xp - circle[0], yp - circle[1]]
    if np.linalg.norm(polar_point) < circle[2]:
        return True
    else:
        return False


# MATLAB compatability utitly borrowed from: https://github.com/bfroehle/pydistmesh
def dense(I, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a
    dense array.
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