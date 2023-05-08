import cell_geometry_formulas as cgf
import compressible_flow_formulas
import initialization as intlzn
from numba import njit
import numpy as np
import plotting
import math


def initialize_moc(E, V, BE, M, a, y):
    """Initializes the initial state using characteristic lines to approximate oblique shock locations and initializes
    the solution using those oblique shocks. FLOW MUST GO FROM LEFT TO RIGHT as it uses the left running characteristic
    lines.

    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param BE: [:, 4] Numpy array boundary Edge Matrix [nodeA, nodeB, cell, boundary flag]
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: state: [:, 4] Numpy array of state vectors, each row is 1 cell's state [rho, rho*u, rho*v, rho*E]
    """
    # Generate the characteristic lines from the inflow locations
    moc_lines = moc_inflow(V, BE, M, a)

    # Reflect the characteristic lines from the inflow until it reaches the rightmost edge of the domain
    moc_lines = moc_reflect(V, BE, M, a, y, moc_lines)

    # Delete any characteristics that didn't get any reflections
    for i in range(len(moc_lines)-1, -1, -1):
        # Only two points -> no reflections
        if moc_lines[i].shape[0] <= 2:
            del moc_lines[i]

    # Visualize the MoC lines with a plot showing them in green
    plotting.plot_moc(E, V, BE, moc_lines, 'moc_lines.png')

    # Trim off the starting points and ending points - these don't matter
    for i in range(len(moc_lines)):
        moc_lines[i] = np.delete(moc_lines[i], (0, -1), axis=0)

    # Find the maximum number of reflections in all the MOC lines
    max_points = 0
    for i in range(len(moc_lines)):
        if moc_lines[i].shape[0] > max_points:
            max_points = moc_lines[i].shape[0]

    # Sort the points into list where each index is the nth reflection
    reflection_points = [np.empty((0, 2)) for i in range(max_points)]
    for i in range(len(moc_lines)):
        for j in range(moc_lines[i].shape[0]):
            reflection_points[j] = np.vstack((reflection_points[j], moc_lines[i][j, :]))

    zones = []
    # Create zones based upon the coordinates of each point
    for i in range(len(reflection_points)):
        # Need to check if reflection has an end point
        if i+1 >= len(reflection_points):
            continue
        # This means that there is a next point - point where reflection ends
        else:
            # x-coords of start of zone
            x_avg_start = reflection_points[i][:, 0].min() # Minimums - upstream dominated
            # Average all x-coords of end of zone
            x_avg_end = reflection_points[i + 1][:, 0].min() # Minimums - upstream dominated

            # Get y-min/max of zone
            y_min = np.min(np.hstack((reflection_points[i][:, 1].flatten(), reflection_points[i + 1][:, 1].flatten())))
            y_max = np.max(np.hstack((reflection_points[i][:, 1].flatten(), reflection_points[i + 1][:, 1].flatten())))

            # Make bounding box of zone
            bounding_box = np.array([[x_avg_start, y_min],
                                     [x_avg_start, y_max],
                                     [x_avg_end, y_min],
                                     [x_avg_end, y_max]])
            # Append to list of zones
            zones.append(bounding_box)

    additional_machs = [None for i in range(len(zones))]
    mach_upstream = M # Start with upstream Mach number as freestream Mach number
    for i in range(len(zones)):
        # Find angle diagonal makes with horizontal
        l, _ = cgf.edge_properties_calculator(zones[i][1], zones[i][2])
        theta = math.acos((zones[i][2, 0] - zones[i][1, 0]) / l) / 2 - (a * math.pi / 180)
        # Run OS calculation
        os_results = compressible_flow_formulas.obliqueshock(theta, mach_upstream, 1, 1, 1, y)
        additional_machs[i] = os_results['M2']
        # Set upstream to be post-OS
        mach_upstream = os_results['M2']

    # Find the centroid position of each cell for initialization
    centroids = cgf.centroid(E, V)
    state = np.zeros((E.shape[0], 4))

    # Check if centroid exists within the box of the zone that has been shocked, if so then apply local zone's Mach and
    # initialize using that local Mach number
    for i in range(state.shape[0]):
        for j in range(len(zones)):
            if zones[j][0, 0] <= centroids[i, 0] <= zones[j][2, 0] and \
                    zones[j][0, 1] <= centroids[i, 1] <= zones[j][1, 1]:
                state[i, :] = intlzn.init_state_mach(additional_machs[j], a, y)
                break
            else:
                state[i, :] = intlzn.init_state_mach(M, a, y)
    return state


def moc_inflow(V, BE, M, a):
    """Generates an N-length list of 2D arrays that contain the initial line segments left-running characteristic lines.

    :param V: Nx2 array of node coordinates, [x-pos, y-pos]
    :param BE: Nx4 array of boundary edge information, [node_A, node_B, cell_i, flag]
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :return: An initial list of characteristic lines that intersect the geometry and go from inflwo to outflow
    """
    # Mach angle
    mu = math.asin((1 / M))

    # Slope of characteristic lines - Pg 390 of JD Anderson Modern Comp Flow 3rd Edition
    dy_dx = math.tan(a * np.pi / 180 + mu)

    # Separate the wall boundary edges from the other boundary edges
    wall_BEs = []
    for be in BE:
        if be[3] == 0:
            wall_BEs.append(be)
    wall_BEs = np.array(wall_BEs)

    # Min/max y-positions of the wall boundary edges used to minimize the number of characteristic lines used
    min_wall_y = (V[wall_BEs[:, 0:2], 1]).min()
    max_wall_y = (V[wall_BEs[:, 0:2], 1]).max()

    # Make the initial left-running characteristic lines, delete the line if it is below minimum/above the maximum y-pos
    moc_lines = []
    for i in range(BE.shape[0]):
        if BE[i, 3] == 3:
            midpoint = (V[BE[i, 0]] + V[BE[i, 1]]) / 2
            delta_x = (V[:, 0]).max() - midpoint[0]
            y_final = midpoint[1] + dy_dx * delta_x
            temp_array = np.array([[midpoint[0], midpoint[1]], [(V[:, 0]).max(), y_final]])
            # If we're not intersecting with the geometry at all, skip the MOC line as it doesn't matter
            if midpoint[1] > max_wall_y or y_final < min_wall_y:
                continue
            moc_lines.append(temp_array)
        else:
            continue

    return moc_lines


def moc_reflect(V, BE, M, a, y, moc_lines):
    """Reflects the characteristic lines off the walls of the flow and continues to march them downstream. Assuming
    that incoming angle is the ramp angle (AoA + upstream Mach angle) and that the outgoing angle is AoA + downstream
    Mach angle.

    :param V: Nx2 array of node coordinates, [x-pos, y-pos]
    :param BE: Nx4 array of boundary edge information, [node_A, node_B, cell_i, flag]
    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :param moc_lines: N length list of Mx2 arrays that make up the coordinate pairs of line segments for MOC lines
    :returns: An updated list of characteristic lines that where each index of the list is an [:, 2] Numpy array of
     points in the path the characteristic line takes as it moves from the inflow to outflow and reflects off
     of the object in the flow
    """
    wall_BEs = []
    for be in BE:
        if be[3] == 0:
            wall_BEs.append(be)
    wall_BEs = np.array(wall_BEs)

    # Bounding box of the domain used in sanity-checking intersection points
    xlim = np.array([np.min(V[:, 0]), np.max(V[:, 0])])
    eps = 1e-12

    # Take each characteristic line and begin reflection process
    for i in range(len(moc_lines)):
        # Initial condition assumes all lines have at least one reflection with an interior wall and that reflection will
        # cause the line to go from a positive slope to negative slope (reflect_up false)
        reflecting = True
        while reflecting:
            # Assume no intersection points to start
            be_intersections = []
            be_intersection_edges = []
            for j in range(wall_BEs.shape[0]):
                # Find the intersection between the walls and MOC lines
                be_intersection = check_intersection2(moc_lines[i][-2, :], moc_lines[i][-1, :],
                                                      V[wall_BEs[j][0]], V[wall_BEs[j][1]])
                # If the intersection DNE - continue to next BE
                if be_intersection[0] is None: continue

                # Skip over boundary edges that lie on the intersection
                if np.linalg.norm(be_intersection - moc_lines[i][-2, :]) < eps: continue

                # Otherwise grab the boundary edge and its intersection point
                be_intersections.append(be_intersection)
                be_intersection_edges.append(wall_BEs[j])

            # If after all wall edges there is no intersection then the MoC line must be on the right side of the domain
            if len(be_intersections) == 0: reflecting = False
            # Still have to do this reflection if an intersection point exists
            else:
                # Left-most (minimum x) intersection point is the only intersection point wanted
                intersection_point = np.array(be_intersections[np.argmin(np.array(be_intersections)[:, 0])])
                intersection_be = wall_BEs[np.argmin(np.array(be_intersections)[:, 0])]
                # Now the end of the MoC line is the point of intersection
                moc_lines[i][-1, :] = intersection_point

                # Specular reflection @ the point of intersection using the unit vector of the upstream side, and the
                # normal vector of the boundary edge that is reflected on
                upstream_length = np.linalg.norm(moc_lines[i][-1, :] - moc_lines[i][-2, :])
                upstream_norm = (moc_lines[i][-1, :] - moc_lines[i][-2, :]) / upstream_length
                _, be_n = cgf.edge_properties_calculator(V[intersection_be[0]], V[intersection_be[1]])

                # Computes the normal vector after an assumed specular reflection
                downstream_norm = upstream_norm - 2 * be_n * np.dot(be_n, upstream_norm)
                downstream_slope = downstream_norm[1] / downstream_norm[0]

                # Projects the vector downstream of reflection to the edge of the domain
                dx = xlim[1] - moc_lines[i][-1, 0]
                dy = dx * downstream_slope
                moc_lines[i] = np.vstack((moc_lines[i], np.array([xlim[1], moc_lines[i][-1, 1] + dy])))

    return moc_lines


def check_intersection2(q, qs, p, pr):
    """Returns the point of intersection if the line segment from p1 -> p2 and p3 -> p4 intersect, None if not

    :param q: Start point of L1
    :param qs: End point of L1
    :param p: Start point of L2
    :param pr: End point of L2
    :returns: The point of intersection between the two line segments [x-intersection, y-intersection], or [None, None]
    """
    # "Parameterization" of the line segments
    s = qs - q
    r = pr - p

    # Setting intersection to be a linear system of equations
    RHS = p - q

    dx1 = s[0]
    dx2 = r[0]
    dy1 = s[1]
    dy2 = r[1]
    A = np.array(([[dx1, -dx2], [dy1, -dy2]]))

    # Attempt to solve - if system cannot be solved then we know it doesn't intersect
    try:
        x = np.linalg.solve(A, RHS)
    except:
        return np.array((None, None))
    else:
        x = np.linalg.solve(A, RHS)

    # Secondary check that makes sure the intersection happens inside the line segments and not farther away
    if 0 <= x[0] <= 1 and 0 <= x[1] <= 1:
        intersection = q + x[0] * s
        return intersection
    else:
        return np.array((None, None))

@njit(cache=True)
def check_intersection3(q, qs, p, pr, xlim, ylim):
    """Numerically finds the intersection between two line segments.
    :param q: Start point of L1
    :param qs: End point of L1
    :param p: Start point of L2
    :param pr: End point of L2
    :param xlim: Bounds of the domain in the x-dir [xmin, xmax]
    :param ylim: Bounds of the domain in the y-dir [ymin, ymax]
    :returns: The point of intersection between the two line segments [x-intersection, y-intersection], or [-Inf, -Inf]
    """

    # Number of steps to march along the parametric vectors
    n = 1e1

    # Two parametric vectors step sizes to go from the start -> end points
    dq = (qs - q) / n
    dp = (pr - p) / n

    # Setup marching vectors
    qtemp = q + 0
    ptemp = p + 0

    # Pre-allocate a distance vector to compare distance between the respective points in the marching vectors
    intersect_distance = np.zeros(np.int(n))

    for i in range(np.int(n)):
        # At each delta increment - compute the location along p and q and then compute the distance between the points
        qtemp += dq
        ptemp += dp
        intersect_distance[i] = np.sqrt(np.sum(np.square(qtemp - ptemp)))
    # Point of intersection should be the minimum distance
    intersection = q + dq * np.argmin(intersect_distance)

    # Run a sanity check to make sure intersection exists within the domain of the mesh
    if xlim[0] <= intersection[0] <= xlim[1] and ylim[0] <= intersection[1] <= ylim[1]:
        return intersection
    else:
        return np.array([-10e3, -10e3])