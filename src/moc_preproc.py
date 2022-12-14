import copy
import math
import numpy as np

import comp_flow
import mesh_processing


def initialize_moc(mesh, config):
    """Initializes initial condition using the MOC-OS approach where the characteristic lines and their reflections are
    used to find locations of oblique shocks.

    :param mesh: Read *.gri file
    :param config: Program runtime configuration
    :return:
    """
    state = np.zeros((mesh['E'].shape[0], 4))

    centroids = np.zeros((mesh['E'].shape[0], 2))

    moc_lines = moc_inflow(mesh, config)
    moc_lines = moc_reflect(mesh, config, moc_lines)

    for i in range(mesh['E'].shape[0]):
        centroids[i, :] = (mesh['V'][mesh['E'][i, 0]] + mesh['V'][mesh['E'][i, 1]] + mesh['V'][mesh['E'][i, 2]]) / 3

    for i in range(len(moc_lines)-1, -1, -1):
        # Only two points -> no reflections
        if moc_lines[i].shape[0] <= 2:
            del moc_lines[i]

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
        # This means that there is a next point (where reflection ends
        else:
            # x-coords of start of zones

            # x_avg_start = reflection_points[i].sum() / reflection_points[i].shape[0] # Avgs - downstream influenced
            x_avg_start = reflection_points[i][:, 0].min() # Minimums - upstream dominated
            # Average all x-coords of end
            # x_avg_end = reflection_points[i + 1].sum() / reflection_points[i + 1].shape[0] # Avgs - downstream influenced
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
    mach_upstream = config.M
    for i in range(len(zones)):
        # Find angle diagonal makes with horizontal
        l, _ = mesh_processing.edge_properties_calculator(zones[i][1], zones[i][2])
        theta = math.acos((zones[i][2, 0] - zones[i][1, 0]) / l) / 2 - (config.a * math.pi / 180)
        # Run OS calculation
        os_results = comp_flow.obliqueshock(theta, mach_upstream, 1, 1, 1, config.y)
        additional_machs[i] = os_results['M2']
        # Set upstream to be post-OS
        mach_upstream = os_results['M2']

    # Check all cell centroids, if it exists within the box of the zone, apply local zone's Mach number for initializing
    for i in range(state.shape[0]):
        for j in range(len(zones)):
            if zones[j][0, 0] <= centroids[i, 0] <= zones[j][2, 0] and \
                    zones[j][0, 1] <= centroids[i, 1] <= zones[j][1, 1]:
                state[i, :] = init_state_mach(additional_machs[j], config)
                break
            else:
                state[i, :] = init_state_mach(config.M, config)

    return state


def moc_inflow(mesh, config):
    """Generates a N-length list of 2D arrays that contain the initial line segments left-running characteristic lines.

    :param mesh: Read in *.gri mesh
    :param config: Config file
    :return:
    """
    mu = math.asin((1 / config.M))  # Mach angle
    theta = config.a  # By definition AoA = theta
    dy_dx = math.tan(theta - mu)  # Formula for slope of MoC Lines

    wall_BEs = []
    for be in mesh['BE']:
        if mesh['Bname'][be[3]] == 'Wall':
            wall_BEs.append(be)
    wall_BEs = np.array(wall_BEs)

    min_wall_y = (mesh['V'][wall_BEs[:, 0:2], 1]).min()
    max_wall_y = (mesh['V'][wall_BEs[:, 0:2], 1]).max()

    moc_lines = []
    for i in range(mesh['BE'].shape[0]):
        if mesh['Bname'][mesh['BE'][i, 3]] == 'Inflow':
            midpoint = (mesh['V'][mesh['BE'][i, 0]] + mesh['V'][mesh['BE'][i, 1]]) / 2
            delta_x = (mesh['V'][:, 0]).max() - midpoint[0]
            y_final = midpoint[1] + dy_dx * delta_x
            temp_array = np.array([[midpoint[0], midpoint[1]], [(mesh['V'][:, 0]).max(), y_final]])
            # If we're not intersecting with the geometry at all, skip the MOC line as it doesn't matter
            if midpoint[1] > max_wall_y or y_final < min_wall_y:
                continue
            moc_lines.append(temp_array)
        else:
            continue

    return moc_lines


def moc_reflect(mesh, config, moc_lines):
    """Specularly reflects the characteristic lines off the walls of the flow and continues to march them downstream.

    :param mesh: Mesh in *.gri format
    :param config: Operating conditions
    :param moc_lines: N length list of Mx2 arrays that make up the coordinate pairs of line segments for MOC lines
    :return:
    """
    mu = math.asin((1 / config.M))  # Mach angle
    theta = config.a  # By definition AoA = theta
    dy_dx = math.tan(theta - mu)  # Formula for slope of MoC Lines

    wall_BEs = []
    for be in mesh['BE']:
        if mesh['Bname'][be[3]] == 'Wall':
            wall_BEs.append(be)
    wall_BEs = np.array(wall_BEs)

    eps = 1e-12

    not_right = True
    while not_right:
        moc_right_count = 0

        be_norms = [None for i in range(len(moc_lines))]
        # Loop handles intersection for upward streamlines
        for i in range(len(moc_lines)):
            intersection_point = moc_lines[i][-1, :]
            for be in wall_BEs:
                be_l, be_n = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

                # Find the intersection between the BE and MOC line
                be_intersection = check_intersection2(moc_lines[i][-2, :], moc_lines[i][-1, :],
                                                      mesh['V'][be[0]], mesh['V'][be[1]])
                # If the intersection DNE - continue to next BE
                if be_intersection[0] is None:
                    continue

                # If somehow get the same point for intersection, skip and continue
                if (abs(moc_lines[i][-2, :] - be_intersection) < eps).any():
                    continue

                # Check the intersection point and make sure that it is on the BE - if both lengths are less than BE
                # length the intersection point must be between them to some degree
                l1, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], be_intersection)
                l2, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[1]], be_intersection)
                if l1 > be_l and l2 > be_l:
                    continue

                # If the BE is further left than previous, then it is the first intersection edge
                if be_intersection[0] < intersection_point[0]:
                    intersection_point = be_intersection
                    moc_lines[i][-1, :] = be_intersection
                    be_norms[i] = be_n

        # Reflect down
        for i in range(len(moc_lines)):
            # Check if the last x-position is at right side of mesh
            if abs(moc_lines[i][-1, 0] - (mesh['V'][:, 0]).max()) < eps:
                continue

            # Otherwise project the streamlines down
            moc_length = np.linalg.norm(moc_lines[i][-1, :] - moc_lines[i][-2, :])
            moc_in = (moc_lines[i][-2, :] - moc_lines[i][-1, :]) / moc_length
            moc_out = moc_in - 2 * be_norms[i] * np.dot(be_norms[i], moc_in)
            slope_out = moc_out[1] / moc_out[0]
            delta_x = (mesh['V'][:, 0]).max() - moc_lines[i][-1, 0]
            y_final = moc_lines[i][-1, 1] + slope_out * delta_x
            moc_lines[i] = np.vstack((moc_lines[i], [(mesh['V'][:, 0]).max(), y_final]))

        be_norms = [None for i in range(len(moc_lines))]
        # Loop handles intersection for downward streamlines
        for i in range(len(moc_lines)):
            intersection_point = moc_lines[i][-1, :]
            for be in wall_BEs:
                be_l, be_n = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

                # Find the intersection between the BE and MOC line
                be_intersection = check_intersection2(moc_lines[i][-2, :], moc_lines[i][-1, :],
                                                      mesh['V'][be[0]], mesh['V'][be[1]])
                # If the intersection DNE - continue to next BE
                if be_intersection[0] is None:
                    continue

                # If somehow get the same point for intersection, skip and continue
                if (abs(moc_lines[i][-2, :] - be_intersection) < eps).any():
                    continue

                # Check the intersection point and make sure that it is on the BE - if both lengths are less than BE
                # length the intersection point must be between them to some degree
                l1, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], be_intersection)
                l2, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[1]], be_intersection)
                if l1 > be_l and l2 > be_l:
                    continue

                # If the BE is further left than previous, then it is the first intersection edge
                if be_intersection[0] <= intersection_point[0]:
                    intersection_point = be_intersection
                    moc_lines[i][-1, :] = be_intersection
                    be_norms[i] = be_n

        # Reflect up
        for i in range(len(moc_lines)):
            # Check if the last x-position is at right side of mesh
            if abs(moc_lines[i][-1, 0] - (mesh['V'][:, 0]).max()) < eps:
                continue

            # Otherwise project the streamlines down
            moc_length = np.linalg.norm(moc_lines[i][-1, :] - moc_lines[i][-2, :])
            moc_in = (moc_lines[i][-2, :] - moc_lines[i][-1, :]) / moc_length
            moc_out = moc_in - 2 * be_norms[i] * np.dot(be_norms[i], moc_in)
            slope_out = moc_out[1] / moc_out[0]
            delta_x = (mesh['V'][:, 0]).max() - moc_lines[i][-1, 0]
            y_final = moc_lines[i][-1, 1] + slope_out * delta_x
            moc_lines[i] = np.vstack((moc_lines[i], [(mesh['V'][:, 0]).max(), y_final]))

        # Find out how many MOC lines are on the right side of the domain
        moc_right_count = len(moc_lines)
        for i in range(len(moc_lines)):
            intersected = False
            intersection_point = moc_lines[i][-1, :]
            for be in wall_BEs:
                be_l, be_n = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

                # Find the intersection between the BE and MOC line
                be_intersection = check_intersection2(moc_lines[i][-2, :], moc_lines[i][-1, :],
                                                      mesh['V'][be[0]], mesh['V'][be[1]])
                # If the intersection DNE - continue to next BE
                if be_intersection[0] is None:
                    continue

                # If somehow get the same point for intersection, skip and continue
                if (abs(moc_lines[i][-2, :] - be_intersection) < eps).any():
                    continue

                # Check the intersection point and make sure that it is on the BE - if both lengths are less than BE
                # length the intersection point must be between them to some degree
                l1, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[0]], be_intersection)
                l2, _ = mesh_processing.edge_properties_calculator(mesh['V'][be[1]], be_intersection)
                if l1 > be_l and l2 > be_l:
                    continue

                # If the BE is further left than previous, then it is the first intersection edge
                if be_intersection[0] <= intersection_point[0]:
                    intersected = True

            if intersected:
                moc_right_count -= 1

        # If all are at right side of domain, then break the reflection process
        if moc_right_count == len(moc_lines):
            not_right = False

    return moc_lines


def check_intersection2(q, qs, p, pr):
    """Returns the point of intersection if the line segment from p1 -> p2 and p3 -> p4 intersect, None if not

    :param q: Start point of L1
    :param qs: End point of L1
    :param p: Start point of L2
    :param pr: End point of L2
    :return:
    """
    s = qs - q
    r = pr - p
    RHS = p - q

    dx1 = s[0]
    dx2 = r[0]
    dy1 = s[1]
    dy2 = r[1]

    A = np.array(([[dx1, -dx2], [dy1, -dy2]]))

    try:
        x = np.linalg.solve(A, RHS)
    except:
        return np.array((None, None))
    else:
        x = np.linalg.solve(A, RHS)

    if 0 <= x[0] <= 1 and 0 <= x[1] <= 1:
        intersection = q + x[0] * s
        return intersection
    else:
        return np.array((None, None))


def init_state_mach(m, config):
    """Initializes the state vector for a local mach number in that state.

    :param m: Local cell's Mach number
    :return:
    """
    initial_condition = np.zeros((4))

    initial_condition[0] = 1  # Rho
    initial_condition[1] = m * math.cos(config.a * math.pi / 180)# Rho*U
    initial_condition[2] = m * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[3] = 1 / (config.y - 1) / config.y + (m ** 2) / 2 # Rho*E

    return initial_condition