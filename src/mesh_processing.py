import itertools

import numpy as np
import math
import copy
import flux
import readgri


def centroid(cells, vertices):
    """Returns an array of X and Y coordinates for each cell.

    :param cells: NP array of vertex coordinate locations
    :param vertices: NP array of elements with their vertices
    :return: X, Y - NP arrays that contain X-Y coordinate pairs of "centroid" of the cell
    """
    X = np.zeros((len(cells)))
    Y = np.zeros((len(cells)))

    for K in range(len(cells)):
        # X & Y coordinates for each cell's vertices
        x = vertices[cells[K]][:, 0]
        y = vertices[cells[K]][:, 1]

        # Forumla for centroid of a triangle:
        # https://byjus.com/maths/centroid-of-a-triangle/
        X[K] = np.sum(x) / len(x)
        Y[K] = np.sum(y) / len(y)

    return X, Y


def edge_properties_calculator(node_a, node_b):
    """ Calculates the length and CCW norm out of a single edge

    :param node_a: X-Y Coordinates of node A
    :param node_b: X-Y Coordinates of node B
    :return length: Length of the edge from A->B
    :return norm: Normal vector out of the edge in CCW fashion: [nx, ny]
    """
    length = math.sqrt((node_b[0] - node_a[0]) ** 2 + (node_b[1] - node_a[1]) ** 2)
    norm = np.array([(node_b[1] - node_a[1]) / length, (node_a[0] - node_b[0]) / length])

    return length, norm


def area_calculator(mesh):
    """Calculates the area of the two triangular cells for the given indices.

    :param mesh: The mesh of the problem holding all cells.
    :return area: Area of the cells
    """
    area = np.zeros(mesh['E'].shape[0])
    for i in range(mesh['E'].shape[0]):
        nodes = mesh['E'][i]
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        # print(cellIndex)
        a, _ = edge_properties_calculator(mesh['V'][nodes[0]], mesh['V'][nodes[1]])
        b, _ = edge_properties_calculator(mesh['V'][nodes[1]], mesh['V'][nodes[2]])
        c, _ = edge_properties_calculator(mesh['V'][nodes[2]], mesh['V'][nodes[0]])

        s = (a + b + c) / 2

        area[i] = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return area


def initialize_boundary(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = config.M * math.cos(config.a * math.pi / 180) # Rho*U
    initial_condition[:, 2] = config.M * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (config.y - 1) / config.y + (config.M) ** 2 / 2 # Rho*E

    return initial_condition

def initialize_boundary_weak(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    m = config.M / 2
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = m * math.cos(config.a * math.pi / 180) # Rho*U
    initial_condition[:, 2] = m * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (config.y - 1) / config.y + (m) ** 2 / 2 # Rho*E

    return initial_condition


def initialize_boundary_dist(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((mesh['E'].shape[0], 4))
    x_min = (mesh['V'][:, 0]).min()
    x_max = (mesh['V'][:, 0]).max()
    y_min = (mesh['V'][:, 1]).min()
    y_max = (mesh['V'][:, 1]).max()

    for i in range(mesh['E'].shape[0]):
        centroid = (mesh['V'][mesh['E'][i, 0]] + mesh['V'][mesh['E'][i, 1]] + mesh['V'][mesh['E'][i, 2]]) / 3

        if centroid[0] <= 0:
            x_scale = centroid[0] / x_min
        else:
            x_scale = centroid[0] / x_max

        if centroid[1] <= 0:
            y_scale = centroid[1] / y_min
        else:
            y_scale = centroid[1] / y_max

        avg_scale = (x_scale + y_scale) / 2

        m_init = config.M * avg_scale

        initial_condition[i, 0] = 1  # Rho
        initial_condition[i, 1] = m_init * math.cos(config.a * math.pi / 180) # Rho*U
        initial_condition[i, 2] = m_init * math.sin(config.a * math.pi / 180) # Rho*V
        initial_condition[i, 3] = 1 / (config.y - 1) / config.y + (m_init) ** 2 / 2 # Rho*E

    return initial_condition


def reorient_ccw(node1, node2, node3, node_list):
    """Re-orients the given set of nodes to be in a counter-clockwise order

    :param node1: first node index
    :param node2: second node index
    :param node3: third node index
    :param node_list: array of node coordinate pairs
    :return: Returns a numpy array w/ the same node indices but in CCW order
    """
    node1_coord = node_list[node1]
    node2_coord = node_list[node2]
    node3_coord = node_list[node3]

    # https://www.geeksforgeeks.org/orientation-3-ordered-points/
    if ((node2_coord[1] - node1_coord[1]) * (node3_coord[0] - node2_coord[0]) -\
            (node3_coord[1] - node2_coord[1]) * (node2_coord[0] - node1_coord[0])) < 0:
        return np.array([node1, node2, node3])
    else:
        return np.array([node3, node2, node1])


def refine_interp_uniform(mesh, state, fname, config):
    """Performs adaptive mesh refinement on the given mesh from the results of the state vector and large jumps in Mach
    number across cell boundaries. Divides flagged cells uniformly, cells w/ >= 2 neighbors flagged uniformly, and cells
    with only 1 edge flagged.

    :param mesh:
    :param state:
    :param config:
    :return:
    """
    flag_cell, split_cell, flag_be, flag_ie, split_be, split_ie = find_uniform_splitting(state, mesh, config)

    new_element_ordering = np.empty((1, 3), dtype=int)
    new_formed_elements = np.empty((1, 3), dtype=int)
    # Perform uniform cell refinement on flagged cells
    for i in range(mesh['E'].shape[0]):
        # If the cell is not flagged in either category it is not changing so add it to the ordering and continue
        if (not i in flag_cell) and (not i in split_cell):
            new_element_ordering = np.vstack((new_element_ordering, mesh['E'][i]))
            continue

        # If the cell is flagged for uniform splitting, perform that operation via 1 -> 4 split where cell i is center
        if i in flag_cell:
            cell_nodes = mesh['E'][i]

            # Midpoints of the cells
            midpoint1 = (mesh['V'][cell_nodes[0]] + mesh['V'][cell_nodes[1]]) / 2
            if not np.any(np.equal(midpoint1, mesh['V']).all(1)): mesh['V'] = np.vstack((mesh['V'], midpoint1))
            midpoint1_index = np.where(np.equal(midpoint1, mesh['V']).all(1))[0][0]

            midpoint2 = (mesh['V'][cell_nodes[1]] + mesh['V'][cell_nodes[2]]) / 2
            if not np.any(np.equal(midpoint2, mesh['V']).all(1)): mesh['V'] = np.vstack((mesh['V'], midpoint2))
            midpoint2_index = np.where(np.equal(midpoint2, mesh['V']).all(1))[0][0]

            midpoint3 = (mesh['V'][cell_nodes[2]] + mesh['V'][cell_nodes[0]]) / 2
            if not np.any(np.equal(midpoint3, mesh['V']).all(1)): mesh['V'] = np.vstack((mesh['V'], midpoint3))
            midpoint3_index = np.where(np.equal(midpoint3, mesh['V']).all(1))[0][0]

            # Center cell replaces the i'th index for uniform cells
            center_triangle = np.array([midpoint1_index, midpoint2_index, midpoint3_index])
            new_element_ordering = np.vstack((new_element_ordering, center_triangle))

            # New triangles go to the "new elements" which are appended to the back of the element ordering
            new_triangle1 = np.array([cell_nodes[0], midpoint1_index, midpoint3_index])
            new_triangle2 = np.array([cell_nodes[1], midpoint1_index, midpoint2_index])
            new_triangle3 = np.array([cell_nodes[2], midpoint2_index, midpoint3_index])
            new_formed_elements = np.vstack((new_formed_elements, new_triangle1, new_triangle2, new_triangle3))

            # Add states to back of state vector to account for new elements being formed
            state = np.vstack((state, state[i], state[i], state[i]))
            continue

        # If the cell is flagged for split down the edge - perform that operation and make 2 new cells
        if i in split_cell:
            cell_nodes = mesh['E'][i]

            # Only one edge is flagged - it is either an internal edge or a boundary edge - need to find right the
            # right one, so we can get the right node that's already been created and assemble triangles accordingly
            if np.any(np.equal(i, split_ie[:, 2])):
                split_edge = split_ie[np.where(np.equal(i, split_ie[:, 2]))[0][0]]
            if np.any(np.equal(i, split_ie[:, 3])):
                split_edge = split_ie[np.where(np.equal(i, split_ie[:, 3]))[0][0]]

            # Grab the midpoint of the recently found split edge
            midpoint_split = (mesh['V'][split_edge[0]] + mesh['V'][split_edge[1]]) / 2
            if not np.any(np.equal(midpoint_split, mesh['V']).all(1)): mesh['V'] = np.vstack((mesh['V'], midpoint_split))
            midpoint_split_index = np.where(np.equal(midpoint_split, mesh['V']).all(1))[0][0]

            # Find the node that forms the bisector with the edge
            common_node = list(cell_nodes)
            common_node.remove(split_edge[0])
            common_node.remove(split_edge[1])

            # One cell replaces ith cell, other is appended to back of the array
            split_cell_1 = np.array([split_edge[0], common_node[0], midpoint_split_index])
            split_cell_2 = np.array([split_edge[1], common_node[0], midpoint_split_index])

            new_element_ordering = np.vstack((new_element_ordering, split_cell_1))
            new_formed_elements = np.vstack((new_formed_elements, split_cell_2))

            state = np.vstack((state, state[i]))
            continue
    new_element_ordering = np.delete(new_element_ordering, 0, axis=0)
    new_formed_elements = np.delete(new_formed_elements, 0, axis=0)
    new_elements_full = np.vstack((new_element_ordering, new_formed_elements))

    # Write out the mesh to fname.gri and then parse it again using the given reading/hashing functionaility
    f = open(fname, 'w')

    # gri header
    f.write('{0} {1} 2\n'.format(mesh['V'].shape[0], new_elements_full.shape[0]))

    # Node information
    for node in mesh['V']:
        f.write('%.15e %.15e\n'%(node[0], node[1]))

    # Boundary edge information
    f.write('{0}\n'.format(len(mesh['Bname'])))

    boundary_pairs = [[] for name in mesh['Bname']]

    # Boundary pairs writing
    for be in mesh['BE']:
        if np.any(np.equal(be, flag_be).all(1)) or np.any(np.equal(be, flag_ie).all(1)):
            midpoint = (mesh['V'][be[0]] + mesh['V'][be[1]]) / 2
            midpoint_index = np.where(np.equal(midpoint, mesh['V']).all(1))[0][0]
            if not [be[0], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append([be[0], midpoint_index])
            if not [be[1], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append([be[1], midpoint_index])
            continue
        boundary_pairs[be[3]].append([be[0], be[1]])
    for i in range(len(mesh['Bname'])):
        f.write('{0} 2 {1}\n'.format(len(boundary_pairs[i]), mesh['Bname'][i]))
        for pair in boundary_pairs[i]:
            f.write('{0} {1}\n'.format(pair[0]+1, pair[1]+1))

    # Element writing
    f.write('{0} 1 TriLagrange\n'.format(new_elements_full.shape[0]))
    for element in new_elements_full:
        temp_element = reorient_ccw(element[0], element[1], element[2], mesh['V'])
        f.write('{0} {1} {2}\n'.format(temp_element[0]+1, temp_element[1]+1, temp_element[2]+1))
    f.close()

    # Generate the new mesh using the hasing provided by the reading of the gri file
    mesh = readgri.readgri(fname)

    return mesh, state


def find_uniform_splitting(state, mesh, config):
    """Creates a unique array of cell indices that are to be split uniformly and split to maintain conformity. Any cell
    that has 2 uniform neighbors becomes a uniform cell, this process continues until either all cells are uniform or
    a cell no longer has 2 uniform neighbors and can be split in two.

    :param state: Nx4 numpy array of state vectors
    :param mesh: Current working mesh
    :param config: Simulation runtime config
    :return: array of flagged cell indices, array of split cell indices, array of flagged be, array of flagged ie
    """
    # Nx2 array [error, cell i]
    error_index_ie = np.zeros((len(mesh['IE']), 2))
    error_index_be = np.zeros((len(mesh['BE']), 2))

    for i in range(len(mesh['BE'])):
        # Boundary Edges
        be = mesh['BE'][i]
        # No error on the inflow cells
        if mesh['Bname'][be[3]] == 'Inflow' or mesh['Bname'][be[3]] == 'Outflow':
            continue
        else:
            be_l, be_n = edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

            # Cell i quantities
            u = state[be[2]][1] / state[be[2]][0]
            v = state[be[2]][2] / state[be[2]][0]
            q = np.dot(be_n, np.array([u, v]))
            flux_c = flux.F_euler_2d(state[be[2]], config)
            h_l = (flux_c[3, 0] + flux_c[3, 1]) / ( flux_c[0, 0] + flux_c[0, 1])
            c = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))

            error = abs(q / c) * be_l

            error_index_be[i, 0] = error
            error_index_be[i, 1] = be[2]

    for i in range(len(mesh['IE'])):
        # Internal Edges
        ie = mesh['IE'][i]
        ie_l, ie_n = edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])

        # Left cell/cell i quantities
        u_l = state[ie[2]]
        u_l_u = u_l[1] / u_l[0]
        u_l_v = u_l[2] / u_l[0]
        q_l = np.linalg.norm([u_l_u, u_l_v])
        flux_l = flux.F_euler_2d(u_l, config)
        h_l = (flux_l[3, 0] + flux_l[3, 1]) / ( flux_l[0, 0] + flux_l[0, 1])
        c_l = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
        m_l = q_l / c_l

        # Right cell/cell N quantities
        u_r = state[ie[3]]
        u_r_u = u_r[1] / u_r[0]
        u_r_v = u_r[2] / u_r[0]
        q_r = np.linalg.norm([u_r_u, u_r_v])
        flux_r = flux.F_euler_2d(u_r, config)
        h_r = (flux_r[3, 0] + flux_r[3, 1]) / ( flux_r[0, 0] + flux_r[0, 1])
        c_r = math.sqrt((config.y - 1) * (h_r - q ** 2 / 2))

        m_r = q_r / c_r

        error = abs(m_l - m_r) * ie_l

        error_index_ie[i, 0] = error
        error_index_ie[i, 1] = ie[2]

    # Total list of all locations where Mach number jumps are too high
    total_error_index = np.vstack((error_index_be, error_index_ie))
    error_index = np.flip(total_error_index[total_error_index[:, 0].argsort()], axis=0)
    error_index = error_index[0:math.floor(config.adaptation_percentage * len(error_index[:, 0])), 1]

    # List of all unique cell locations where we want to refine and divide the triangle
    flagged_cells = []
    for cell in error_index:
        if not (cell in flagged_cells):
            flagged_cells.append(int(cell))

    flagged_be = np.empty((1, 4), dtype=int)
    flagged_ie = np.empty((1, 4), dtype=int)
    # Now go through and find all the flagged internal edges and boundary edges
    for cell in flagged_cells:
        if cell in mesh['BE'][:, 2]:
            be_index = np.where(np.equal(cell, mesh['BE'][:, 2]))[0]
            for index in be_index:
                if not np.any(np.equal(mesh['BE'][index], flagged_be).all(1)):
                    flagged_be = np.vstack((flagged_be, mesh['BE'][index]))
    for cell in flagged_cells:
        cell_nodes = mesh['E'][cell]
        # Use pairs of nodes to find the IEs of the cell that's flagged
        nodepairs = np.array([[cell_nodes[0], cell_nodes[1]],
                             [cell_nodes[1], cell_nodes[0]],
                             [cell_nodes[1], cell_nodes[2]],
                             [cell_nodes[2], cell_nodes[1]],
                             [cell_nodes[2], cell_nodes[0]],
                             [cell_nodes[0], cell_nodes[2]]])
        for row in nodepairs:
            if np.any(np.equal(row, mesh['IE'][:, 0:2]).all(1)):
                ie_index = np.where(np.equal(row, mesh['IE'][:, 0:2]).all(1))[0][0]
                if not np.any(np.equal(mesh['IE'][ie_index], flagged_ie).all(1)):
                    flagged_ie = np.vstack((flagged_ie, mesh['IE'][ie_index]))
    flagged_be = np.delete(flagged_be, 0, axis=0)
    flagged_ie = np.delete(flagged_ie, 0, axis=0)

    checking_neighbors = True
    while (checking_neighbors):
        num_flagged_edges = np.zeros((mesh['E'].shape[0]), dtype=int)
        for i in range(mesh['E'].shape[0]):
            if i in flagged_cells: continue
            num_flagged_edges[i] += len(np.where(np.equal(flagged_be[:, 2], i))[0])
            num_flagged_edges[i] += len(np.where(np.equal(flagged_ie[:, 2], i))[0])
            num_flagged_edges[i] += len(np.where(np.equal(flagged_ie[:, 3], i))[0])
            if num_flagged_edges[i] >= 2 and not i in flagged_cells:
                flagged_cells.append(i)
                # Update the flagged edges once again to account for addition to flagged cells
                for cell in flagged_cells:
                    if cell in mesh['BE'][:, 2]:
                        be_index = np.where(np.equal(cell, mesh['BE'][:, 2]))[0]
                        for index in be_index:
                            if not np.any(np.equal(mesh['BE'][index], flagged_be).all(1)):
                                flagged_be = np.vstack((flagged_be, mesh['BE'][index]))
                for cell in flagged_cells:
                    cell_nodes = mesh['E'][cell]
                    # Use pairs of nodes to find the IEs of the cell that's flagged
                    nodepairs = np.array([[cell_nodes[0], cell_nodes[1]],
                                          [cell_nodes[1], cell_nodes[0]],
                                          [cell_nodes[1], cell_nodes[2]],
                                          [cell_nodes[2], cell_nodes[1]],
                                          [cell_nodes[2], cell_nodes[0]],
                                          [cell_nodes[0], cell_nodes[2]]])
                    for row in nodepairs:
                        if np.any(np.equal(row, mesh['IE'][:, 0:2]).all(1)):
                            ie_index = np.where(np.equal(row, mesh['IE'][:, 0:2]).all(1))[0][0]
                            if not np.any(np.equal(mesh['IE'][ie_index], flagged_ie).all(1)):
                                flagged_ie = np.vstack((flagged_ie, mesh['IE'][ie_index]))
        # If the only cells left have only 1 flagged edge left, then we know it is a cell to be split and can exit loop
        if not np.any(np.greater_equal(num_flagged_edges, 2)): checking_neighbors = False
    split_cells = np.where(np.equal(num_flagged_edges, 1))[0]

    # Only IE can be split - find the cells that only have 1 edge in the flagged IE and split them on that edge
    flag_ie_split = np.empty((1, 4), dtype=int)
    for edge in flagged_ie:
        if edge[2] in split_cells or edge[3] in split_cells:
            if not np.any(np.equal(edge, flag_ie_split).all(1)):
                flag_ie_split = np.vstack((flag_ie_split, edge))
    flag_ie_split = np.delete(flag_ie_split, 0, axis=0)

    return flagged_cells, split_cells, flagged_be, flagged_ie, None, flag_ie_split


# Anything below this line is doesn't work/WIP/is useless - keeping for now, may revisit later
#-----------------------------------------------------------------------------------------------------------------------
def angle_loc(node1, node2, node3, node_coords):
    """Finds the angle for node1 using the Law of Cosines. Lengths of sides come from
    looking up the coordinates of the nodes.

    :param node1: Node index and location in the triangle where the angle is computed
    :param node2: Secondary node index
    :param node3: Tertiarty node index
    :param node_coords: Array of node coordinates
    :return:
    """
    length1, _ = edge_properties_calculator(node_coords[node2], node_coords[node3])
    length2, _ = edge_properties_calculator(node_coords[node1], node_coords[node3])
    length3, _ = edge_properties_calculator(node_coords[node1], node_coords[node2])

    angle = math.acos((length1 ** 2 - length2 ** 2 - length3 ** 2) / (-2 * length2 * length3))

    return angle

def refine_interp(mesh, state, config):
    """Refines the mesh based upon jumps in Mach number across cell boundaries. Also interpolates the solution to the
    new mesh by copying the value from one cell to the next.

    :param mesh: Dictionary of the mesh in the KFID-GRI format
    :param state: Nx4 array of state vectors used to calculate Mach numbers
    :param config: Configuration class for the simulation which has working fluid information
    :return:
    """
    # Nx2 array with errors in the cell index in question
    error_index_ie = np.zeros((len(mesh['IE']), 2))
    error_index_be = np.zeros((len(mesh['BE']), 2))

    for i in range(len(mesh['BE'])):
        # Boundary Edges
        be = mesh['BE'][i]
        # No error on the inflow cells
        if mesh['Bname'][be[3]] == 'Inflow' or mesh['Bname'][be[3]] == 'Outflow':
            continue
        else:
            be_l, be_n = edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

            # Cell i quantities
            u = state[be[2]][1] / state[be[2]][0]
            v = state[be[2]][2] / state[be[2]][0]
            q = np.dot(be_n, np.array([u, v]))
            flux_c = flux.F_euler_2d(state[be[2]], config)
            h_l = (flux_c[3, 0] + flux_c[3, 1]) / (flux_c[0, 0] + flux_c[0, 1])
            c = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
            # c = helper.mach_calc_single(state[be[2]], config)

            error = abs(q / c) * be_l

            error_index_be[i, 0] = error
            error_index_be[i, 1] = be[2]

    for i in range(len(mesh['IE'])):
        # Internal Edges
        ie = mesh['IE'][i]
        ie_l, ie_n = edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])

        # Left cell/cell i quantities
        u_l = state[ie[2]]
        u_l_u = u_l[1] / u_l[0]
        u_l_v = u_l[2] / u_l[0]
        q_l = np.linalg.norm([u_l_u, u_l_v])
        flux_l = flux.F_euler_2d(u_l, config)
        h_l = (flux_l[3, 0] + flux_l[3, 1]) / (flux_l[0, 0] + flux_l[0, 1])
        c_l = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
        # This formulation finds speed of sound as function of H = E + p/rho
        # c_l = helper.mach_calc_single(u_l, config)
        m_l = q_l / c_l

        # Right cell/cell N quantities
        u_r = state[ie[3]]
        u_r_u = u_r[1] / u_r[0]
        u_r_v = u_r[2] / u_r[0]
        q_r = np.linalg.norm([u_r_u, u_r_v])
        flux_r = flux.F_euler_2d(u_r, config)
        h_r = (flux_r[3, 0] + flux_r[3, 1]) / (flux_r[0, 0] + flux_r[0, 1])
        c_r = math.sqrt((config.y - 1) * (h_r - q ** 2 / 2))

        # c_r = helper.mach_calc_single(u_r, config)

        m_r = q_r / c_r

        error = abs(m_l - m_r) * ie_l

        error_index_ie[i, 0] = error
        error_index_ie[i, 1] = ie[2]

    # Total list of all locations where Mach number jumps are too high
    total_error_index = np.vstack((error_index_be, error_index_ie))
    error_index = np.flip(total_error_index[total_error_index[:, 0].argsort()], axis=0)
    error_index = error_index[0:math.floor(config.adaptation_percentage * len(error_index[:, 0])), 1]

    # List of all unique cell locations where we want to refine and divide the triangle
    flagged_cells = []
    for cell in error_index:
        if not (cell in flagged_cells):
            flagged_cells.append(int(cell))
    flagged_cells = np.array(flagged_cells)

    # Make a copy of the sections of the mesh that are going to be edited with deletion - by making a full copy, we
    # can reference the original without modifying the copy
    new_elements = copy.deepcopy(mesh['E'])
    new_ie = copy.deepcopy(mesh['IE'])
    new_be = copy.deepcopy(mesh['BE'])

    # Index lists that keep track of what rows need to be deleted later on
    delete_index_cells = []
    delete_index_be = []
    delete_index_ie = []

    # Generate new list of cells and the nodes on boundaries and use edgehash for our making edges
    for flagged_cell in flagged_cells:
        og_nodes = np.array(mesh['V'][mesh['E'][flagged_cell]])

        # New set of nodes to add to the 'V' array
        new_v1 = [(og_nodes[0][0] + og_nodes[1][0]) / 2, (og_nodes[0][1] + og_nodes[1][1]) / 2]
        new_v2 = [(og_nodes[1][0] + og_nodes[2][0]) / 2, (og_nodes[1][1] + og_nodes[2][1]) / 2]
        new_v3 = [(og_nodes[2][0] + og_nodes[0][0]) / 2, (og_nodes[2][1] + og_nodes[0][1]) / 2]
        new_nodes = np.array([new_v1, new_v2, new_v3])

        # Old total # of nodes used for indexing
        old_v_len = len(mesh['V'][:, 0])

        # Add new nodes to the node list - only ever appending so don't need to do anything fancy
        mesh['V'] = np.concatenate((mesh['V'], new_nodes))

        # Four new elements as a mix of the original nodes and the 3 new nodes
        # Have to use index of new nodes, so its old length + 3 but also  -1 to account for zero-indexing
        # Vn1 = old length, Vn2 = old length + 1, Vn3 = old length + 2
        new_e1 = np.array([mesh['E'][flagged_cell][0], old_v_len, old_v_len + 2])
        new_e2 = np.array([old_v_len, mesh['E'][flagged_cell][1], old_v_len + 1])
        new_e3 = np.array([old_v_len + 1, mesh['E'][flagged_cell][2], old_v_len + 2])
        new_e4 = np.array([old_v_len, old_v_len + 1, old_v_len + 2])

        # Four new state vectors, one per element
        new_state = np.array([state[flagged_cell], state[flagged_cell], state[flagged_cell], state[flagged_cell]])

        # Remove the old element & state from the list and add the new ones
        delete_index_cells.append(np.where(np.all(new_elements == mesh['E'][flagged_cell], axis=1))[0])
        new_elements = np.concatenate((new_elements, np.array([new_e1, new_e2, new_e3, new_e4])))
        state = np.concatenate((state, new_state))

        # Split the edges based on if the cell is a boundary cell or if it is an internal cell
        if flagged_cell in mesh['BE'][:, 2]:
            ## Boundary cells

            # Node indicies for original nodes
            og_nodes_index = mesh['E'][flagged_cell]

            # Finding which "other" cell is needed using some indexing, i-j-k refer to picture
            og_edges = np.array([[og_nodes_index[0], og_nodes_index[1]],
                                 [og_nodes_index[1], og_nodes_index[2]],
                                 [og_nodes_index[2], og_nodes_index[0]]])

            # Know that 1 cell is a boundary edge - that must be found first - other two are then IE
            # TODO: REPLACE LOOPS W/ INDEXING
            # Don't know which edge is the actual boundary edge - have to check all 3, but only one of them is
            for be in mesh['BE']:
                if np.all(og_edges[0] == be[0:2]) or np.all(np.flip(og_edges[0]) == be[0:2]) and be[2] == flagged_cell:
                    boundary_edge = be
                    delete_be = 0
                    delete_index_be.append(np.where(np.all(be == mesh['BE'], axis=1))[0])
                if np.all(og_edges[1] == be[0:2]) or np.all(np.flip(og_edges[1]) == be[0:2]) and be[2] == flagged_cell:
                    boundary_edge = be
                    delete_be = 1
                    delete_index_be.append(np.where(np.all(be == mesh['BE'], axis=1))[0])
                if np.all(og_edges[2] == be[0:2]) or np.all(np.flip(og_edges[2]) == be[0:2]) and be[2] == flagged_cell:
                    boundary_edge = be
                    delete_be = 2
                    delete_index_be.append(np.where(np.all(be == mesh['BE'], axis=1))[0])
            # Remove the boundary edge from our local edge group
            og_edges = np.delete(og_edges, delete_be, axis=0)

            cell_i_index = 0
            cell_j_index = 0
            # TODO: FIND A WAY TO DO THIS W/ INDEXING INSTEAD OF A LOOP
            for ie in mesh['IE']:
                if np.all(og_edges[0] == ie[0:2]) or np.all(np.flip(og_edges[0]) == ie[0:2]) and ie[2] == flagged_cell:
                    cell_i_index = ie[3]
                    delete_index_ie.append(np.where(np.all(ie == mesh['IE'], axis=1))[0])
                if np.all(og_edges[1] == ie[0:2]) or np.all(np.flip(og_edges[1]) == ie[0:2]) and ie[2] == flagged_cell:
                    cell_j_index = ie[3]
                    delete_index_ie.append(np.where(np.all(ie == mesh['IE'], axis=1))[0])

            # Same ordering as the above - double [0]'s are for getting the int value from the array
            new_e1_index = \
                np.where(np.all([og_nodes_index[0], old_v_len, old_v_len + 2] == new_elements[:, :], axis=1))[0][0]
            new_e2_index = \
                np.where(np.all([old_v_len, og_nodes_index[1], old_v_len + 1] == new_elements[:, :], axis=1))[0][0]
            new_e3_index = \
                np.where(np.all([old_v_len + 1, og_nodes_index[2], old_v_len + 2] == new_elements[:, :], axis=1))[0][0]
            new_e4_index = np.where(np.all([old_v_len, old_v_len + 1, old_v_len + 2] == new_elements[:, :], axis=1))[0][
                0]

            # First 4 are splitting of internal edges and boundary edges
            new_edge_1 = np.array([og_nodes_index[0], old_v_len, new_e1_index, cell_i_index])  # IE
            new_edge_2 = np.array([old_v_len, og_nodes_index[1], new_e1_index, cell_i_index])  # IE
            new_edge_3 = np.array([og_nodes_index[1], old_v_len + 1, new_e2_index, cell_j_index])  # IE
            new_edge_4 = np.array([old_v_len + 1, og_nodes_index[2], new_e2_index, cell_j_index])  # IE

            new_edge_5 = np.array([og_nodes_index[0], old_v_len + 2, new_e1_index, be[3]])  # BE
            new_edge_6 = np.array([old_v_len + 2, og_nodes_index[2], new_e3_index, be[3]])  # BE

            # Next 3 are the internal edges for the internally created fourth element
            new_edge_7 = np.array([old_v_len, old_v_len + 1, new_e2_index, new_e4_index])
            new_edge_8 = np.array([old_v_len + 1, old_v_len + 2, new_e3_index, new_e4_index])
            new_edge_9 = np.array([old_v_len + 2, old_v_len, new_e1_index, new_e4_index])

            # Concatenate the new edges into the new internal edge array
            if not new_edge_1 in new_ie: new_ie = np.vstack((new_ie, new_edge_1))  # IE
            if not new_edge_2 in new_ie: new_ie = np.vstack((new_ie, new_edge_2))  # IE
            if not new_edge_3 in new_ie: new_ie = np.vstack((new_ie, new_edge_3))  # IE
            if not new_edge_4 in new_ie: new_ie = np.vstack((new_ie, new_edge_4))  # IE

            if not new_edge_5 in new_be: new_be = np.vstack((new_be, new_edge_5))  # BE
            if not new_edge_6 in new_be: new_be = np.vstack((new_be, new_edge_6))  # BE

            if not new_edge_7 in new_ie: new_ie = np.vstack((new_ie, new_edge_7))  # IE
            if not new_edge_8 in new_ie: new_ie = np.vstack((new_ie, new_edge_8))  # IE
            if not new_edge_9 in new_ie: new_ie = np.vstack((new_ie, new_edge_9))  # IE

        else:
            ## Internal cells
            # Node indicies for original nodes
            og_nodes_index = mesh['E'][flagged_cell]

            # Finding which "other" cell is needed using some indexing, i-j-k refer to picture
            og_edges = np.array([[og_nodes_index[0], og_nodes_index[1]],
                                 [og_nodes_index[1], og_nodes_index[2]],
                                 [og_nodes_index[2], og_nodes_index[0]]])

            # TODO: FIND A WAY TO DO THIS W/ INDEXING INSTEAD OF A LOOP
            for ie in mesh['IE']:
                if np.all(og_edges[0] == ie[0:2]) or np.all(np.flip(og_edges[0]) == ie[0:2]) and ie[2] == flagged_cell:
                    cell_i_index = ie[3]
                    delete_index_ie.append(np.where(np.all(ie == mesh['IE'], axis=1))[0])
                if np.all(og_edges[1] == ie[0:2]) or np.all(np.flip(og_edges[1]) == ie[0:2]) and ie[2] == flagged_cell:
                    cell_j_index = ie[3]
                    delete_index_ie.append(np.where(np.all(ie == mesh['IE'], axis=1))[0])
                if np.all(og_edges[2] == ie[0:2]) or np.all(np.flip(og_edges[2]) == ie[0:2]) and ie[2] == flagged_cell:
                    cell_k_index = ie[3]
                    delete_index_ie.append(np.where(np.all(ie == mesh['IE'], axis=1))[0])

            # Same ordering as the above - double [0]'s are for getting the int value from the array
            new_e1_index = \
                np.where(np.all([og_nodes_index[0], old_v_len, old_v_len + 2] == new_elements[:, :], axis=1))[0][0]
            new_e2_index = \
                np.where(np.all([old_v_len, og_nodes_index[1], old_v_len + 1] == new_elements[:, :], axis=1))[0][0]
            new_e3_index = \
                np.where(np.all([old_v_len + 1, og_nodes_index[2], old_v_len + 2] == new_elements[:, :], axis=1))[0][0]
            new_e4_index = np.where(np.all([old_v_len, old_v_len + 1, old_v_len + 2] == new_elements[:, :], axis=1))[0][
                0]

            # First 6 are splitting of outer edges
            new_edge_1 = np.array([og_nodes_index[0], old_v_len, new_e1_index, cell_i_index])
            new_edge_2 = np.array([old_v_len, og_nodes_index[1], new_e2_index, cell_i_index])
            new_edge_3 = np.array([og_nodes_index[1], old_v_len + 1, new_e2_index, cell_j_index])
            new_edge_4 = np.array([old_v_len + 1, og_nodes_index[2], new_e3_index, cell_j_index])
            new_edge_5 = np.array([og_nodes_index[2], old_v_len + 2, new_e3_index, cell_k_index])
            new_edge_6 = np.array([old_v_len + 2, og_nodes_index[0], new_e1_index, cell_k_index])

            # Next 3 are the internal edges for the internally created fourth element
            new_edge_7 = np.array([old_v_len, old_v_len + 1, new_e2_index, new_e4_index])
            new_edge_8 = np.array([old_v_len + 1, old_v_len + 2, new_e3_index, new_e4_index])
            new_edge_9 = np.array([old_v_len + 2, old_v_len, new_e1_index, new_e4_index])

            # Concatenate the new edges into the new internal edge array
            if not new_edge_1 in new_ie: new_ie = np.vstack((new_ie, new_edge_1))
            if not new_edge_2 in new_ie: new_ie = np.vstack((new_ie, new_edge_2))
            if not new_edge_3 in new_ie: new_ie = np.vstack((new_ie, new_edge_3))
            if not new_edge_4 in new_ie: new_ie = np.vstack((new_ie, new_edge_4))
            if not new_edge_5 in new_ie: new_ie = np.vstack((new_ie, new_edge_5))
            if not new_edge_6 in new_ie: new_ie = np.vstack((new_ie, new_edge_6))
            if not new_edge_7 in new_ie: new_ie = np.vstack((new_ie, new_edge_7))
            if not new_edge_8 in new_ie: new_ie = np.vstack((new_ie, new_edge_8))
            if not new_edge_9 in new_ie: new_ie = np.vstack((new_ie, new_edge_9))

    # Delete original elements/edges that no longer exist
    # new_elements = np.delete(new_elements, delete_index_cells, axis=0)
    # state = np.delete(state, delete_index_cells, axis=0)

    new_ie_2 = []
    for index in delete_index_ie:
        new_ie_2.append(int(index))
    new_be_2 = []
    for index in delete_index_be:
        new_be_2.append(int(index))

    new_ie = np.delete(new_ie, new_ie_2, axis=0)
    new_be = np.delete(new_be, new_be_2, axis=0)

    new_mesh = {'V': mesh['V'], 'E': new_elements, 'BE': new_be, 'IE': new_ie, 'Bname': mesh['Bname']}

    return new_mesh, state


def cell_split(unique_cells, edge_info, mesh, state, config):
    """Splits a cell based on the number of flagged edges associated with it - splitting process maintains conformity

    :param unique_cells: array of all unique flagged cells
    :param edge_info: Nx6 List [index, edge, "be/ie" flag]
    :param mesh: Current working mesh
    :param state: Current state vector
    :param config: Simulation runtime config
    :return: Returns arrays w/ the following information in numpy array format:
            new boundary edges, new internal edges, new cells, new state vector, indices of boundary edges to delete,
            indices of internal edges to delete, indices of cells to delete, indices of state vectors to delete
    """
    new_ie = []
    new_be = []
    new_cells = []
    new_states = []

    delete_ie = []
    delete_be = []
    delete_cells = []
    delete_states = []

    current_num_ele = mesh['E'].shape[0]
    # For each flagged cell - get our flagged edges
    for cell in unique_cells:
        cell_index = cell
        edges_flagged = np.empty((4,), dtype=int)
        for edge in edge_info:
            if edge[1][2] == cell_index or (edge[1][3] == cell_index and edge[2] != 'be'):
                edges_flagged = np.vstack((edges_flagged, edge[1]))
        edges_flagged = np.delete(edges_flagged, 0, axis=0)

        # With the set of flagged edges do the splitting processes
        if edges_flagged.shape[0] == 1:
            # 1 flagged edge -> 1 edge introduced so cell is split in 2 from a bisector on the third node

            # Index number of our new node that splits the cell
            midpoint = (mesh['V'][edges_flagged[0]] + mesh['V'][1]) / 2
            midpoint = np.where(np.equal(midpoint, mesh['V']).all(1))[0][0]

            # The node that forms the bisector with the midpoint node
            cell_nodes = list(mesh['E'][cell_index])
            bisect_node = cell_nodes.remove([edges_flagged[0], edges_flagged[1]])

            # 2 new cells from splitting the single cell
            new_cells.append([edges_flagged[0], bisect_node, midpoint])
            new_cells.append([edges_flagged[1], bisect_node, midpoint])
            delete_cells.append(cell_index)
            current_num_ele += 2

            # 3 new edges for the cell - check if the bisected edge is a BE, the entirely new edge must be IE
            new_ie.append([midpoint, bisect_node, current_num_ele - 2, current_num_ele - 1])

            for edge in edge_info:
                if np.equal(edge, edges_flagged).all(1) and edge[2] == 'be':
                    new_be.append([edges_flagged[0], midpoint, current_num_ele - 2, edges_flagged[3]])
                    new_be.append([midpoint, edges_flagged[1], current_num_ele - 1, edges_flagged[3]])
                    delete_be.append(edges_flagged)

                if np.equal(edge, edges_flagged).all(1) and edge[2] == 'ie':
                    other_cell = int(list(edges_flagged).remove(edges_flagged[0], edges_flagged[1], cell_index))
                    new_be.append([edges_flagged[0], midpoint, current_num_ele - 2, other_cell])
                    new_be.append([midpoint, edges_flagged[1], current_num_ele - 1, other_cell])
                    delete_ie.append(edges_flagged)
            # Copy the current cell into the new cell
            new_states.append([state[cell_index], state[cell_index]])
            delete_states.append(cell_index)

        if edges_flagged.shape[0] == 2:
            # TODO: FIND A NEW SOLUTION FOR 2+ SPLITS - CAN'T GET THIS AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            # 2 flagged edges -> 6 edges introduced so cell is split in 3

            # Index number of our new nodes that splits the cell
            midpoint_1 = (mesh['V'][edges_flagged[0, 0]] + mesh['V'][0, 1]) / 2
            midpoint_1 = np.where(np.equal(midpoint_1, mesh['V']).all(1))[0][0]

            midpoint_2 = (mesh['V'][edges_flagged[1, 0]] + mesh['V'][1, 1]) / 2
            midpoint_2 = np.where(np.equal(midpoint_2, mesh['V']).all(1))[0][0]

            # The node that forms the bisector with the midpoint node
            cell_nodes = list(mesh['E'][cell_index])
            if edges_flagged[0, 0] == edges_flagged[1, 0]:
                common_node = edges_flagged[0, 0]
            elif edges_flagged[0, 0] == edges_flagged[1, 1]:
                common_node = edges_flagged[0, 0]
            else:
                common_node = edges_flagged[0, 1]

            # 3 new cells from splitting the single cell
            new_cells.append([midpoint_1, midpoint_2, common_node])
            # Splitting across depends on the larger angle - check angle w/ law of cosines
            not_shared = cell_nodes.remove(common_node)

            angle1 = angle_loc(not_shared[0], common_node, not_shared[1])
            angle2 = angle_loc(not_shared[1], common_node, not_shared[0])

            current_num_ele += 3

            small_triangle = current_num_ele - 3
            small_triangle_ele = [midpoint_1, midpoint_2, common_node]
            split_triangle = current_num_ele - 2

            base_triangle = current_num_ele - 1

            if angle1 > angle2:  # We cut angle 1 so not_shared[0] is cut in two for the splitting purpose
                new_cells.append([not_shared[0], not_shared[1], midpoint_1])
                new_cells.append([not_shared[0], midpoint_2, midpoint_1])

                split_triangle_ele = [not_shared[0], midpoint_2, midpoint_1]
                base_triangle_ele = [not_shared[0], not_shared[1], midpoint_1]

                new_ie.append([not_shared[0], midpoint_2, split_triangle, base_triangle])
            else:
                new_cells.append([not_shared[0], not_shared[1], midpoint_2])
                new_cells.append([not_shared[1], midpoint_2, midpoint_1])

                split_triangle_ele = [not_shared[1], midpoint_2, midpoint_1]
                base_triangle_ele = [not_shared[0], not_shared[1], midpoint_2]

                new_ie.append([not_shared[1], midpoint_2, split_triangle, base_triangle])

            delete_cells.append(cell_index)

            for flag, edge in itertools.product(edges_flagged, edge_info):
                if np.equal(flag, edge[1]).all(1) and edge[2] == 'be':
                    # Find the edge and which new cells it makes up - then append and delete all BE
                    be_midpoint = (mesh['V'][edge[1][0]] + mesh['V'][edge[1][1]]) / 2
                    be_midpoint = np.where(np.equal(be_midpoint, mesh['V']).all(1))[0][0]

                    # Logic to figure out which index is new boundary cell number

                if np.equal(flag, edge[2]).all(1) and edge[2] == 'ie':
                    # Find the IE that matches and go from new cell -> other cell
                    pass
            # Add new unique internal edges
            new_ie.append([midpoint_1, midpoint_2, small_triangle, split_triangle])

        if edges_flagged.shape[0] == 3:
            pass

    return None, None, None, None, None, None, None, None


def refine_interp_write(mesh, state, fname, config):
    """Refines the mesh based upon jumps in Mach number across cell boundaries. Also interpolates the solution to the
    new mesh by copying the value from one cell to the next.

    :param mesh: Dictionary of the mesh in the KFID-GRI format
    :param state: Nx4 array of state vectors used to calculate Mach numbers
    :param fname: Filename for the refined mesh's gri file
    :param config: Configuration class for the simulation which has working fluid information
    :return: Returns the refined mesh and state and writes the new mesh to a fname.gri file
    """
    flagged_cells, flagged_be, flagged_ie = find_mach_jumps(state, mesh, config)

    new_nodes = np.empty((2,))
    # Generate the array of all needed nodes for all cell refinement operations
    for edge in flagged_be:
        midpoint = (mesh['V'][edge[0]] + mesh['V'][edge[1]]) / 2
        new_nodes = np.vstack((new_nodes, midpoint))
    for edge in flagged_ie:
        midpoint = (mesh['V'][edge[0]] + mesh['V'][edge[1]]) / 2
        new_nodes = np.vstack((new_nodes, midpoint))
    new_nodes = np.delete(new_nodes, 0, axis=0)
    mesh['V'] = np.vstack((mesh['V'], new_nodes))

    split_data = []
    # Generate the list structure used with cell_split function
    for edge in flagged_be:
        if edge[2] in flagged_cells:
            split_data.append([edge[2], edge, 'be'])
    for edge in flagged_ie:
        if edge[2] in flagged_cells:
            split_data.append([edge[2], edge, 'ie'])
        elif edge[3] in flagged_cells:
            split_data.append([edge[3], edge, 'ie'])

    # Get the information for the new cells/edges and the old cells/edges to delete
    new_be, new_ie, new_cells, new_state, be_to_delete, ie_to_delete, cells_to_delete, state_delete = \
        cell_split(flagged_cells, split_data, mesh, state, config)

    return


def refine_interp_write_node_placement(mesh, state, fname, config):
    """Refines the mesh using node-insertion, interpolates the state to the new mesh, and saves the mesh under
    the fname.gri filename and .gri format

    :param mesh: Dictionary of the mesh in the KFID-GRI format
    :param state: Nx4 array of state vectors used to calculate Mach numbers
    :param fname: Filename for the refined mesh's gri file
    :param config: Configuration class for the simulation which has working fluid information
    :return: Returns the refined mesh and state and writes the new mesh to a fname.gri file
    """
    # Nx2 array [error, cell i]
    error_index_ie = np.zeros((len(mesh['IE']), 2))
    error_index_be = np.zeros((len(mesh['BE']), 2))

    for i in range(len(mesh['BE'])):
        # Boundary Edges
        be = mesh['BE'][i]
        # No error on the inflow cells
        if mesh['Bname'][be[3]] == 'Inflow' or mesh['Bname'][be[3]] == 'Outflow':
            continue
        else:
            be_l, be_n = edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

            # Cell i quantities
            u = state[be[2]][1] / state[be[2]][0]
            v = state[be[2]][2] / state[be[2]][0]
            q = np.dot(be_n, np.array([u, v]))
            flux_c = flux.F_euler_2d(state[be[2]], config)
            h_l = (flux_c[3, 0] + flux_c[3, 1]) / (flux_c[0, 0] + flux_c[0, 1])
            c = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
            # c = helper.mach_calc_single(state[be[2]], config)

            error = abs(q / c) * be_l

            error_index_be[i, 0] = error
            error_index_be[i, 1] = be[2]

    for i in range(len(mesh['IE'])):
        # Internal Edges
        ie = mesh['IE'][i]
        ie_l, ie_n = edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])

        # Left cell/cell i quantities
        u_l = state[ie[2]]
        u_l_u = u_l[1] / u_l[0]
        u_l_v = u_l[2] / u_l[0]
        q_l = np.linalg.norm([u_l_u, u_l_v])
        flux_l = flux.F_euler_2d(u_l, config)
        h_l = (flux_l[3, 0] + flux_l[3, 1]) / (flux_l[0, 0] + flux_l[0, 1])
        c_l = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
        # This formulation finds speed of sound as function of H = E + p/rho
        # c_l = helper.mach_calc_single(u_l, config)
        m_l = q_l / c_l

        # Right cell/cell N quantities
        u_r = state[ie[3]]
        u_r_u = u_r[1] / u_r[0]
        u_r_v = u_r[2] / u_r[0]
        q_r = np.linalg.norm([u_r_u, u_r_v])
        flux_r = flux.F_euler_2d(u_r, config)
        h_r = (flux_r[3, 0] + flux_r[3, 1]) / (flux_r[0, 0] + flux_r[0, 1])
        c_r = math.sqrt((config.y - 1) * (h_r - q ** 2 / 2))

        # c_r = helper.mach_calc_single(u_r, config)

        m_r = q_r / c_r

        error = abs(m_l - m_r) * ie_l

        error_index_ie[i, 0] = error
        error_index_ie[i, 1] = ie[2]

    # Total list of all locations where Mach number jumps are too high
    total_error_index = np.vstack((error_index_be, error_index_ie))
    error_index = np.flip(total_error_index[total_error_index[:, 0].argsort()], axis=0)
    error_index = error_index[0:math.floor(config.adaptation_percentage * len(error_index[:, 0])), 1]

    # List of all unique cell locations where we want to refine and divide the triangle
    flagged_cells = []
    for cell in error_index:
        if not (cell in flagged_cells):
            flagged_cells.append(int(cell))

    # Loop over the flagged cells and update the mesh with new information regarding new cells/internal edges/nodes
    for cell_index in flagged_cells:
        # Add the new cell centroid to the mesh
        cell_coords = mesh['V'][mesh['E'][cell_index]]
        new_centroid = np.sum(cell_coords, axis=0) / 3
        mesh['V'] = np.vstack((mesh['V'], new_centroid))

        centroid_index = np.where(np.equal(new_centroid, mesh['V']).all(1))[0][0]

        num_cells = mesh['E'].shape[0]

        # Append the cell array to include the two additional cells
        mesh['E'] = np.vstack((mesh['E'],
                               reorient_ccw(mesh['E'][cell_index, 0], mesh['E'][cell_index, 1], centroid_index,
                                            mesh['V']),
                               reorient_ccw(mesh['E'][cell_index, 1], mesh['E'][cell_index, 2], centroid_index,
                                            mesh['V'])))

        # Find the internal edges and update the left/right cells to account for new edges
        possible_edges = mesh['IE'][
            np.where(np.logical_or(np.equal(cell_index, mesh['IE'][:, 2]), np.equal(cell_index, mesh['IE'][:, 3])))]

        if np.any(np.equal(mesh['E'][cell_index, 0:2], possible_edges[:, 0:2]).all(1)):
            ie_fix_1_i = np.where(np.equal(mesh['E'][cell_index, 0:2], mesh['IE'][:, 0:2]).all(1))[0][0]
            ie_fix_1 = mesh['IE'][ie_fix_1_i]
        if np.any(np.equal(mesh['E'][cell_index, :0:-1], possible_edges[:, 0:2]).all(1)):
            ie_fix_1_i = np.where(np.equal(mesh['E'][cell_index, :0:-1], mesh['IE'][:, 0:2]).all(1))[0][0]
            ie_fix_1 = mesh['IE'][ie_fix_1_i]

        if np.any(np.equal(mesh['E'][cell_index, 1::], possible_edges[:, 0:2]).all(1)):
            ie_fix_2_i = mesh['IE'][np.where(np.equal(mesh['E'][cell_index, 1::], mesh['IE'][:, 0:2]).all(1))][0][0]
            ie_fix_2 = mesh['IE'][ie_fix_2_i]
        if np.any(np.equal(mesh['E'][cell_index, :0:-1], possible_edges[:, 0:2]).all(1)):
            ie_fix_2_i = mesh['IE'][np.where(np.equal(mesh['E'][cell_index, :0:-1], mesh['IE'][:, 0:2]).all(1))][0][0]
            ie_fix_2 = mesh['IE'][ie_fix_2_i]

        if ie_fix_1[2] == cell_index:
            mesh['IE'][ie_fix_1_i, 2] = num_cells + 1
        else:
            mesh['IE'][ie_fix_1_i, 2] = num_cells + 1

        if ie_fix_2[2] == cell_index:
            mesh['IE'][ie_fix_2_i, 2] = num_cells + 2
        else:
            mesh['IE'][ie_fix_2_i, 2] = num_cells + 2

        # Append the new internal edges
        new_ies = np.array([[mesh['E'][cell_index, 0], centroid_index, cell_index, num_cells + 1],
                            [mesh['E'][cell_index, 1], centroid_index, num_cells + 1, num_cells + 2],
                            [mesh['E'][cell_index, 2], centroid_index, num_cells + 2, cell_index]])
        mesh['IE'] = np.vstack((mesh['IE'], new_ies))

        # Replace the index of the flagged cell with one of the newly created cells
        mesh['E'][cell_index] = reorient_ccw(mesh['E'][cell_index, 0], mesh['E'][cell_index, 2], centroid_index,
                                             mesh['V'])

    return mesh, state
