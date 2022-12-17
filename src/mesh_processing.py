import itertools

import numpy as np
import math
import copy
import flux
import readgri


def centroid(mesh):
    """Returns an array of centroids for all cells in the mesh.

    :param mesh: Mesh in dictionary format
    :return: centroids of all cells in mesh
    """
    centroids = np.sum(mesh['V'][mesh['E']], axis=1) / 3

    return centroids


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
    # TODO: Vectorize with Numpy and have it do all cells at once
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



# TODO: Break the AMR into its own file
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

# TODO: Remake another form with standard flagged-edge splitting 1 half, 2 largest angle, 3 uniform
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

# TODO: Vectorize out loops if possible using logical indexing - speed up computation time
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