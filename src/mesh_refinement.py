import cell_geometry_formulas as cgf
import numpy as np
import math
import readgri
import flux
from numba import njit
import helper


@njit(cache=True)
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
    if ((node2_coord[1] - node1_coord[1]) * (node3_coord[0] - node2_coord[0]) - \
        (node3_coord[1] - node2_coord[1]) * (node2_coord[0] - node1_coord[0])) < 0:
        return np.array([node1, node2, node3])
    else:
        return np.array([node3, node2, node1])


def refine_interp_uniform(mesh, state, y, f, fname):
    """Performs adaptive mesh refinement on the given mesh from the results of the state vector and large jumps in Mach
    number across cell boundaries. Divides flagged cells uniformly, cells w/ >= 2 neighbors flagged uniformly, and cells
    with only 1 edge flagged.

    :param mesh: Mesh of the domain
    :param state: Nx4 state vector
    :param y: Ratio of specific heats - gamma
    :param f: Fraction of cells to be refined
    :param fname: Filename that the newly saved mesh is saved as
    :return:
    """
    flag_cell, split_cell, flag_be, flag_ie, split_be, split_ie = find_uniform_splitting(state, mesh, y, f)

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
            if not np.any(np.equal(midpoint_split, mesh['V']).all(1)): mesh['V'] = np.vstack(
                (mesh['V'], midpoint_split))
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
        f.write('%.15e %.15e\n' % (node[0], node[1]))

    # Boundary edge information
    f.write('{0}\n'.format(len(mesh['Bname'])))

    boundary_pairs = [[] for name in mesh['Bname']]

    # Boundary pairs writing
    for be in mesh['BE']:
        if np.any(np.equal(be, flag_be).all(1)) or np.any(np.equal(be, flag_ie).all(1)):
            midpoint = (mesh['V'][be[0]] + mesh['V'][be[1]]) / 2
            midpoint_index = np.where(np.equal(midpoint, mesh['V']).all(1))[0][0]
            if not [be[0], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append(
                [be[0], midpoint_index])
            if not [be[1], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append(
                [be[1], midpoint_index])
            continue
        boundary_pairs[be[3]].append([be[0], be[1]])
    for i in range(len(mesh['Bname'])):
        f.write('{0} 2 {1}\n'.format(len(boundary_pairs[i]), mesh['Bname'][i]))
        for pair in boundary_pairs[i]:
            f.write('{0} {1}\n'.format(pair[0] + 1, pair[1] + 1))

    # Element writing
    f.write('{0} 1 TriLagrange\n'.format(new_elements_full.shape[0]))
    for element in new_elements_full:
        temp_element = reorient_ccw(element[0], element[1], element[2], mesh['V'])
        f.write('{0} {1} {2}\n'.format(temp_element[0] + 1, temp_element[1] + 1, temp_element[2] + 1))
    f.close()

    # Generate the new mesh using the hashing provided by the reading of the gri file
    mesh = readgri.readgri(fname)

    return mesh, state


def find_uniform_splitting(state, mesh, y, f):
    """Creates a unique array of cell indices that are to be split uniformly and split to maintain conformity. Any cell
    that has 2 uniform neighbors becomes a uniform cell, this process continues until either all cells are uniform or
    a cell no longer has 2 uniform neighbors and can be split in two.

    :param state: Nx4 numpy array of state vectors
    :param mesh: Current working mesh
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
            be_l, be_n = cgf.edge_properties_calculator(mesh['V'][be[0]], mesh['V'][be[1]])

            # Cell i quantities
            u = state[be[2]][1] / state[be[2]][0]
            v = state[be[2]][2] / state[be[2]][0]
            q = np.dot(be_n, np.array([u, v]))
            flux_c = flux.F_euler_2d(state[be[2]], y)
            h_l = (flux_c[3, 0] + flux_c[3, 1]) / (flux_c[0, 0] + flux_c[0, 1])
            c = math.sqrt((y - 1) * (h_l - q ** 2 / 2))

            error = abs(q / c) * be_l

            error_index_be[i, 0] = error
            error_index_be[i, 1] = be[2]

    for i in range(len(mesh['IE'])):
        # Internal Edges
        ie = mesh['IE'][i]
        ie_l, ie_n = cgf.edge_properties_calculator(mesh['V'][ie[0]], mesh['V'][ie[1]])

        # Left cell/cell i quantities
        u_l = state[ie[2]]
        u_l_u = u_l[1] / u_l[0]
        u_l_v = u_l[2] / u_l[0]
        q_l = np.linalg.norm([u_l_u, u_l_v])
        flux_l = flux.F_euler_2d(u_l, y)
        h_l = (flux_l[3, 0] + flux_l[3, 1]) / (flux_l[0, 0] + flux_l[0, 1])
        c_l = math.sqrt((y - 1) * (h_l - q_l ** 2 / 2))
        m_l = q_l / c_l

        # Right cell/cell N quantities
        u_r = state[ie[3]]
        u_r_u = u_r[1] / u_r[0]
        u_r_v = u_r[2] / u_r[0]
        q_r = np.linalg.norm([u_r_u, u_r_v])
        flux_r = flux.F_euler_2d(u_r, y)
        h_r = (flux_r[3, 0] + flux_r[3, 1]) / (flux_r[0, 0] + flux_r[0, 1])
        c_r = math.sqrt((y - 1) * (h_r - q_r ** 2 / 2))

        m_r = q_r / c_r

        error = abs(m_l - m_r) * ie_l

        error_index_ie[i, 0] = error
        error_index_ie[i, 1] = ie[2]

    # Total list of all locations where Mach number jumps are too high
    total_error_index = np.vstack((error_index_be, error_index_ie))
    error_index = np.flip(total_error_index[total_error_index[:, 0].argsort()], axis=0)
    error_index = error_index[0:math.floor(f * len(error_index[:, 0])), 1]

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


@njit(cache=True)
def find_flagged_edges(state, E, V, IE, BE, f, y):
    """Finds the flagged edges and flags cells and then generates a Cell-to-Edge matrix of what edges of each cell need
    to be refined to fit the 1-2-3 flagged refinement pattern.

    :param state: State vector array
    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param IE: Internal edge array [nodeA, nodeB, cell left, cell right]
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param f: Number of edges to flag for refinement
    :param y: Ratio of specific heats - gamma
    """
    ele_to_flag_edges = np.zeros((E.shape[0], 3))

    # Mach number in every cell
    mach = helper.calculate_mach(state, y)

    # There is one error per boundary edge and internal edge
    error = np.zeros((IE.shape[0] + BE.shape[0], 3))

    for i in range(IE.shape[0]):
        l, _ = cgf.edge_properties_calculator(V[IE[i, 0]], V[IE[i, 1]])
        error[i, :] = np.array((np.abs(mach[IE[i, 2]] - mach[IE[i, 3]]) * l, IE[i, 2], IE[i, 3]))

    for i in range(BE.shape[0]):
        l, n = cgf.edge_properties_calculator(V[BE[i, 0]], V[BE[i, 1]])

        u = state[BE[i, 2], 1] / state[BE[i, 2], 0]
        v = state[BE[i, 2], 2] / state[BE[i, 2], 0]
        q = np.dot(n, np.array((u, v)))
        error[IE.shape[0] + i, :] = np.array((np.abs(q / mach[BE[i, 2]]) * l, BE[i, 2], -1))

    error = error[(-error[:, 0]).argsort()]
    error = error[0:math.ceil(error.shape[0] * f), :]

    # Generate a unique list of cell indices that are flagged for refinement
    flagged_cells = np.unique(error[:, 1::].flatten())

    # Generate a unique listing of node pairs that are used to track the flagged edges
    node_pairs = np.empty((0, 2))
    for cell in flagged_cells:
        if cell == -1: continue
        pair1 = np.array((E[int(cell), 0], E[int(cell), 1]))
        pair2 = np.array((E[int(cell), 1], E[int(cell), 2]))
        pair3 = np.array((E[int(cell), 2], E[int(cell), 0]))
        temp_pair = np.vstack((pair1, pair2, pair3))
        node_pairs = np.vstack((node_pairs, np.reshape(temp_pair, (3, 2))))

    for i in range(E.shape[0]):
        # If the index is one of the flagged cells we can skip checking for edges since we know all edges are flagged
        if (np.abs(flagged_cells - i)).min() == 0:
            ele_to_flag_edges[i] = np.array((1, 2, 3))
            continue

        # 6 Possible combinations of node pairs per cell that might be in the node_pairs array
        pair_1a = np.array((E[i, 0], E[i, 1]), dtype=np.int32)
        pair_2a = np.array((E[i, 1], E[i, 2]), dtype=np.int32)
        pair_3a = np.array((E[i, 2], E[i, 0]), dtype=np.int32)

        pair_1b = np.flip(pair_1a)
        pair_2b = np.flip(pair_2a)
        pair_3b = np.flip(pair_3a)

        # Check the pair 1's to see if this edge is flagged
        if (np.sum(np.abs(node_pairs - pair_1a), axis=1)).min() == 0 or \
                (np.sum(np.abs(node_pairs - pair_1b), axis=1)).min() == 0:
            ele_to_flag_edges[i, 0] = 1

        # Check the pair 2's to see if this edge is flagged
        if (np.sum(np.abs(node_pairs - pair_2a), axis=1)).min() == 0 or \
                (np.sum(np.abs(node_pairs - pair_2b), axis=1)).min() == 0:
            ele_to_flag_edges[i, 1] = 2

        # Check the pair 3's to see if this edge is flagged
        if (np.sum(np.abs(node_pairs - pair_3a), axis=1)).min() == 0 or \
                (np.sum(np.abs(node_pairs - pair_3b), axis=1)).min() == 0:
            ele_to_flag_edges[i, 2] = 3

    return ele_to_flag_edges


@njit(cache=True)
def split_cell(state, E, V, E2e):
    """Splits the cells (E) using the E2e matrix which says which edge needs to be refined. Refinement is done according
    to the 1-2-3 flagging (1 flag -> 2 cells, 2 flag -> 3 cells, 3 flag -> 4 cells).

    :param state: Nx4 state vector array
    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param IE: Internal edge array [nodeA, nodeB, cell left, cell right]
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param E2e: Element-2-Edges for which edges need to be refined
    """
    new_E = np.empty(shape=(0, 3), dtype=np.int32)
    new_state = np.empty(shape=(0, 4))

    for i in range(E2e.shape[0]):
        # Checking how many new elements based on the hashing matrix
        new_ele_check = 0
        for j in range(E2e[i].shape[0]):
            if E2e[i, j] == 0: continue
            else: new_ele_check += 1

        # If no new cells are to be added, then add the current cell to the new cells
        if new_ele_check == 0:
            new_E_temp = np.reshape(np.array((E[i, 0], E[i, 1], E[i, 2]), dtype=np.int32), (new_ele_check+1, 3))

        # If one new cell is to be added then we have cell splitting along the flagged edge
        if new_ele_check == 1:
            # Find which edge is flagged
            for j in range(E2e[i].shape[0]):
                if E2e[i, j] != 0:
                    # Perform the cell splitting
                    midpoint = (V[E[i][j]] + V[E[i][j - 2]]) / 2

                    V = np.vstack((V, np.reshape(midpoint, (1, 2))))

                    midpointi = np.argmin(np.sum(np.abs(V - midpoint), axis=1))

                    new_E_temp1 = np.array((midpointi, E[i][j], E[i][j - 1]), dtype=np.int32)
                    new_E_temp2 = np.array((midpointi, E[i][j - 1], E[i][j - 2]), dtype=np.int32)
                    new_E_temp = np.vstack((new_E_temp1, new_E_temp2))

        # If two new cells are to be added then split the cells along the two flagged edges and then largest angle
        if new_ele_check == 2:
            # Find which edge isn't flagged
            for j in range(E2e[i].shape[0]):
                if E2e[i, j] == 0:
                    midpoint1 = (V[E[i][j - 2]] + V[E[i][j - 1]]) / 2
                    midpoint2 = (V[E[i][j - 1]] + V[E[i][j]]) / 2
                    temp_midpoints = np.vstack((midpoint1, midpoint2))
                    V = np.vstack((V, np.reshape(temp_midpoints, (2, 2))))

                    midpoint1i = np.argmin(np.sum(np.abs(V - midpoint1), axis=1))
                    midpoint2i = np.argmin(np.sum(np.abs(V - midpoint2), axis=1))

                    # Check to see which angle is bigger - that is the one that gets split
                    l_not_flagged, _ = cgf.edge_properties_calculator(V[E[i][j]], V[E[i][j - 2]])
                    l_f1, _ = cgf.edge_properties_calculator(V[E[i][j - 2]], V[E[i][j - 1]])
                    l_f2, _ = cgf.edge_properties_calculator(V[E[i][j - 1]], V[E[i][j]])

                    vec_not_flag = V[E[i][j]] - V[E[i][j - 2]]
                    vec_f1 = V[E[i][j - 1]] - V[E[i][j - 2]]
                    vec_f2 = V[E[i][j - 1]] - V[E[i][j]]

                    # Angles between the not flagged and the flagged edge
                    angle_f1 = np.arccos(np.dot(vec_not_flag, vec_f1) / (l_not_flagged * l_f1))
                    angle_f2 = np.arccos(np.dot(vec_not_flag, vec_f2) / (l_not_flagged * l_f2))

                    if angle_f1 > angle_f2:
                        new_E_temp1 = np.array((E[i][j - 1], midpoint1i, midpoint2i), dtype=np.int32)
                        new_E_temp2 = np.array((midpoint1i, E[i][j - 2], midpoint2i), dtype=np.int32)
                        new_E_temp3 = np.array((midpoint2i, E[i][j - 2], E[i][j]), dtype=np.int32)
                    else:
                        new_E_temp1 = np.array((E[i][j - 1], midpoint1i, midpoint2i), dtype=np.int32)
                        new_E_temp2 = np.array((midpoint1i, E[i][j], midpoint2i), dtype=np.int32)
                        new_E_temp3 = np.array((midpoint1i, E[i][j - 2], E[i][j]), dtype=np.int32)

                    new_E_temp = np.vstack((new_E_temp1, new_E_temp2, new_E_temp3))

        # If 3 new cells are to be added we need to do uniform refinement
        if new_ele_check == 3:
            midpoint1 = (V[E[i][0]] + V[E[i][1]]) / 2
            midpoint2 = (V[E[i][1]] + V[E[i][2]]) / 2
            midpoint3 = (V[E[i][2]] + V[E[i][0]]) / 2
            temp_midpoints = np.vstack((midpoint1, midpoint2, midpoint3))
            V = np.vstack((V, np.reshape(temp_midpoints, (3, 2))))

            # Get the indices in V where the new midpoint nodes are placed
            midpoint1i = np.argmin(np.sum(np.abs(V - midpoint1), axis=1))
            midpoint2i = np.argmin(np.sum(np.abs(V - midpoint2), axis=1))
            midpoint3i = np.argmin(np.sum(np.abs(V - midpoint3), axis=1))

            # Add the new cells to the new cell array
            new_E_temp1 = np.array((E[i][0], midpoint1i, midpoint3i), dtype=np.int32)
            new_E_temp2 = np.array((midpoint1i, E[i][1], midpoint2i), dtype=np.int32)
            new_E_temp3 = np.array((midpoint2i, E[i][2], midpoint3i), dtype=np.int32)
            new_E_temp4 = np.array((midpoint1i, midpoint2i, midpoint3i), dtype=np.int32)

            new_E_temp = np.vstack((new_E_temp1, new_E_temp2, new_E_temp3, new_E_temp4))

        new_E = np.vstack((new_E, new_E_temp))

        # Copy the new state vectors to the new state vector stack
        new_state_temp = np.empty((0, 4))
        for j in range(new_ele_check+1):
            new_state_temp = np.vstack((new_state_temp, np.reshape(state[i], (1, 4))))
        new_state = np.vstack((new_state, new_state_temp))

    return new_state, new_E, V


@njit(cache=True)
def split_boundaries(E, V, BE, E2e):
    """Splits the boundary edges (BE) using the E2e matrix which says which edge needs to be refined. It checks if a
    boundary edge needs to be refined by running it through E2e and if it does, it updates the boundary edge node pairs.

    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param E2e: Element-2-Edges for which edges need to be refined
    """
    new_BE = np.empty((0, 3), dtype=np.int32)
    for i in range(BE.shape[0]):
        be_split = True
        for j in range(E2e[BE[i, 2]].shape[0]):
            # Check to see if the edge is flagged
            if E2e[BE[i, 2], j] != 0:
                # Check to see if the flagged edge matches the BE
                flagged_edge = np.array((E[i][j], E[i][j - 1]), dtype=np.int32)
                flagged_edge_rev = np.array((E[i][j - 1], E[i][j]), dtype=np.int32)

                if (BE[i, 0:2] - flagged_edge).sum() == 0 or (BE[i, 0:2] - flagged_edge_rev).sum() == 0:
                    midpoint = (V[BE[i, 0]] + V[BE[i, 1]]) / 2
                    midpointi = np.argmin(np.sum(np.abs(V - midpoint), axis=1))

                    new_BE_temp1 = np.array((BE[i, 0], midpointi, BE[i, 3]), dtype=np.int32)
                    new_BE_temp2 = np.array((midpointi, BE[i, 1], BE[i, 3]), dtype=np.int32)
                    new_BE_temp = np.vstack((new_BE_temp1, new_BE_temp2))
                    new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (2, 3))))
                    be_split = False
        if be_split:
            new_BE_temp = np.array((BE[i, 0], BE[i, 1], BE[i, 3]), dtype=np.int32)
            new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (1, 3))))

    return new_BE

@njit(cache=True)
def write_new_mesh(E, V, new_BE, fname):
    """Writes the refined mesh to a *.gri file.

    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param new_BE: Nx3 matrix, [nodeA, nodeB, BC Flag]
    :param fname: Filename to write the mesh to
    """
    # Write out the mesh to fname.gri and then parse it again using the given reading/hashing functionaility
    f = open(fname, 'w')

    # gri header
    f.write('{0} {1} 2\n'.format(V.shape[0], E.shape[0]))

    # Node information
    for node in V:
        f.write('%.15e %.15e\n' % (node[0], node[1]))

    # TODO: Reconfigure this BE writing to work with new_BE
    # Boundary edge information
    # f.write('{0}\n'.format(len(mesh['Bname'])))
    #
    # boundary_pairs = [[] for name in mesh['Bname']]
    #
    # # Boundary pairs writing
    # for be in mesh['BE']:
    #     if np.any(np.equal(be, flag_be).all(1)) or np.any(np.equal(be, flag_ie).all(1)):
    #         midpoint = (mesh['V'][be[0]] + mesh['V'][be[1]]) / 2
    #         midpoint_index = np.where(np.equal(midpoint, mesh['V']).all(1))[0][0]
    #         if not [be[0], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append(
    #             [be[0], midpoint_index])
    #         if not [be[1], midpoint_index] in boundary_pairs[be[3]]: boundary_pairs[be[3]].append(
    #             [be[1], midpoint_index])
    #         continue
    #     boundary_pairs[be[3]].append([be[0], be[1]])
    # for i in range(len(mesh['Bname'])):
    #     f.write('{0} 2 {1}\n'.format(len(boundary_pairs[i]), mesh['Bname'][i]))
    #     for pair in boundary_pairs[i]:
    #         f.write('{0} {1}\n'.format(pair[0] + 1, pair[1] + 1))

    # Element writing
    f.write('{0} 1 TriLagrange\n'.format(E.shape[0]))
    for element in E:
        temp_element = reorient_ccw(element[0], element[1], element[2], V)
        f.write('{0} {1} {2}\n'.format(temp_element[0] + 1, temp_element[1] + 1, temp_element[2] + 1))
    f.close()


# @njit(cache=True)
def adapt_mesh(state, E, V, IE, BE, f, y, fname):
    """Adapts the mesh using the error between cell states which is described as jumps in Mach number.

    :param state: State vector array
    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param IE: Internal edge array [nodeA, nodeB, cell left, cell right]
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param f: Number of edges to flag for refinement
    :param y: Ratio of specific heats - gamma
    :param fname: Filename to write the mesh to
    """
    hashed_refinement_matrix = find_flagged_edges(state, E, V, IE, BE, f, y)
    new_state, new_E, new_V = split_cell(state, E, V, hashed_refinement_matrix)
    new_boundaries = split_boundaries(E, new_V, BE, hashed_refinement_matrix)
    return
    # write_new_mesh(new_E, new_V, new_boundaries, fname)

