import cell_geometry_formulas as cgf
from numba import njit
import numpy as np
import readgri
import helper
import math


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
    if ((node2_coord[1] - node1_coord[1]) * (node3_coord[0] - node2_coord[0]) -
        (node3_coord[1] - node2_coord[1]) * (node2_coord[0] - node1_coord[0])) < 0:
        return np.array([node1, node2, node3])
    else:
        return np.array([node3, node2, node1])


# TODO: Re-do a lot of this algorithm to avoid double-precision logical operations
@njit(cache=True)
def find_flagged_edges(state, E, V, IE, BE, f, y):
    """Finds the flagged edges and flags cells and then generates a Cell-to-Edge matrix of what edges of each cell need
    to be refined to fit the 1-2-3 flagged refinement pattern.

    :param state: [:, 4] Numpy array of state vectors, each row is 1 cell's state [rho, rho*u, rho*v, rho*E]
    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param IE: Internal edge array [nodeA, nodeB, cell left, cell right]
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param f: Number of edges to flag for refinement
    :param y: Ratio of specific heats - gamma
    :returns: ele_to_flag_edges: [E.shape[0], 3] Numpy array that tells which edge is flagged for refinement on that cell
    """
    ele_to_flag_edges = np.zeros((E.shape[0], 3))

    # Mach number in every cell
    mach = helper.calculate_mach(state, y)

    # Error for all the internal edges
    ieLength, _ = cgf.edgePropertiesCalculator(IE[:, 0:2], V)
    errorIE = np.vstack((np.multiply(np.abs(mach[IE[:, 2]] - mach[IE[:, 3]]), ieLength), IE[:, 2], IE[:, 3]))
    errorIE = np.transpose(errorIE)

    # Error for all the boundary edges that are not input and output edges - those are ignored, and they're always the
    # last 2 BCs in the boundary list
    numBoundariesZeroIndex = (np.unique(BE[:, 3])).shape[0] - 1
    notInflowOutflowLogical = np.logical_or(BE[:, 3] == numBoundariesZeroIndex - 1, BE[:, 3] == numBoundariesZeroIndex)
    beToError = BE[np.logical_not(notInflowOutflowLogical)]

    beLength, beNorm = cgf.edgePropertiesCalculator(beToError[:, 0:2], V)

    u = np.divide(state[beToError[:, 2], 1], state[beToError[:, 2], 0])
    v = np.divide(state[beToError[:, 2], 1], state[beToError[:, 2], 0])
    q = np.multiply(u, beNorm[:, 0]) + np.multiply(v, beNorm[:, 1])

    errorBE = np.vstack((np.multiply(np.divide(q, mach[beToError[:, 2]]), beLength),
                         beToError[:, 2], np.zeros(beToError.shape[0]) - 1))
    errorBE = np.transpose(errorBE)

    # Combine the errors and trim it down to the adaptation percentage
    error = np.vstack((errorIE, errorBE))
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
def split_cells(state, E, V, E2e):
    """Splits the cells (E) using the E2e matrix which says which edge needs to be refined. Refinement is done according
    to the 1-2-3 flagging (1 flag -> 2 cells, 2 flag -> 3 cells, 3 flag -> 4 cells).

    :param state: [:, 4] Numpy array of state vectors, [rho, rho*u, rho*v, rho*E]
    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param E2e: Element-2-Edges for which edges need to be refined
    :returns: new_state: The new state vector array interpolated onto the new mesh, new_E: The new Element-2-Node array,
     V: The new array of node x-y positions
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
                    # Check to see if the midpoints are already in the node matrix - if not, add them
                    if (np.sum(V - midpoint, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint, (1, 2))))

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
                    # Check to see if the midpoints are already in the node matrix - if not, add them
                    if (np.sum(V - midpoint1, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint1, (1, 2))))
                    if (np.sum(V - midpoint2, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint2, (1, 2))))

                    midpoint1i = np.argmin(np.sum(np.abs(V - midpoint1), axis=1))
                    midpoint2i = np.argmin(np.sum(np.abs(V - midpoint2), axis=1))

                    # Check to see which angle is bigger - that is the one that gets split

                    # TODO: Clean up this hacky bullshit to make Numba happy
                    rowNodes = E[i]
                    edgeLengthsIndices = np.vstack((np.array([rowNodes[j], rowNodes[j-2]]),
                                                    np.array([rowNodes[j-2], rowNodes[j-1]]),
                                                    np.array([rowNodes[j-1], rowNodes[j]])))
                    edgeLengths, _ = cgf.edgePropertiesCalculator(edgeLengthsIndices, V)
                    l_not_flagged = edgeLengths[0]
                    l_f1 = edgeLengths[1]
                    l_f2 = edgeLengths[2]

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
            # Check to see if the midpoints are already in the node matrix - if not, add them
            if (np.sum(V - midpoint1, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint1, (1, 2))))
            if (np.sum(V - midpoint2, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint2, (1, 2))))
            if (np.sum(V - midpoint3, axis=1)).min() != 0: V = np.vstack((V, np.reshape(midpoint3, (1, 2))))

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

    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param BE: [:, 4] Numpy array boundary Edge Matrix [nodeA, nodeB, cell, boundary flag]
    :param E2e: Element-2-Edges for which edges need to be refined
    :returns: new_BE: [:, 3] Numpy array of nodes that make up the new split boundaries
    """
    new_BE = np.empty((0, 3), dtype=np.int32)

    for be in BE:
        be_node_pair = be[0:2]
        be_node_pair_flipped = np.flip(be_node_pair)

        # Check to see which edge the pair of nodes corresponds to on the BE's cell, then check to see if it was flagged
        # for refinement, if it was then split the BE
        if (np.sum(be_node_pair - E[be[2]][0:2]) == 0 and E2e[be[2], 0] != 0) or \
        (np.sum(be_node_pair_flipped - E[be[2]][0:2]) == 0 and E2e[be[2], 0] != 0):
            midpoint = (V[be[0]] + V[be[1]]) / 2
            midpointi = np.argmin(np.sum(np.abs(V - midpoint), axis=1))
            new_BE_temp1 = np.array((be[0], midpointi, be[3]), dtype=np.int32)
            new_BE_temp2 = np.array((midpointi, be[1], be[3]), dtype=np.int32)
            new_BE_temp = np.vstack((new_BE_temp1, new_BE_temp2))
            new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (2, 3))))
            continue

        elif (np.sum(be_node_pair - E[be[2]][1::]) == 0 and E2e[be[2], 1] != 0) or \
        (np.sum(be_node_pair_flipped - E[be[2]][1::]) == 0 and E2e[be[2], 1] != 0):
            midpoint = (V[be[0]] + V[be[1]]) / 2
            midpointi = np.argmin(np.sum(np.abs(V - midpoint), axis=1))
            new_BE_temp1 = np.array((be[0], midpointi, be[3]), dtype=np.int32)
            new_BE_temp2 = np.array((midpointi, be[1], be[3]), dtype=np.int32)
            new_BE_temp = np.vstack((new_BE_temp1, new_BE_temp2))
            new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (2, 3))))
            continue

        elif (np.sum(be_node_pair - np.flip(E[be[2]])[0::2]) == 0 and E2e[be[2], 2] != 0) or \
        (np.sum(be_node_pair_flipped - np.flip(E[be[2]])[0::2]) == 0 and E2e[be[2], 2] != 0):
            midpoint = (V[be[0]] + V[be[1]]) / 2
            midpointi = np.argmin(np.sum(np.abs(V - midpoint), axis=1))
            new_BE_temp1 = np.array((be[0], midpointi, be[3]), dtype=np.int32)
            new_BE_temp2 = np.array((midpointi, be[1], be[3]), dtype=np.int32)
            new_BE_temp = np.vstack((new_BE_temp1, new_BE_temp2))
            new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (2, 3))))
            continue
        else:
            new_BE_temp = np.array((be[0], be[1], be[3]), dtype=np.int32)
            new_BE = np.vstack((new_BE, np.reshape(new_BE_temp, (1, 3))))

    return new_BE


def write_new_mesh(E, V, new_BE, Bname, fname):
    """Writes the refined mesh to a *.gri file.

    :param E: Element-2-Node matrix
    :param V: Coordinates of nodes
    :param new_BE: Nx3 matrix, [nodeA, nodeB, BC Flag]
    :param Bname: Boundary edge names
    :param fname: Filename to write the mesh to
    :returns: Nothing. Writes the new mesh to fname.
    """
    # Write out the mesh to fname.gri and then parse it again using the given reading/hashing functionaility
    f = open(fname, 'w')

    # gri header
    f.write('{0} {1} 2\n'.format(V.shape[0], E.shape[0]))

    # Node information
    for node in V:
        f.write('%.15e %.15e\n' % (node[0], node[1]))

    # Boundary Edge pair writing
    num_unique_bes = np.unique(new_BE[:, 2])
    f.write('%i\n' % (num_unique_bes.shape[0]))
    for i in num_unique_bes:
        be_slice = new_BE[new_BE[:, 2] == i, 0:2]

        f.write('%i %i %s\n' % (be_slice.shape[0], 2, Bname[i]))
        for row in be_slice:
            f.write('%i %i\n' % (row[0] + 1, row[1] + 1))

    # Element writing
    f.write('{0} 1 TriLagrange\n'.format(E.shape[0]))
    for element in E:
        temp_element = reorient_ccw(element[0], element[1], element[2], V)
        f.write('{0} {1} {2}\n'.format(temp_element[0] + 1, temp_element[1] + 1, temp_element[2] + 1))
    f.close()


def adapt_mesh(state, E, V, IE, BE, Bname, f, y, fname):
    """Adapts the mesh using the error between cell states which is described as jumps in Mach number.

    :param state: [:, 4] Numpy array of state vectors, each row is 1 cell's state [rho, rho*u, rho*v, rho*E]
    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :param IE: Internal edge array [nodeA, nodeB, cell left, cell right]
    :param BE: Boundary edge array [nodeA, nodeB, cell index, boundary flag]
    :param Bname: Boundary edge names
    :param f: Number of edges to flag for refinement
    :param y: Ratio of specific heats - gamma
    :param fname: Filename to write the mesh to
    :returns: new_state: New state vector array interpolated onto the new mesh, mesh: Newly refined mesh
    """
    hashed_refinement_matrix = find_flagged_edges(state, E, V, IE, BE, f, y)
    new_state, new_E, new_V = split_cells(state, E, V, hashed_refinement_matrix)
    new_boundaries = split_boundaries(E, new_V, BE, hashed_refinement_matrix)
    write_new_mesh(new_E, new_V, new_boundaries, Bname, fname)
    mesh = readgri.readgri(fname)
    return new_state, mesh

