import numpy as np
from numba import njit
from scipy import sparse
import math
import flux_roe


def edgehash2(E, B):
    """Another version of the edge hashing function originally written by Krzysztof J. Fidkowski @ University of
    Michigan. This one was written by myself, Erik Rutyna, as a way to understand his version.

    :param E: [:, 3] numpy array that is the Element-2-Node matrix
    :param B: [:, 2] numpy array that consists of all the boundary edges
    :returns: IE: [:, 4] numpy array of internal edges [nodeA, nodeB, cell i, cell j],
    BE: [:, 4] numpy array of boundary edges [nodeA, nodeB, cell i, boundary edge flag]
    """
    Ne = E.shape[0]; Nn = np.amax(E)+1
    IE = np.empty((1, 4), dtype=int)
    BE = np.empty((1, 4), dtype=int)
    # Loop over cells - start grabbing node-pair combinations for unique edges
    for e in range(Ne):
        np1, np2, np3 = [E[e, 0], E[e, 1]], [E[e, 1], E[e, 2]], [E[e, 2], E[e, 0]]
        # Check if each pair is in the boundary pairs set, if it is then we know that this edge is a BE, and append to
        # BE matrix using the format [node1, node2, cell index number, boundary index number]
        for pair in [np1, np2, np3]:
            for i in range(len(B)):
                # Logically check for both nodes being in any of B[i] - works even if cell order is somehow changed
                if np.any(np.logical_or(np.logical_and(np.equal(pair[0], B[i][:, 0]), np.equal(pair[1], B[i][:, 1])), np.logical_and(np.equal(pair[1], B[i][:, 0]), np.equal(pair[0], B[i][:, 1])))):
                    # If the logic check is passed then the boundary node pair is for sure in this cell - append edge
                    # to the array
                    BE = np.vstack((BE, np.array([pair[0], pair[1], e, i])))
                    # Cannot continue due to corner cases - must loop over all Boundaries
            # Once past the boundary loops it means that the pair must be an internal edge, lots of logical checks
            if np.any(np.delete(np.logical_and(np.any(np.equal(pair[0], E[:, :]), axis=1), np.any(np.equal(pair[1], E[:, :]), axis=1)), e)):
                other_cell_index = list(np.where(np.logical_and(np.any(np.equal(pair[0], E[:, :]), axis=1),
                                                                     np.any(np.equal(pair[1], E[:, :]), axis=1)))[0])
                other_cell_index.remove(e)
                ie_temp = np.array([pair[0], pair[1], e, other_cell_index[0]])
                ie_temp_switch = np.array([pair[0], pair[1], other_cell_index[0], e])
                ie_temp_switch2 = np.array([pair[1], pair[0], e, other_cell_index[0]])
                ie_temp_switch3 = np.array([pair[1], pair[0], other_cell_index[0], e])
                # Check to see if this edge or a flip of the cell indices was already added to the list of IEs, if not
                # add it to the array of IEs

                # Fix duplicate problem for node pair swapping
                if not np.any(np.equal(ie_temp, IE).all(1)) and not np.any(np.equal(ie_temp_switch, IE).all(1))\
                        and not np.any(np.equal(ie_temp_switch2, IE).all(1)) and not np.any(np.equal(ie_temp_switch3, IE).all(1)):
                    IE = np.vstack((IE, ie_temp))

    IE = np.delete(IE, 0, axis=0)
    BE = np.delete(BE, 0, axis=0)

    return IE, BE


@njit(cache=True)
def edge_properties_calculator(node_a, node_b):
    """ Calculates the length and CCW norm out of a single edge

    :param node_a: X-Y Coordinates of node A
    :param node_b: X-Y Coordinates of node B
    :returns: length: Length of the edge from A->B, norm: Normal vector out of the edge in CCW fashion: [nx, ny]
    """

    length = math.sqrt((node_b[0] - node_a[0]) ** 2 + (node_b[1] - node_a[1]) ** 2)
    norm = np.array([(node_b[1] - node_a[1]) / length, (node_a[0] - node_b[0]) / length])

    return length, norm

# The following functions were written by Krzysztof J. Fidkowski @ University of Michigan
#-----------------------------------------------------------
# Identifies interior and boundary edges given element-to-node
# IE contains (n1, n2, elem1, elem2) for each interior edge
# BE contains (n1, n2, elem, bgroup) for each boundary edge
def edgehash(E, B):
    Ne = E.shape[0]; Nn = np.amax(E)+1
    H = sparse.lil_matrix((Nn, Nn), dtype=int)
    IE = np.zeros([int(np.ceil(Ne*1.5)),4], dtype=int)
    ni = 0
    for e in range(Ne):
        for i in range(3):
            n1, n2 = E[e,i], E[e,(i+1)%3]
            if (H[n2,n1] == 0):
                H[n1,n2] = e+1
            else:
                eR = H[n2,n1]-1
                IE[ni,:] = n1, n2, e, eR
                H[n2,n1] = 0
                ni += 1
    IE = IE[0:ni,:]
    # boundaries
    nb0 = nb = 0
    for g in range(len(B)): nb0 += B[g].shape[0]
    BE = np.zeros([nb0,4], dtype=int)
    for g in range(len(B)):
        Bi = B[g]
        for b in range(Bi.shape[0]):
            n1, n2 = Bi[b,0], Bi[b,1]
            if (H[n1,n2] == 0): n1,n2 = n2,n1
            BE[nb,:] = n1, n2, H[n1,n2]-1, g
            nb += 1
    return IE, BE


#-----------------------------------------------------------
def readgri(fname):
    f = open(fname, 'r')
    Nn, Ne, dim = [int(s) for s in f.readline().split()]
    # read vertices
    V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
    # read boundaries
    NB = int(f.readline())
    B = []; Bname = []
    for i in range(NB):
        s = f.readline().split(); Nb = int(s[0]); Bname.append(s[2])
        Bi = np.array([[int(s)-1 for s in f.readline().split()] for n in range(Nb)])
        B.append(Bi)
    # read elements
    Ne0 = 0; E = []
    while (Ne0 < Ne):
        s = f.readline().split(); ne = int(s[0])
        Ei = np.array([[int(s)-1 for s in f.readline().split()] for n in range(ne)])
        E = Ei if (Ne0==0) else np.concatenate((E,Ei), axis=0)
        Ne0 += ne
    f.close()
    # make IE, BE structures
    IE, BE = edgehash(E, B)
    Mesh = {'V':V, 'E':E, 'IE':IE, 'BE':BE, 'Bname':Bname }
    return Mesh

#-----------------------------------------------------------
def writegri(Mesh, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']; Bname = Mesh['Bname'];
    Nv, Ne, Nb = V.shape[0], E.shape[0], BE.shape[0]
    f = open(fname, 'w')
    f.write('%d %d 2\n'%(Nv, Ne))
    for i in range(Nv):
        f.write('%.15e %.15e\n'%(V[i,0], V[i,1]));
    nbg = 0
    for i in range(Nb): nbg = max(nbg, BE[i,3])
    nbg += 1
    f.write('%d\n'%(nbg))
    for g in range(nbg):
        nb = 0
        for i in range(Nb): nb += (BE[i,3] == g)
        f.write('%d 2 %s\n'%(nb, Bname[g]))
        for i in range(Nb):
            if (BE[i,3]==g): f.write('%d %d\n'%(BE[i,0]+1, BE[i,1]+1))
    f.write('%d 1 TriLagrange\n'%(Ne))
    for i in range(Ne):
        f.write('%d %d %d\n'%(E[i,0]+1, E[i,1]+1, E[i,2]+1))
    f.close()


# -----------------------------------------------------------
def adapt(Mesh, U, saveprefix, iadapt, config):
    print('\nAdapting the mesh: iteration = %d\n' % (iadapt))
    V = Mesh['V'];
    E = Mesh['E'];
    IE = Mesh['IE'];
    BE = Mesh['BE']
    Nv, Ne, Ni, Nb = V.shape[0], E.shape[0], IE.shape[0], BE.shape[0]

    # put interior and boundary edges together into one structure
    IBE = np.zeros([Ni + Nb, 7], dtype=np.int)
    for i in range(Ni): IBE[i, :] = [IE[i, 0], IE[i, 1], IE[i, 2], IE[i, 3], IE[i, 4], IE[i, 5], -1]
    for i in range(Nb): IBE[Ni + i, :] = [BE[i, 0], BE[i, 1], BE[i, 2], -1, BE[i, 4], -1, BE[i, 3]]
    Nbi = Ni + Nb
    # make error and refinement indicators over all edges
    EInd = np.zeros(Nbi); RInd = np.zeros(Nbi,dtype=np.int);

    # Nx2 array [error, cell i]
    error_index_ie = np.zeros((len(Mesh['IE']), 2))
    error_index_be = np.zeros((len(Mesh['BE']), 2))
    # sort error indicator
    nref = int(0.03*Nbi);

    for i in range(len(Mesh['BE'])):
        # Boundary Edges
        be = Mesh['BE'][i]
        # No error on the inflow cells
        if Mesh['Bname'][be[3]] == 'Inflow' or Mesh['Bname'][be[3]] == 'Outflow':
            continue
        else:
            be_l, be_n = edge_properties_calculator(Mesh['V'][be[0]], Mesh['V'][be[1]])

            # Cell i quantities
            u = U[be[2]][1] / U[be[2]][0]
            v = U[be[2]][2] / U[be[2]][0]
            q = np.dot(be_n, np.array([u, v]))
            flux_c = flux_roe.stateFluxEuler2D(U[be[2]], config)
            h_l = (flux_c[3, 0] + flux_c[3, 1]) / ( flux_c[0, 0] + flux_c[0, 1])
            c = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))

            error = abs(q / c) * be_l

            error_index_be[i, 0] = error
            error_index_be[i, 1] = be[2]

    for i in range(len(Mesh['IE'])):
        # Internal Edges
        ie = Mesh['IE'][i]
        ie_l, ie_n = edge_properties_calculator(Mesh['V'][ie[0]], Mesh['V'][ie[1]])

        # Left cell/cell i quantities
        u_l = U[ie[2]]
        u_l_u = u_l[1] / u_l[0]
        u_l_v = u_l[2] / u_l[0]
        q_l = np.linalg.norm([u_l_u, u_l_v])
        flux_l = flux_roe.stateFluxEuler2D(u_l, config)
        h_l = (flux_l[3, 0] + flux_l[3, 1]) / ( flux_l[0, 0] + flux_l[0, 1])
        c_l = math.sqrt((config.y - 1) * (h_l - q ** 2 / 2))
        m_l = q_l / c_l

        # Right cell/cell N quantities
        u_r = U[ie[3]]
        u_r_u = u_r[1] / u_r[0]
        u_r_v = u_r[2] / u_r[0]
        q_r = np.linalg.norm([u_r_u, u_r_v])
        flux_r = flux.stateFluxEuler2D(u_r, config)
        h_r = (flux_r[3, 0] + flux_r[3, 1]) / ( flux_r[0, 0] + flux_r[0, 1])
        c_r = math.sqrt((config.y - 1) * (h_r - q ** 2 / 2))

        m_r = q_r / c_r

        error = abs(m_l - m_r) * ie_l

        error_index_ie[i, 0] = error
        error_index_ie[i, 1] = ie[2]

    # Total list of all locations where Mach number jumps are too high
    total_error_index = np.vstack((error_index_be, error_index_ie))
    error_index = np.flip(total_error_index[total_error_index[:, 0].argsort()], axis=0)
    error_index = error_index[0:math.floor(0.03 * len(error_index[:, 0])), 1]
    sortInd = np.argsort(-error_index);

    # build E2I structure = map from elements to edges
    E2I = np.zeros([Ne, 3], dtype=np.int)
    for i in range(Nbi):
        eL, eR, fL, fR = IBE[i, 2], IBE[i, 3], IBE[i, 4], IBE[i, 5]
        E2I[eL, fL] = i;
        if (eR >= 0): E2I[eR, fR] = i

    # flag edges (index refers to added node index)
    for i in range(nref):
        ie = sortInd[i]
        RInd[ie] = Nv + i

    # flag more edges
    neflag = 0
    for e in range(Ne):
        nr = 0;
        I = E2I[e, :]
        for k in range(3):
            if (RInd[I[k]] > 0):
                if (nr == 0): neflag += 1
                nr += 1
        if (nr > 0):
            for k in range(3):
                if (RInd[I[k]] == 0):
                    RInd[I[k]] = -(Nv + nref)
                    nref += 1

    for i in range(Nbi): RInd[i] = abs(RInd[i])

    # print('iadapt=%d, new nref=%d, neflag=%d\n'%(iadapt,nref, neflag));

    # expand V
    Vnew = np.zeros([Nv + nref, 2])
    for i in range(Nv): Vnew[i, :] = V[i, :]

    # expand BE
    BEnew = np.zeros([Nb + nref, 4], dtype=np.int)
    for i in range(Nb): BEnew[i, :] = [BE[i, j] for j in range(4)]
    nbface = Nb

    for ie in range(Nbi):
        nn = RInd[ie]
        if (nn == 0): continue
        # calculate new node coordinates
        n1, n2 = IBE[ie, 0], IBE[ie, 1]
        Vnew[nn, :] = 0.5 * (V[n1, :] + V[n2, :])
        if (IBE[ie, 6] >= 0):
            BEnew[ie - Ni, :] = [n1, nn, IBE[ie, 2], IBE[ie, 6]]
            BEnew[nbface, :] = [nn, n2, IBE[ie, 2], IBE[ie, 6]]
            nbface += 1
    BEnew = BEnew[0:nbface, :]

    # estimate how many elements to add, allocate memory
    nrefe = 3 * nref  # over-estimate
    Enew = np.zeros([Ne + nrefe, 3], dtype=np.int)  # new element list
    Eparent = np.ones(Ne + nrefe, dtype=np.int);
    Eparent *= -1;
    for i in range(Ne): Enew[i, :] = E[i, :]
    nelem = Ne  # index of next element to be added
    # loop over elements, refine, augment list
    for e in range(Ne):
        nr = 0;
        I = E2I[e, :]
        for k in range(3):
            if (RInd[I[k]] > 0): nr += 1
        if (nr == 0):
            continue
        elif (nr == 1):
            for k in range(3):
                if (RInd[I[k]] > 0): break
            e0, e1 = e, nelem
            n0, n1, n2, n3 = E[e, k], E[e, (k + 1) % 3], E[e, (k + 2) % 3], RInd[I[k]]
            Enew[e0, :] = [n0, n1, n3]
            Enew[e1, :] = [n0, n3, n2]
            Eparent[e0] = Eparent[e1] = e;
            nelem += 1
        elif (nr == 2):
            for k in range(3):
                if (RInd[I[k]] == 0): break
            e0, e1, e2 = e, nelem, nelem + 1
            n0, n1, n2 = E[e, k], E[e, (k + 1) % 3], E[e, (k + 2) % 3]
            m1, m2 = RInd[I[(k + 1) % 3]], RInd[I[(k + 2) % 3]]
            Enew[e0, :] = [n0, m2, m1]
            Enew[e1, :] = [n1, m1, m2]
            Enew[e2, :] = [n2, m1, n1]
            Eparent[e0] = Eparent[e1] = Eparent[e2] = e;
            nelem += 2
        elif (nr == 3):
            e0, e1, e2, e3 = e, nelem, nelem + 1, nelem + 2
            n0, n1, n2 = E[e, 0], E[e, 1], E[e, 2],
            m0, m1, m2 = RInd[I[0]], RInd[I[1]], RInd[I[2]]
            Enew[e0, :] = [m0, m1, m2]
            Enew[e1, :] = [n0, m2, m1]
            Enew[e2, :] = [n1, m0, m2]
            Enew[e3, :] = [n2, m1, m0]
            Eparent[e0] = Eparent[e1] = Eparent[e2] = Eparent[e3] = e;
            nelem += 3
    Enew = Enew[0:nelem, :]

    print('iadapt=%d, Ne=%d -> nelem=%d\n' % (iadapt, Ne, nelem));

    # map solution, U
    sr = U.shape[1];
    Uold = U.copy()
    U = np.zeros([nelem, sr])
    for i in range(Ne): U[i, :] = Uold[i, :]
    for i in range(nelem - Ne): U[Ne + i, :] = Uold[Eparent[Ne + i], :]

    # write out new .gri file
    fname = '%s_mesh%02d.gri' % (saveprefix, iadapt)
    Mesh['E'] = Enew;
    Mesh['V'] = Vnew;
    Mesh['BE'] = BEnew;
    writegri(Mesh, fname);
    # read in new .gri file (overwrite structures)
    Mesh = readgri(fname)

    return Mesh, U
#-----------------------------------------------------------
