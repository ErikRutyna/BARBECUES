import numpy as np
#-----------------------------------------------------------
def adapt(Mesh, U, saveprefix, iadapt):
    print('\nAdapting the mesh: iteration = %d\n'%(iadapt))
    getEdgeInfo(Mesh); # get local edge information
    V = Mesh['V']; E = Mesh['E']; IE = Mesh['IE']; BE = Mesh['BE']
    Normals = getNormals(Mesh);
    inormal, bnormal = Normals['inormal'], Normals['bnormal']
    ilength, blength = Normals['ilength'], Normals['blength']
    INFLOW, OUTFLOW, ENGINE, EXIT = getBname(Mesh)
    Nv, Ne, Ni, Nb = V.shape[0], E.shape[0], IE.shape[0], BE.shape[0]

    # put interior and boundary edges together into one structure
    IBE = np.zeros([Ni+Nb,7], dtype=np.int)
    for i in range(Ni): IBE[i,:] = [IE[i,0], IE[i,1], IE[i,2], IE[i,3], IE[i,4], IE[i,5], -1]
    for i in range(Nb): IBE[Ni+i,:] = [BE[i,0], BE[i,1], BE[i,2], -1, BE[i,4], -1, BE[i,3]]
    Nbi = Ni + Nb

    # make error and refinement indicators over all edges
    EInd = np.zeros(Nbi); RInd = np.zeros(Nbi,dtype=np.int);
    # interior edges first
    for i in range(Ni):
        eL, eR, h = IBE[i,2], IBE[i,3], ilength[i]
        ML, MR = getMach(U[eL,:]), getMach(U[eR,:])
        EInd[i] = abs(ML-MR)*h
    # boundary edges next
    for i in range(Nb):
        eL, h = BE[i,2], blength[i]
        if (BE[i,3] == ENGINE):
            Mb = getMachn(U[eL,:], bnormal[i,:])
            EInd[Ni+i] = Mb*h

    # sort error indicator
    nref = int(0.03*Nbi);
    sortInd = np.argsort(-EInd);

    # build E2I structure = map from elements to edges
    E2I = np.zeros([Ne,3], dtype=np.int)
    for i in range(Nbi):
        eL, eR, fL, fR = IBE[i,2], IBE[i,3], IBE[i,4], IBE[i,5]
        E2I[eL,fL] = i;
        if (eR >= 0): E2I[eR,fR] = i

    # flag edges (index refers to added node index)
    for i in range(nref):
        ie = sortInd[i]
        RInd[ie] = Nv+i

    # flag more edges
    neflag=0
    for e in range(Ne):
        nr = 0; I = E2I[e,:]
        for k in range(3):
            if (RInd[I[k]] > 0):
                if (nr == 0): neflag += 1
                nr += 1
        if (nr > 0):
            for k in range(3):
                if (RInd[I[k]] == 0):
                    RInd[I[k]] = -(Nv+nref)
                    nref += 1

    for i in range(Nbi): RInd[i] = abs(RInd[i])

    #print('iadapt=%d, new nref=%d, neflag=%d\n'%(iadapt,nref, neflag));

    # expand V
    Vnew = np.zeros([Nv + nref, 2])
    for i in range(Nv): Vnew[i,:] = V[i,:]

    # expand BE
    BEnew = np.zeros([Nb+nref,4], dtype=np.int)
    for i in range(Nb): BEnew[i,:] = [BE[i,j] for j in range(4)]
    nbface = Nb

    for ie in range(Nbi):
        nn = RInd[ie]
        if (nn == 0): continue
        # calculate new node coordinates
        n1,n2 = IBE[ie,0], IBE[ie,1]
        Vnew[nn,:] = 0.5*(V[n1,:] + V[n2,:])
        if (IBE[ie,6] >= 0):
            BEnew[ie-Ni,:] = [n1, nn, IBE[ie,2], IBE[ie,6]]
            BEnew[nbface,:] = [nn, n2, IBE[ie,2], IBE[ie,6]]
            nbface += 1
    BEnew = BEnew[0:nbface,:]


    # estimate how many elements to add, allocate memory
    nrefe = 3*nref # over-estimate
    Enew = np.zeros([Ne+nrefe,3], dtype=np.int) # new element list
    Eparent = np.ones(Ne+nrefe, dtype=np.int); Eparent *= -1;
    for i in range(Ne): Enew[i,:] = E[i,:]
    nelem = Ne # index of next element to be added
    # loop over elements, refine, augment list
    for e in range(Ne):
        nr = 0; I = E2I[e,:]
        for k in range(3):
            if (RInd[I[k]] > 0): nr += 1
        if (nr == 0): continue
        elif (nr == 1):
            for k in range(3):
                if (RInd[I[k]] > 0): break
            e0, e1 = e, nelem
            n0, n1, n2, n3 = E[e,k], E[e,(k+1)%3], E[e,(k+2)%3], RInd[I[k]]
            Enew[e0,:] = [n0, n1, n3]
            Enew[e1,:] = [n0, n3, n2]
            Eparent[e0] = Eparent[e1] = e;
            nelem += 1
        elif (nr == 2):
            for k in range(3):
                if (RInd[I[k]] == 0): break
            e0, e1, e2 = e, nelem, nelem+1
            n0, n1, n2 = E[e,k], E[e,(k+1)%3], E[e,(k+2)%3]
            m1, m2 = RInd[I[(k+1)%3]], RInd[I[(k+2)%3]]
            Enew[e0,:] = [n0, m2, m1]
            Enew[e1,:] = [n1, m1, m2]
            Enew[e2,:] = [n2, m1, n1]
            Eparent[e0] = Eparent[e1] = Eparent[e2] = e;
            nelem += 2
        elif (nr == 3):
            e0, e1, e2, e3 = e, nelem, nelem+1, nelem+2
            n0, n1, n2 = E[e,0], E[e,1], E[e,2],
            m0, m1, m2 = RInd[I[0]], RInd[I[1]], RInd[I[2]]
            Enew[e0,:] = [m0, m1, m2]
            Enew[e1,:] = [n0, m2, m1]
            Enew[e2,:] = [n1, m0, m2]
            Enew[e3,:] = [n2, m1, m0]
            Eparent[e0] = Eparent[e1] = Eparent[e2] = Eparent[e3] = e;
            nelem += 3
    Enew = Enew[0:nelem,:]


    print('iadapt=%d, Ne=%d -> nelem=%d\n'%(iadapt, Ne,nelem));

    # map solution, U
    sr = U.shape[1];
    Uold = U.copy()
    U = np.zeros([nelem,sr])
    for i in range(Ne): U[i,:] = Uold[i,:]
    for i in range(nelem-Ne): U[Ne+i,:] = Uold[Eparent[Ne+i],:]

    # write out new .gri file
    fname = '%s_mesh%02d.gri'%(saveprefix, iadapt)
    Mesh['E'] = Enew; Mesh['V'] = Vnew; Mesh['BE'] = BEnew;
    writegri(Mesh, fname);
    # read in new .gri file (overwrite structures)
    Mesh = readgri(fname)

    return Mesh, U