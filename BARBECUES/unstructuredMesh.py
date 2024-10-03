import numpy as np
from numba import njit
from scipy import sparse

class UnstructuredMesh:

    def __init__(self, filepath):
        self.filepath = filepath

        # Use Krzysztof J. Fidkowski's GRI mesh reader
        self.griDictionary = self.readGRI(filepath)

        # Copies of data for ease of use
        self.nodes          = self.griDictionary['V']
        self.elements       = self.griDictionary['E']
        self.internalEdges  = self.griDictionary['IE']
        self.boundaryEdges  = self.griDictionary['BE']
        self.boundaryName   = self.griDictionary['Bname']

        # Edge properties for flux calculations
        self.internalEdgeLengths, self.internalEdgeNorms = self.edgePropertiesCalculator(self.internalEdges[:, 0:2], self.nodes)
        self.boundaryEdgeLengths, self.boundaryEdgeNorms = self.edgePropertiesCalculator(self.boundaryEdges[:, 0:2], self.nodes)

        # Cell Centroid Information
        self.cellCentroids = self.cellCentroidCalculator(self.elements, self.nodes)

        # Cell Area Information
        self.cellArea = self.cellAreaCalculator(self.elements, self.nodes)

    def edgePropertiesCalculator(self, edgeIndices, nodes):
        return self.edgePropertiesCalculatorNJIT(edgeIndices, nodes)

    @staticmethod
    @njit(cache=True)
    def edgePropertiesCalculatorNJIT(edgeIndices, nodes):
        deltaX = nodes[edgeIndices[:, 0], 0] - nodes[edgeIndices[:, 1], 0]
        deltaY = nodes[edgeIndices[:, 0], 1] - nodes[edgeIndices[:, 1], 1]

        length = np.sqrt(np.multiply(deltaX, deltaX) + np.multiply(deltaY, deltaY))

        norm = np.transpose(np.vstack((np.divide(-deltaY, length), np.divide(deltaX, length))))

        return length, norm


    def cellCentroidCalculator(self, elements, nodes):
        return self.centroidCalculatorNJIT(elements, nodes)

    @staticmethod
    @njit(cache=True)
    def centroidCalculatorNJIT(elements, nodes):
        return (nodes[elements[:, 0]] + nodes[elements[:, 1]] + nodes[elements[:, 2]]) / 3


    def cellAreaCalculator(self, elements, nodes):

        edgeLengthA, _ = self.edgePropertiesCalculatorNJIT(elements[:, 0:2], nodes)
        edgeLengthB, _ = self.edgePropertiesCalculatorNJIT(elements[:, 1::], nodes)
        edgeLengthC, _ = self.edgePropertiesCalculatorNJIT(elements[:, 2::-2], nodes)

        return self.cellAreaCalculatorNJIT(edgeLengthA, edgeLengthB, edgeLengthC)

    @staticmethod
    @njit(cache=True)
    def cellAreaCalculatorNJIT(a, b, c):

        s = (a + b + c) / 2

        sMinusA = s - a
        sMinusB = s - b
        sMinusC = s - c

        return np.sqrt(np.multiply(s, np.multiply(sMinusA, np.multiply(sMinusB, sMinusC))))

    # ---------------------------------------------------------------------------------------
    # The following functions were written by Krzysztof J. Fidkowski @ University of Michigan
    # ---------------------------------------------------------------------------------------
    def readGRI(self, fname):
        f = open(fname, 'r')
        Nn, Ne, dim = [int(s) for s in f.readline().split()]
        # read vertices
        V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
        # read boundaries
        NB = int(f.readline())
        B = []
        Bname = []
        for i in range(NB):
            s = f.readline().split()
            Nb = int(s[0])
            Bname.append(s[2])
            Bi = np.array([[int(s) - 1 for s in f.readline().split()] for n in range(Nb)])
            B.append(Bi)
        # read elements
        Ne0 = 0
        E = []
        while (Ne0 < Ne):
            s = f.readline().split()
            ne = int(s[0])
            Ei = np.array([[int(s) - 1 for s in f.readline().split()] for n in range(ne)])
            E = Ei if (Ne0 == 0) else np.concatenate((E, Ei), axis=0)
            Ne0 += ne
        f.close()
        # make IE, BE structures
        IE, BE = self.edgehash(E, B)
        Mesh = {'V': V, 'E': E, 'IE': IE, 'BE': BE, 'Bname': Bname}
        return Mesh

    def writeGRI(self, Mesh, fname):
        V = Mesh['V']
        E = Mesh['E']
        BE = Mesh['BE']
        Bname = Mesh['Bname']
        Nv, Ne, Nb = V.shape[0], E.shape[0], BE.shape[0]
        f = open(fname, 'w')
        f.write('%d %d 2\n' % (Nv, Ne))
        for i in range(Nv):
            f.write('%.15e %.15e\n' % (V[i, 0], V[i, 1]))
        nbg = 0
        for i in range(Nb): nbg = max(nbg, BE[i, 3])
        nbg += 1
        f.write('%d\n' % (nbg))
        for g in range(nbg):
            nb = 0
            for i in range(Nb): nb += (BE[i, 3] == g)
            f.write('%d 2 %s\n' % (nb, Bname[g]))
            for i in range(Nb):
                if (BE[i, 3] == g): f.write('%d %d\n' % (BE[i, 0] + 1, BE[i, 1] + 1))
        f.write('%d 1 TriLagrange\n' % (Ne))
        for i in range(Ne):
            f.write('%d %d %d\n' % (E[i, 0] + 1, E[i, 1] + 1, E[i, 2] + 1))
        f.close()

    def edgehash(self, E, B):
        Ne = E.shape[0]
        Nn = np.amax(E) + 1
        H = sparse.lil_matrix((Nn, Nn), dtype=int)
        IE = np.zeros([int(np.ceil(Ne * 1.5)), 4], dtype=int)
        ni = 0
        for e in range(Ne):
            for i in range(3):
                n1, n2 = E[e, i], E[e, (i + 1) % 3]
                if (H[n2, n1] == 0):
                    H[n1, n2] = e + 1
                else:
                    eR = H[n2, n1] - 1
                    IE[ni, :] = n1, n2, e, eR
                    H[n2, n1] = 0
                    ni += 1
        IE = IE[0:ni, :]
        # boundaries
        nb0 = nb = 0
        for g in range(len(B)): nb0 += B[g].shape[0]
        BE = np.zeros([nb0, 4], dtype=int)
        for g in range(len(B)):
            Bi = B[g]
            for b in range(Bi.shape[0]):
                n1, n2 = Bi[b, 0], Bi[b, 1]
                if (H[n1, n2] == 0): n1, n2 = n2, n1
                BE[nb, :] = n1, n2, H[n1, n2] - 1, g
                nb += 1
        return IE, BE
