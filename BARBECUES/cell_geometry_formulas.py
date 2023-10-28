from numba import njit
import numpy as np
import math


@njit(cache=True)
def centroid(E, V):
    """Returns an array of centroids for all cells in the mesh.

    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :returns: [:, 2] Numpy array of x-y coordinates of the centroids of all cells
    """
    centroids = (V[E[:, 0]] + V[E[:, 1]] + V[E[:, 2]]) / 3

    return centroids


@njit(cache=True)
def edgePropertiesCalculator(edgeIndices, V):
    """ Calculates the length and CCW norm out of a single edge for a set of edges defined by edgeIndices

    :param edgeIndices: [:, 2] Numpy array of indices in V, [nodeA, nodeB]
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :returns: length: Length of the edge from A->B, norm: Normal vector out of the edge in CCW fashion: [nx, ny]
    """
    deltaX = V[edgeIndices[:, 0], 0] - V[edgeIndices[:, 1], 0]
    deltaY = V[edgeIndices[:, 0], 1] - V[edgeIndices[:, 1], 1]

    length = np.sqrt(np.multiply(deltaX, deltaX) + np.multiply(deltaY, deltaY))

    norm = np.vstack((np.divide(-deltaY, length), np.divide(deltaX, length)))
    norm = np.transpose(norm)
    return length, norm


@njit(cache=True)
def areaCalculator(E, V):
    """Calculates the area of the two triangular cells for the given indices.

    :param E: [:, 3] Numpy array of the Element-2-Node hashing
    :param V: [:, 2] Numpy array of x-y coordinates of node locations
    :returns: Area - [:, 1] Numpy array of area of the cells
    """
    edgeLengthA, _ = edgePropertiesCalculator(E[:, 0:2], V)
    edgeLengthB, _ = edgePropertiesCalculator(E[:, 1::], V)
    edgeLengthC, _ = edgePropertiesCalculator(E[:, 2::-2], V)

    s = (edgeLengthA + edgeLengthB + edgeLengthC) / 2

    sMinusA = s - edgeLengthA
    sMinusB = s - edgeLengthB
    sMinusC = s - edgeLengthC

    area = np.sqrt(np.multiply(s,
                               np.multiply(sMinusA,
                                           np.multiply(sMinusB, sMinusC))))

    return area