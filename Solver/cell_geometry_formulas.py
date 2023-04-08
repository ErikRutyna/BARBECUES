from numba import njit
import numpy as np
import math


@njit(cache=True)
def centroid(E, V):
    """Returns an array of centroids for all cells in the mesh.

    :param E: Element-2-Node Matrix
    :param V: Node coordinates
    :return: centroids of all cells in mesh
    """
    centroids = np.zeros((E.shape[0], 2))

    for i in range(centroids.shape[0]):
        c1 = V[E[i, 0], :]
        c2 = V[E[i, 1], :]
        c3 = V[E[i, 2], :]
        centroids[i] = np.divide(np.add(c1, np.add(c2, c3)), 3)

    return centroids


@njit(cache=True)
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


@njit(cache=True)
def area_calculator(E, V):
    """Calculates the area of the two triangular cells for the given indices.

    :param E: Element-2-Node Matrix
    :param V: Node coordinates
    :return area: Area of the cells
    """
    area = np.zeros(E.shape[0])
    for i in range(E.shape[0]):
        nodes = E[i]
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        # print(cellIndex)
        a, _ = edge_properties_calculator(V[nodes[0]], V[nodes[1]])
        b, _ = edge_properties_calculator(V[nodes[1]], V[nodes[2]])
        c, _ = edge_properties_calculator(V[nodes[2]], V[nodes[0]])

        s = (a + b + c) / 2

        area[i] = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area