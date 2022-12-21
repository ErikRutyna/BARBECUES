import numpy as np
import math



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