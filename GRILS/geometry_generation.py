from numba import njit
import numpy as np
import math
import os

@njit(cache=True)
def reorient_ccw(v1, v2, v3, V):
    """Re-orients the nodes to be in counter-clockwise order.

    :param v1: First node index
    :param v2: Second node index
    :param v3: Third node index
    :param V: [:, 2] x-y coordinate pair array of node positions
    :returns: Returns a numpy array w/ the same node indices but in CCW order
    """
    v1coord = V[v1]
    v2coord = V[v2]
    v3coord = V[v3]

    # https://www.geeksforgeeks.org/orientation-3-ordered-points/
    if ((v2coord[1] - v1coord[1]) * (v3coord[0] - v2coord[0]) -
        (v3coord[1] - v2coord[1]) * (v2coord[0] - v1coord[0])) < 0:
        return np.array([v1, v2, v3])
    else:
        return np.array([v3, v2, v1])


@njit(cache=True)
def edgeLengthCalc(node_a, node_b):
    """Calculates the length of a single edge.

    :param node_a: [x, y] coordinates of node A
    :param node_b: [x, y] coordinates of node B
    :returns: length: Length of the edge from A->B
    """
    length = math.sqrt((node_b[0] - node_a[0]) ** 2 + (node_b[1] - node_a[1]) ** 2)

    return length


@njit(cache=True)
def cellQualityCalculator(E, V):
    """Calculates the average cell quality of a given mesh with E elements and nodes at V positions.

    :param E: [:, 3] Element-2-Node matrix where each row are 3 indices in V, [v1, v2, v3]
    :param V: [:, 2] x-y coordinates for each node
    :returns: np.mean(quality): Mean cell quality
    """
    quality = np.zeros(E.shape[0])
    for i in range(E.shape[0]):
        nodes = E[i]
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        a = edgeLengthCalc(V[nodes[0]], V[nodes[1]])
        b = edgeLengthCalc(V[nodes[1]], V[nodes[2]])
        c = edgeLengthCalc(V[nodes[2]], V[nodes[0]])

        s = (a + b + c) / 2

        quality[i] = 4 *  math.sqrt(3) * np.sqrt(s * (s - a) * (s - b) * (s - c)) / (a ** 2 + b ** 2 + c ** 2)
    return np.mean(quality)


def sqr_bbox_wall_gen(L, W, ds):
    """Generates a [:, 2] numpy array whose values are points along the edge of an "L x W" rectangle.

    :param L: Unitless length of the computational domain
    :param W: Unitless width (height in 2D) of the computational domain
    :param ds: Maximum distance between two adjacent points
    :returns: V: [:, 2] x-y coordinate pairs that enclose an L x W bounding box with ds step size
    """
    xCoordLeftToRight = np.arange(-L / 2, L / 2 + ds, ds)
    yCoordTop = np.ones(xCoordLeftToRight.shape[0]) * W / 2
    yCoordBot = -np.ones(xCoordLeftToRight.shape[0]) * W / 2

    yCoordBotToTop = np.arange(-W / 2, W / 2, ds)
    xCoordLeft = -np.ones(yCoordBotToTop.shape[0]) * L / 2
    xCoordRight = np.ones(yCoordBotToTop.shape[0]) * L / 2

    xCoordinates = np.hstack((xCoordLeftToRight, xCoordRight[1::], np.flip(xCoordLeftToRight[1::]), xCoordLeft))
    yCoordinates = np.hstack((yCoordTop, np.flip(yCoordBotToTop[1::]), yCoordBot[1::], yCoordBotToTop))

    V = np.array(([xCoordinates, yCoordinates])).T
    return V

def circle_bbox_wall_gen(r, dtheta):
    """Generates a [:, 2] numpy array whose values are points along the edge of an "r" sized circle.

    :param r: Radius
    :param dtheta: Discretized theta
    :returns: V - [:, 2] x-y coordinate pairs that enclose a circle with radius "r" and an arc size of "r*dtheta"
    """
    theta = np.arange(0, 360 + dtheta, dtheta)
    theta = theta[1::] * np.pi / 180

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    V = np.array(([x, y])).T

    return V


def diamond_bbox_wall_gen(c, theta, ds):
    """Generates a [:, 2] numpy array that consists of points outlining a diamond shape with chord length c and half
    angle theta.

    :param c: Straight line chord length from leading edge of the diamond to the trailing edge
    :param theta: Half angle of the diamond, measured from the cord going counter-clockwise
    :param ds: Unitless distance step sized used for marching down the perimeter of the diamond

    :returns: V - [:, 2] x-y coordinate pairs that make the perimeter of the diamond shape
    """
    # Assume center of diamond is at x-y position of [0, 0]
    y_max = np.sin(np.deg2rad(theta / 2)) * c / 2
    y_min = -y_max
    x_max = c / 2
    x_min = -x_max

    # Position vectors that go from the leading edge to the trailing edge of the diamond
    leadingEdgeTopPosVec = np.array((-x_min, y_min))

    # The number of points we want to discretize either the LE or TE
    nPointsEdge = np.int(np.ceil(np.linalg.norm(leadingEdgeTopPosVec) / ds))

    leadingEdgeTopX = np.linspace(x_min, 0, nPointsEdge)
    leadingEdgeBotX = np.linspace(x_min, 0, nPointsEdge)

    trailingEdgeTopX = np.linspace(0, x_max, nPointsEdge)
    trailingEdgeBotX = np.linspace(0, x_max, nPointsEdge)

    leadingEdgeTopY = np.linspace(0, y_max, nPointsEdge)
    leadingEdgeBotY = np.linspace(0, y_min, nPointsEdge)

    trailingEdgeTopY = np.linspace(y_max, 0, nPointsEdge)
    trailingEdgeBotY = np.linspace(y_min, 0, nPointsEdge)

    x = np.hstack((leadingEdgeTopX.flatten()[1::], trailingEdgeTopX[1::].flatten(),
                   np.flip(trailingEdgeBotX)[1::].flatten(), np.flip(leadingEdgeBotX)[1::].flatten()))
    y = np.hstack((leadingEdgeTopY.flatten()[1::], trailingEdgeTopY[1::].flatten(),
                   np.flip(trailingEdgeBotY)[1::].flatten(), np.flip(leadingEdgeBotY)[1::].flatten()))
    V = np.array((x, y)).T
    return V


def triangle_bbox_wall_gen(p1, p2, p3, ds):
    """Generates a [:, 2] numpy array whose values are points along the edge of a triangle are made up of the x-y
    vertices "p1", "p2", "p3".

    :param p1: Vertex with points [x, y]
    :param p2: Vertex with points [x, y]
    :param p3: Vertex with points [x, y]
    :param ds: Distance step sized used for discretizing down the lengths between two vertices.
    :returns: V - [:, 2] x-y coordinate pairs that enclose a triangle with points "p1", "p2", and "p3"
    """

    # Position vectors
    q1 = p2 - p1
    q2 = p3 - p2
    q3 = p1 - p3


    perim = np.linalg.norm(q1) + np.linalg.norm(q2) + np.linalg.norm(q3)

    total_steps = np.ceil(perim / ds)

    q1_steps = int(np.ceil(np.linalg.norm(q1) / perim * total_steps))
    q2_steps = int(np.ceil(np.linalg.norm(q2) / perim * total_steps))
    q3_steps = int(np.ceil(np.linalg.norm(q3) / perim * total_steps))

    q1x = np.linspace(p1[0], p2[0], q1_steps)
    q1y = np.linspace(p1[1], p2[1], q1_steps)

    q2x = np.linspace(p2[0], p3[0], q2_steps)
    q2y = np.linspace(p2[1], p3[1], q2_steps)

    q3x = np.linspace(p3[0], p1[0], q3_steps)
    q3y = np.linspace(p3[1], p1[1], q3_steps)

    x = np.hstack((q1x[1::], q2x[1::], q3x[1::]))
    y = np.hstack((q1y[1::], q2y[1::], q3y[1::]))
    V = np.array((x, y)).T

    return V


def csv_bbox_Wall_gen(fname):
    """Reads a *.csv file that has nodes in Nx2: x,y -like position and turns this into a set of nodes for bounding
    box geometry.

    :param fname: Filename under Meshes directory
    :returns: V - [:, 2] x-y coordinate pairs from the fname csv file
    """
    V = np.genfromtxt(os.path.dirname(os.getcwd()) + '\Meshes\\' + fname, dtype=np.double, delimiter=',')
    return V


def internal_walls(wall_points, looped):
    """Returns an array of [:, 4] x-y coordinates that make up each edge for a given path of edges. Each row of the
    array consists of [x1, y1, x2, y2] for where [x1, y1] refer to the first point in a line segment of the edge for the
    given path of edges, and [x2, y2] consist of the second point in the line segment of the edge for the given path
    of edges.

    :param wall_points: [:, 2] array of x-y coordinate pairs that make up the nodes for all edges
    :param looped: Boolean - True for when the wall loops back from the point to the first, false if the wall does not
    :returns: [:, 4] x-y coordinates in the form described.
    """
    wall_edges = []
    for i in range(wall_points.shape[0]):
        if i == (wall_points.shape[0]-1) and looped:
            wall_edges.append(np.array([wall_points[-1], wall_points[0]]).flatten())
        else: wall_edges.append(np.array([wall_points[i], wall_points[i+1]]).flatten())
    return np.array(wall_edges)
