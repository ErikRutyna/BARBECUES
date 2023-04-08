import os.path
from numba import njit
import numpy as np


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


# TODO: Write this smarter like the airfoil code
def sqr_bbox_wall_gen(L, W, ds):
    """Generates a numpy array that consists of coordinates of cell vertices located along the walls of the bounding box
    of the computational domain.

    :param L: Length of the computational domain
    :param W: Width (height in 2D space) of the computational domain

    return: V - [N, 2] x-y coordinate pairs of points that enclose an L x W bounding box with ds step
    """
    # Start at [pos, pos] and go clockwise
    x_L = np.arange(-L / 2, L / 2 + ds, ds)
    y_L_top = np.ones(x_L.shape[0]) * W / 2
    y_L_bot = -np.ones(x_L.shape[0]) * W / 2

    y_w = np.arange(-W / 2, W / 2, ds)
    x_W_left = -np.ones(y_w.shape[0]) * L / 2
    x_W_right = np.ones(y_w.shape[0]) * L / 2

    x_coord = np.hstack((x_L, x_W_right[1::], np.flip(x_L[1::]), x_W_left))
    y_coord = np.hstack((y_L_top, np.flip(y_w[1::]), y_L_bot[1::], y_w))

    V = np.array(([x_coord, y_coord])).T
    return V

def circle_bbox_wall_gen(r, dtheta):
    """Generates a numpy array that consists of coordinates of cell vertices for a circle with radius r with segments
    every dtheta intervals

    :param r: Radius
    :param dtheta: Discretized theta

    return: V - [N, 2] x-y coordinate pairs of points that enclose an L x W bounding box with ds step
    """
    theta = np.arange(0, 360 + dtheta, dtheta)
    theta = theta[1::] * np.pi / 180

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    V = np.array(([x, y])).T

    return V


def diamond_bbox_wall_gen(c, theta, ds):
    """Generates a [N, 2] numpy array that consists of the diamond shape with chord length c and half-angle theta.

    :param c: Straight line chord length from leading edge to trailing edge of the airfoil
    :param theta: Half-angle of the airfoil
    :param ds: Distance step sized used for marching down the length of the airfoil
    """


    # Assume center of airfoil is at x-y position of [0, 0]
    y_max = np.sin(np.deg2rad(theta / 2)) * c / 2
    y_min = -y_max
    x_max = c / 2
    x_min = -x_max

    # Position vectors that go from the leading edge to the trailing edge of
    # the diamond airfoil
    le_top_pos_vec = np.array((-x_min, y_min))

    # The number of points we want to discretize either the LE or TE of the
    # airfoil
    num_points_edge = np.int(np.ceil(np.linalg.norm(le_top_pos_vec) / ds))

    le_top_x_pos = np.linspace(x_min, 0, num_points_edge)
    le_bot_x_pos = np.linspace(x_min, 0, num_points_edge)

    te_top_x_pos = np.linspace(0, x_max, num_points_edge)
    te_bot_x_pos = np.linspace(0, x_max, num_points_edge)

    le_top_y_pos = np.linspace(0, y_max, num_points_edge)
    le_bot_y_pos = np.linspace(0, y_min, num_points_edge)

    te_top_y_pos = np.linspace(y_max, 0, num_points_edge)
    te_bot_y_pos = np.linspace(y_min, 0, num_points_edge)

    x = np.hstack((le_top_x_pos.flatten()[1::], te_top_x_pos[1::].flatten(), np.flip(te_bot_x_pos)[1::].flatten(), np.flip(le_bot_x_pos)[1::].flatten()))
    y = np.hstack((le_top_y_pos.flatten()[1::], te_top_y_pos[1::].flatten(), np.flip(te_bot_y_pos)[1::].flatten(), np.flip(le_bot_y_pos)[1::].flatten()))
    V = np.array((x, y)).T
    return V


def triangle_bbox_wall_gen(p1, p2, p3, ds):
    """Generates a [N, 2] numpy array that consists of the points that generate a triangle defined by vertices p1, p2,
    and p3.

    :param p1: Vertex with points [x, y]
    :param p2: Vertex with points [x, y]
    :param p3: Vertex with points [x, y]
    :param ds: Distance step sized used for discretizing down the lengths between two vertices.
    :return V: nodes that make the triangle
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
    """ Reads a *.csv file that has nodes in Nx2: x,y -like position and turns
    this into a set of nodes for bounding box geometry

    :param fname: Filename under Meshes directory
    :return V: [N, 2] array of nodes in x-y form
    """
    V = np.genfromtxt(os.path.dirname(os.getcwd()) + '\Meshes\\' + fname,
                   dtype=np.double, delimiter=',')

    return V


def internal_walls(wall_points, looped):
    """Returns an [N, 4] array of x-y coordinates that make up each edge for a given path of edges.

    :param wall_points: [N, 2] array of x-y coordinate pairs that make up the nodes for all edges
    :param looped: Boolean - True for when the wall loops back from the point to the first, false if the wall does not
    """
    wall_edges = []
    for i in range(wall_points.shape[0]):
        if i == (wall_points.shape[0]-1) and looped:
            wall_edges.append(np.array([wall_points[-1], wall_points[0]]).flatten())
        else: wall_edges.append(np.array([wall_points[i], wall_points[i+1]]).flatten())
    return np.array(wall_edges)
