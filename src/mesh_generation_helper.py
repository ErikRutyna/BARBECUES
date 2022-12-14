import numpy as np
import math


def shapes(circle, square, rectangle):
    """Demonstrate the mesher's ability to work on geometry of different types and various node spacings.

    :param circle: [x0, y0, r]
    :param square: [x0, y0, l]
    :param rectangle: [x0, y0, l, w]
    :return:
    """
    # Spacing parameter
    N = 20
    buffer_spacing = 6 / N / 3

    # Circle - created by discretizing the perimeters
    inner_circle = add_circle(circle[0], circle[1], circle[2], int(N/2), 0)
    # small_circle = add_circle(circle[0], circle[1], circle[2] + buffer_spacing * 3, N, 30 * math.pi / 180)
    # large_circle = add_circle(circle[0], circle[1], circle[2] + buffer_spacing * 5, N, 0)
    large_circ = [circle[0], circle[1], circle[2] + buffer_spacing * 5]

    # Square - create pairs by tracing perimeter slices
    inner_square = add_rectangle(square[0], square[1], square[2], square[2], int(N/8))
    # small_square = add_rectangle(square[0], square[1], square[2] + buffer_spacing * 3, square[2] + buffer_spacing * 3, int(N/2))
    # large_square = add_rectangle(square[0], square[1], square[2] + buffer_spacing * 5, square[2] + buffer_spacing * 5, int(N/2))
    large_sqr = [square[0], square[1], square[2] + buffer_spacing * 5]

    # Rectangle
    inner_rect = add_rectangle(rectangle[0], rectangle[1], rectangle[2], rectangle[3], int(N/2))
    # small_rect = add_rectangle(rectangle[0], rectangle[1], rectangle[2] + buffer_spacing * 3, rectangle[3] + buffer_spacing * 3, N)
    # large_rect = add_rectangle(rectangle[0], rectangle[1], rectangle[2] + buffer_spacing * 5, rectangle[3] + buffer_spacing * 5, N)
    large_rct = [rectangle[0], rectangle[1], rectangle[2] + buffer_spacing * 5, rectangle[3] + buffer_spacing * 5]

    shape_perimeters = np.vstack((inner_circle,
                                  inner_square,
                                  inner_rect))
    # Populates the domain with the other nodes that are not inside any of the 3 pieces of geometry
    # other_nodes = generate_grid(6, 6, N)

    return np.vstack((shape_perimeters))


def add_rectangle(x0, y0, l, w, n):
    """Creates a set of nodes that define the perimeter of a rectangle using n number of points to discretize the
    perimeter.

    :param x0: x-coordinate of centroid
    :param y0: y-coordinate of centroid
    :param l: Length of the rectangle (distance from one edge in y to other)
    :param w: Width of the rectangle (distance from one edge in x to other)
    :param n: Number of nodes to discretize the perimeter
    """
    n_l = 4
    n_w = 10

    # bottom left -> bottom right -> top right -> top left -> bottom left
    x_nodes = np.concatenate((np.linspace(x0 - (l/2), x0 + (l/2), n_w)[0:n_w-1], np.ones(n_l) * (x0 + (l/2)),
                              np.linspace(x0 + (l/2), x0 - (l/2), n_w)[0:n_w-1], np.ones(n_l) * (x0 - (l/2))))
    y_nodes = np.concatenate((np.ones(n_w) * (y0 - (w/2)), np.linspace(y0 - (w/2), y0 + (w/2), n_l)[0:n_l-1],
                              np.ones(n_w) * (y0 + (w/2)), np.linspace(y0 + (w/2), y0 - (w/2), n_l)[0:n_l-1]))

    rectangle_perimeter = np.zeros((x_nodes.shape[0], 2))
    rectangle_perimeter[:, 0] = x_nodes
    rectangle_perimeter[:, 1] = y_nodes

    return rectangle_perimeter


def add_circle(x0, y0, r, n, phase):
    """Creates a set of nodes that define the perimeter of a circle using n number of points to discretize the
    perimeter.

    :param x0: x-coordinate of the origin
    :param y0: y-coordinate of the origin
    :param r: Radius of the circle
    :param n: Number of points to discretize the circle by
    :param phase: Offset the start of angles by some small amount
    :return:
    """
    theta = np.linspace(0 + phase, 2*math.pi + phase, n)
    x_coords = (x0 + r * np.cos(theta))[0:n - 1]
    y_coords = (y0 + r * np.sin(theta))[0:n - 1]

    circle_perimeter = np.zeros((x_coords.shape[0], 2))
    circle_perimeter[:, 0] = x_coords
    circle_perimeter[:, 1] = y_coords

    return circle_perimeter


def generate_grid(x, y, N):
    """Generate grid points across the domain.

    :param x: x-length of domain
    :param y: y-length of domain
    :param circle: [x0, y0, r]
    :param square: [x0, y0, l]
    :param rectangle: [x0, y0, l, w]
    :param buffer_spacing: 1/3 of the average spacing (ds) between any node's x & y coordinates
    :return:
    """
    # Make the equidistant spacing of nodes
    x_coords = np.linspace(-x/2, x/2, N)
    y_coords = np.linspace(-y/2, y/2, N)
    xnodes, ynodes = np.meshgrid(x_coords, y_coords)
    xnodes[1::2, :] += x / N # Shifting to make triangles different

    # Flatten and zip them together to form a Nx2 array of coordinate pairs
    xnodes = xnodes.flatten()
    ynodes = ynodes.flatten()

    valid_nodes = np.zeros((xnodes.shape[0], 2))
    valid_nodes[:, 0] = xnodes
    valid_nodes[:, 1] = ynodes

    return valid_nodes


#-----------------------------------------------------------------------------------------------------------------------
def add_arc(origin, radius, angle_range, spacing):
    """Creates points that make a circular arc at origin with radius R that goes from angle_range[0] -> angle_range[1]

    :param origin: [x, y] coordinate pair for the origin of the circle
    :param radius: Radius of the circle that forms the arc
    :param angle_range: [start, end] range of angles to include in the arc in degrees
    :param spacing: Spacing for number of points that display the arc
    :return: Returns an updated list of nodes that has any nodes inside the circle removed from the array.
    """
    # Add points that make the perimeter of the circle/arc
    spacing_theta = radius * spacing
    thetas = np.linspace(angle_range[0], angle_range[1],
                         math.ceil(abs(angle_range[0] - angle_range[1]) / spacing_theta))

    arc_points = np.zeros((thetas.shape[0], 2))
    for i in range(thetas.shape[0]):
        arc_points[i, :] = np.array([radius * math.cos(thetas[i]), radius * math.sin(thetas[i])]) + origin

    return arc_points


def remove_on_rectangle(corners, points):
    """Checks if the point exists on the boundary for the perimeter of the rectangle.

    :param corners: 1D array representation of the corners, every set of 2 is a corner coordinate point
    :param points: The nodes of the domain
    :return: Returns an updated list of points that exist outside the perimeter of the rectangle
    """
    check_bottom = (points[:, 0] >= corners[0]) & (points[:, 0] <= corners[2]) & (points[:, 1] == corners[1])
    check_right = (points[:, 1] >= corners[3]) & (points[:, 1] <= corners[5]) & (points[:, 0] == corners[2])
    check_top = (points[:, 1] >= corners[3]) & (points[:, 1] <= corners[5]) & (points[:, 1] == corners[0])
    check_left = (points[:, 1] >= corners[3]) & (points[:, 1] <= corners[5]) & (points[:, 0] == corners[2])

    return points[np.logical_not(check_bottom & check_right & check_top & check_left)]


def remove_on_arc(x0, y0, radius, theta0, theta1, points):
    """Checks if the point exists on the wedge drawn by the radius and angles from the origin.

    :param x0: x-coordinate of origin
    :param y0: y-coordinate of origin
    :param radius: Radius of the arc
    :param theta0: Starting polar angle of the arc
    :param theta1: Ending polar angle of the arc
    :param points: Nodes of the domain
    :return: Returns an updated list of points that exist outside the perimeter of the arc
    """
    # Subtract out origin for reference
    points[:, 0] -= x0; points[:, 1] -= y0

    points_polar = np.zeros((points.shape[0], 2))
    # Convert to polar
    for i in range(points.shape[0]):
        points_polar[i, 0] = np.linalg.norm(points[i, :])
        points_polar[i, 1] = (math.atan2(points[i, 1], points[i, 0]) + 2*math.pi) % 2*math.pi

    # Add back origin
    points[:, 0] += x0; points[:, 1] += y0

    # Check for points within angle ranges and radius ranges
    radius_points = points_polar[:, 0] == radius
    radius_angles = np.logical_and(points_polar[:, 1] > theta0, points_polar[:, 1] < theta1)

    small_radius = points_polar[:, 0] < radius
    edge_angles = np.logical_and(points_polar[:, 1] == theta0, points_polar[:, 1] == theta1)

    points = points[np.logical_not(radius_points & radius_angles)]
    points = points[np.logical_not(small_radius & edge_angles)]

    return points


def check_in_rectangle2(corners, point):
    """Checks if the point exists in inside the perimeter of the rectangle.

    :param corners: 1D array representation of the corners, every set of 2 is a corner coordinate point
    :param point: The nodes of the domain
    :return: Returns an updated list of points that exist outside the perimeter of the rectangle
    """
    check_l_left = np.logical_and((point[0] > corners[0]), (point[1] > corners[1]))
    check_l_right = np.logical_and((point[0] < corners[2]), (point[1] > corners[3]))
    check_u_right = np.logical_and((point[0] < corners[4]), (point[1] < corners[5]))
    check_u_left = np.logical_and((point[0] > corners[6]), (point[1] < corners[7]))

    if check_u_left & check_l_left & check_u_right & check_l_right: return True
    else: return False


def check_in_arc(x0, y0, radius, theta0, theta1, point):
    """Checks if the point exists within the wedge drawn by the radius and angles from the origin.

    :param x0: x-coordinate of origin
    :param y0: y-coordinate of origin
    :param radius: Radius of the arc
    :param theta0: Starting polar angle of the arc
    :param theta1: Ending polar angle of the arc
    :param points: Nodes of the domain
    :return: Returns an updated list of points that exist outside the perimeter of the arc
    """
    # Subtract out origin for reference
    point[0] -= x0; point[1] -= y0

    points_polar = np.zeros((2))
    # Convert to polar
    points_polar[0] = np.linalg.norm(point)
    temp_angle = math.atan2(point[1], point[0])
    if temp_angle < 0: temp_angle = abs(temp_angle) + math.pi
    points_polar[1] = temp_angle

    # Add back origin
    point[:, 0] += x0; point[:, 1] += y0

    # Check for points within angle ranges and radius ranges
    small_radius = points_polar[0] < radius * 0.9999
    right_angles = np.logical_and(points_polar[1] >= theta0, points_polar[1] <= theta1)

    if small_radius & right_angles: return True
    else: return False





