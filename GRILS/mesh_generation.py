import geometry_generation as geom_gen
import numpy as np
import os
import distmesh as dm


# TODO: Go over the GRILS program and add some proper Python Linting to make it look pretty
def main():
    # Define the size of the bounding box of the computational domain, L can be used if a square domain is desired
    L = 4
    bbox_out = np.array([-L, L, -L, L])

    # Bounding box wall edges
    bbox_walls = geom_gen.sqr_bbox_wall_gen(2*L, 2*L, 0.2)

    # Circle
    interior_walls = geom_gen.circle_bbox_wall_gen(0.25, 15)

    # Flat Plate
    interior_walls = geom_gen.sqr_bbox_wall_gen(2, 2, 0.2)

    # Triangle
    test_tri = geom_gen.triangle_bbox_wall_gen(np.array((-2., 1.)), np.array((1., 0.75)), np.array((1., 0.6)), 0.1)


    # Diamond
    test_foil = geom_gen.diamond_bbox_wall_gen(2, 10, 0.1)

    # CSV
    test_csv = geom_gen.csv_bbox_Wall_gen('small_square.csv')

    # Anonymous function that serves as a signed distance function to determine how to move points to generate a "high
    # quality mesh"; nested ddiff statements can be used to overlap shapes and do boolean intersections
    sdf = lambda p: dm.ddiff(dm.dpoly(bbox_walls, p),
                             dm.dpoly(test_foil, p))

    # Generate nodes and cells in the domain via a level-set based algorithm
    # V, T = dm.distmesh2d(sdf, 0.25, bbox_out, np.vstack((bbox_walls, interior_walls)), 0.2, 1.2, 6e-1, 1e-3)

    # Writes the mesh out to the "Meshes" directory
    # write_mesh(bbox_out, V, T, internal_walls(interior_walls, True), None, 'demo_plate.gri')

    # Diamond airfoil generation
    V, T = dm.distmesh2d(sdf, 0.15, bbox_out, np.vstack((bbox_walls, test_foil)), 0.2, 1.2, 1e-3, 1e-3)

    # Triangle Test generation
    # V, T = dm.distmesh2d(sdf, 0.15, bbox_out, np.vstack((bbox_walls, test_tri)), 0.2, 1.2, 6e-1, 1e-3)

    # CSV Read Test generation
    # V, T = dm.distmesh2d(sdf, 0.2, bbox_out, np.vstack((bbox_walls, test_csv)), 0.2, 1.2, 6e-1, 1e-3)

    # Writes the mesh out to the "Meshes" directory
    write_mesh(bbox_out, V, T, internal_walls(test_foil, True), None, 'diamond_airfoil.gri')
    return


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


def write_mesh(bbox, V, T, interior=None, exit=None, fname='mesh.gri'):
    """Writes the newly made mesh as a *.gri file in the ../Meshes/ directory if it exists and creates it if it doesn't

    :param bbox: [xmin, xmax, ymin, ymax] x-y coordinate pair array of edges that make up the bounding box of the domain
    :param V: [N, 2] x-y coordinate pair array of nodes
    :param T: [N, 3] Indices of V that make up the triangular elements
    :param interior: [N, 2] length list of x-y coordinate pairs of path that make up interior wall edges
    :param exit: [N, 2] length list of x-y coordinate pairs of path that make up exit wall edges
    :param fname: Filename to save the mesh as
    """
    eps = 1e-12

    inflow = []
    outflow = []
    exit = []
    wall = []

    if not os.path.isdir(os.path.join(os.getcwd(), '../Meshes/')):
        os.mkdir(os.path.join(os.getcwd(), '../Meshes/'))
    os.chdir(os.path.join(os.getcwd(), '../Meshes/'))

    interior_V = []
    if interior is not None:
        for i in range(interior.shape[0]):
            interior_V.append((np.argmin(np.abs(V - interior[i, 0:2]).sum(1)), np.argmin(np.abs(V - interior[i, 2::]).sum(1))))
    interior_V = np.array(interior_V)

    f = open(fname, 'w')

    f.write('{0} {1} 2\n'.format(V.shape[0], T.shape[0]))

    for node in V:
        f.write('%.15e %.15e\n' % (node[0], node[1]))

    for element in T:
        edge1 = np.array([element[0], element[1]])
        edge2 = np.array([element[1], element[2]])
        edge3 = np.array([element[2], element[0]])

        # x-y coordinate pairs for the edges
        Ve1 = V[edge1]
        Ve2 = V[edge2]
        Ve3 = V[edge3]

        # x-y coordinates of the midpoints of the edges
        Me1 = Ve1.sum(0) / 2
        Me2 = Ve2.sum(0) / 2
        Me3 = Ve3.sum(0) / 2

        # Check if midpoint x-coordinate is on LHS/RHS
        if abs(Me1[0] - bbox[0]) < eps: inflow.append([edge1])
        if abs(Me1[0] - bbox[1]) < eps: outflow.append([edge1])

        if abs(Me2[0] - bbox[0]) < eps: inflow.append([edge2])
        if abs(Me2[0] - bbox[1]) < eps: outflow.append([edge2])

        if abs(Me3[0] - bbox[0]) < eps: inflow.append([edge3])
        if abs(Me3[0] - bbox[1]) < eps: outflow.append([edge3])

        # Check if midpoint y-coordinate is on bottom/top
        if abs(Me1[1] - bbox[2]) < eps: inflow.append([edge1])
        if abs(Me1[1] - bbox[3]) < eps: outflow.append([edge1])

        if abs(Me2[1] - bbox[2]) < eps: inflow.append([edge2])
        if abs(Me2[1] - bbox[3]) < eps: outflow.append([edge2])

        if abs(Me3[1] - bbox[2]) < eps: inflow.append([edge3])
        if abs(Me3[1] - bbox[3]) < eps: outflow.append([edge3])

        # Now check if the edge midpoint lines up with a midpoint of an interior "wall" edge
        if interior is not None:
            if np.abs(edge1 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge1) - interior_V).sum(1).min() < eps:
                wall.append(edge1)
            if np.abs(edge2 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge2) - interior_V).sum(1).min() < eps:
                wall.append(edge2)
            if np.abs(edge3 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge3) - interior_V).sum(1).min() < eps:
                wall.append(edge3)
        # TODO: Setup (Even if basic) to account for zones that have ATPR (and remove it from outflow)

    # Number of exit BC's depending on what the inputs exist
    num_bc = 2
    if len(wall) != 0: num_bc += 1
    if len(exit) != 0: num_bc += 1
    f.write('{0}\n'.format(num_bc))

    # Write out internal boundary edges
    wall = np.squeeze(np.array(wall))
    f.write('{0} 2 Wall\n'.format(wall.shape[0]))
    for pair in wall:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    # Write out the external boundary edges
    inflow = np.squeeze(np.array(inflow))
    f.write('{0} 2 Inflow\n'.format(inflow.shape[0]))
    for pair in inflow:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    outflow = np.squeeze(np.array(outflow))
    f.write('{0} 2 Outflow\n'.format(outflow.shape[0]))
    for pair in outflow:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    # Write out the elements
    f.write('{0} 1 TriLagrange\n'.format(T.shape[0]))
    for element in T:
        temp_element = geom_gen.reorient_ccw(element[0], element[1], element[2], V)
        f.write('{0} {1} {2}\n'.format(temp_element[0] + 1, temp_element[1] + 1, temp_element[2] + 1))
    f.close()


if __name__ == "__main__":
    main()