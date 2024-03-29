import geometryGeneration as geomGen
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import distmesh as dm
import numpy as np
import os


def main():
    """This main function can be used for mesh generation. Various examples have been provided inside the function and
    can be used as a reference when one might want to make their own mesh. The process follows a similar pattern for
    any mesh generated this way.

    1. The outer bounding box (outer edges of the computational domain) are defined
    2. The outer bounding box has its edges discretized
    3. An internal shape is generated and has its edges discretized into an array of points, OR a set of points is read
        in from a .csv file
    4. An anonymous (lambda function) signed distance function is generated that can be used to find the closest point
        inside the domain to the nearest boundary. For most cases of "object-in-flow" this is a boolean intersection of
        two polygon signed distance functions. This example is shown
    5. The distmesh algorithm is called. The default parameters have been optimized, but can be varied depending on how
        much the user cares about mesh quality. It is recommended to have a desired edge of length approximately equal
        to 1/2 the smallest edge length. Smaller iteration count is used as a default since 5 examples are run.
    6. The mesh is written out to file.

    Returns
    -------
    Writes out the mesh in ../Meshes/ directory in the *.gri format
    """
    # Step 1. Defining the Bounding Box of the computational domain. For all examples provided, we will be using the
    # same sized computational domain, a square with side length 2*L.
    L = 4
    compDomainBoundingBox = np.array([-L, L, -L, L])

    # Step 2. The bounding box of the computational domain is discretized. It is recommended to use approximately the
    # same value for discretizing both the computational domain and the desired edge length in distmesh.
    compDomainBoundingBoxEdges = geomGen.sqr_bbox_wall_gen(2 * L, 2 * L, 0.2)

    # ------------------------------------------------------------------------------------
    # Steps 3 - 6 will be unique to each piece of geometry. Some examples are shown below.
    # ------------------------------------------------------------------------------------

    # Circle
    # Step 3. Generating the circle with a radius of 1 unit and a theta-step of 15 degrees
    demoCircle = geomGen.circle_bbox_wall_gen(1, 15)
    # Step 4. Anonymous signed distance function for a boolean intersection of the bounding box square and the circle
    demoCircleSDF = lambda p: dm.ddiff(dm.dpoly(compDomainBoundingBoxEdges, p),
                             dm.dpoly(demoCircle, p))
    # Step 5. Calling distmesh
    V, T = dm.distmesh2d(demoCircleSDF, 0.2, compDomainBoundingBox, np.vstack((compDomainBoundingBoxEdges, demoCircle)),
                         0.15, 1.1, 100)
    # Step 6. Writing the mesh
    writeMesh(compDomainBoundingBox, V, T, geomGen.internalWalls(demoCircle, True), None, 'demoCircle')


    # Flat Plate
    # Step 3. Generating a flat plate with a length of 2 units and a width of 0.2 units and a spacing of 0.1 units
    demoPlate = geomGen.sqr_bbox_wall_gen(2, 0.2, 0.1)
    # Step 4. Anonymous signed distance function for a boolean intersection of the bounding box square and the circle
    demoPlateSDF = lambda p: dm.ddiff(dm.dpoly(compDomainBoundingBoxEdges, p),
                             dm.dpoly(demoPlate, p))
    # Step 5. Calling distmesh
    V, T = dm.distmesh2d(demoPlateSDF, 0.2, compDomainBoundingBox, np.vstack((compDomainBoundingBoxEdges, demoPlate)),
                         0.15, 1.1, 100)
    # Step 6. Writing the mesh
    writeMesh(compDomainBoundingBox, V, T, geomGen.internalWalls(demoPlate, True), None, 'demoPlate')


    # Triangle
    # Step 3. Generating a triangle with the 3 given points and a distance step size of 0.1 units
    demoTri = geomGen.triangle_bbox_wall_gen(np.array((-2., 1.)), np.array((1., 0.75)), np.array((1., 0.6)), 0.1)
    # Step 4. Anonymous signed distance function for a boolean intersection of the bounding box square and the circle
    demoTriSDF = lambda p: dm.ddiff(dm.dpoly(compDomainBoundingBoxEdges, p),
                             dm.dpoly(demoTri, p))
    # Step 5. Calling distmesh
    V, T = dm.distmesh2d(demoTriSDF, 0.2, compDomainBoundingBox, np.vstack((compDomainBoundingBoxEdges, demoTri)),
                         0.15, 1.1, 100)
    # Step 6. Writing the mesh
    writeMesh(compDomainBoundingBox, V, T, geomGen.internalWalls(demoTri, True), None, 'demoTriangle')


    # Diamond
    # Step 3. Generating a diamond with a chord length of 2 units and a half angle of 10 degrees with distance step size of 0.1 units
    demoDiamond = geomGen.diamond_bbox_wall_gen(2, 10, 0.1)
    # Step 4. Anonymous signed distance function for a boolean intersection of the bounding box square and the circle
    demoDiamondSDF = lambda p: dm.ddiff(dm.dpoly(compDomainBoundingBoxEdges, p),
                             dm.dpoly(demoDiamond, p))
    # Step 5. Calling distmesh
    V, T = dm.distmesh2d(demoDiamondSDF, 0.2, compDomainBoundingBox, np.vstack((compDomainBoundingBoxEdges, demoDiamond)),
                         0.15, 1.1, 100)
    # Step 6. Writing the mesh - this one has a line segment on the entire RHS wall as an "Exit" condition
    writeMesh(compDomainBoundingBox, V, T, geomGen.internalWalls(demoDiamond, True), None, 'demoDiamond')
    return


def writeMesh(bbox, V, T, interior=None, exitBCList=None, fname='mesh'):
    """Writes the newly made mesh as a *.gri file in the ../Meshes/ directory if it exists and creates it if it doesn't

    :param bbox: [xmin, xmax, ymin, ymax] x-y coordinate pair array of edges that make up the bounding box of the domain
    :param V: [N, 2] x-y coordinate pair array of nodes
    :param T: [N, 3] Indices of V that make up the triangular elements
    :param interior: [N, 2] Numpy array of x-y coordinate pairs of path that make up interior wall edges
    :param exitBC: N-length list of 4-element Numpy arrays of x-y coordinates that make up the line segments that define the exit condition [x1 y1 x2 y2]
    :param fname: Filename to save the mesh as
    """
    eps = 1e-12

    inflow = []
    outflow = []
    exitBC = []
    wall = []

    # Look for the mesh output directory, if it cannot find it, make it and move the working directory there
    if not os.path.isdir(os.path.join(os.getcwd(), '../Meshes/')):
        os.mkdir(os.path.join(os.getcwd(), '../Meshes/'))
    os.chdir(os.path.join(os.getcwd(), '../Meshes/'))

    interior_V = []
    if interior is not None:
        for i in range(interior.shape[0]):
            interior_V.append((np.argmin(np.abs(V - interior[i, 0:2]).sum(1)), np.argmin(np.abs(V - interior[i, 2::]).sum(1))))
    interior_V = np.array(interior_V)

    f = open(fname +'.gri', 'w')

    # Node writing
    f.write('{0} {1} 2\n'.format(V.shape[0], T.shape[0]))

    for node in V:
        f.write('%.15e %.15e\n' % (node[0], node[1]))

    # Looping over the elements to find out where to apply boundary conditions
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

        # Tracker to make sure edges are accounted for, this gets updated when they've been assigned to a BC flag array
        edgeFlagTracker = np.array((False, False, False))

        # Check if the edge is an interior wall edge
        if interior is not None:
            if np.abs(edge1 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge1) - interior_V).sum(1).min() < eps:
                wall.append(edge1)
                edgeFlagTracker[0] = True
            if np.abs(edge2 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge2) - interior_V).sum(1).min() < eps:
                wall.append(edge2)
                edgeFlagTracker[1] = True
            if np.abs(edge3 - interior_V).sum(1).min() < eps or np.abs(np.flip(edge3) - interior_V).sum(1).min() < eps:
                wall.append(edge3)
                edgeFlagTracker[2] = True

        # Check if the edge is a supersonic exit edge
        if exitBCList is not None:
            for i in range(len(exitBCList)):
                startExitBC = np.array((exitBCList[i][0], exitBCList[i][2]))
                endExitBC = np.array((exitBCList[i][1], exitBCList[i][3]))
                if geomGen.onLineSegment(startExitBC, endExitBC, Me1) and not edgeFlagTracker[0]:
                        exitBC.append(edge1)
                        edgeFlagTracker[0] = True
                if geomGen.onLineSegment(startExitBC, endExitBC, Me2) and not edgeFlagTracker[1]:
                        exitBC.append(edge2)
                        edgeFlagTracker[1] = True
                if geomGen.onLineSegment(startExitBC, endExitBC, Me3) and not edgeFlagTracker[2]:
                    exitBC.append(edge3)
                    edgeFlagTracker[2] = True

        # Check if the edge is an outflow edge, RIGHT/TOP edges of bounding box
        if abs(Me1[0] - bbox[1]) < eps and not edgeFlagTracker[0]: outflow.append([edge1]); edgeFlagTracker[0] = True
        if abs(Me2[0] - bbox[1]) < eps and not edgeFlagTracker[1]: outflow.append([edge2]); edgeFlagTracker[1] = True
        if abs(Me3[0] - bbox[1]) < eps and not edgeFlagTracker[2]: outflow.append([edge3]); edgeFlagTracker[2] = True

        if abs(Me1[1] - bbox[3]) < eps and not edgeFlagTracker[0]: outflow.append([edge1]); edgeFlagTracker[0] = True
        if abs(Me2[1] - bbox[3]) < eps and not edgeFlagTracker[1]: outflow.append([edge2]); edgeFlagTracker[1] = True
        if abs(Me3[1] - bbox[3]) < eps and not edgeFlagTracker[2]: outflow.append([edge3]); edgeFlagTracker[2] = True

        # Check if the edge is an inflow edge, LEFT/BOT edges of bounding box
        if abs(Me1[0] - bbox[0]) < eps and not edgeFlagTracker[0]: inflow.append([edge1]); edgeFlagTracker[0] = True
        if abs(Me2[0] - bbox[0]) < eps and not edgeFlagTracker[1]: inflow.append([edge2]); edgeFlagTracker[1] = True
        if abs(Me3[0] - bbox[0]) < eps and not edgeFlagTracker[2]: inflow.append([edge3]); edgeFlagTracker[2] = True

        if abs(Me1[1] - bbox[2]) < eps and not edgeFlagTracker[0]: inflow.append([edge1]); edgeFlagTracker[0] = True
        if abs(Me2[1] - bbox[2]) < eps and not edgeFlagTracker[1]: inflow.append([edge2]); edgeFlagTracker[1] = True
        if abs(Me3[1] - bbox[2]) < eps and not edgeFlagTracker[2]: inflow.append([edge3]); edgeFlagTracker[2] = True

    # Number of exit BC's depending on what the inputs exist
    num_bc = 2
    if len(wall) != 0:
        num_bc += 1
    if len(exitBC) != 0:
        num_bc += 1

    f.write('{0}\n'.format(num_bc))

    # Write out internal boundary edges
    wall = np.squeeze(np.array(wall))
    f.write('{0} 2 Wall\n'.format(wall.shape[0]))
    for pair in wall:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    # Write out the external boundary edges
    if len(exitBC) != 0:
        exitBC = np.squeeze(np.array(exitBC))
        f.write('{0} 2 Exit\n'.format(exitBC.shape[0]))
        for pair in exitBC:
            f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    outflow = np.squeeze(np.array(outflow))
    f.write('{0} 2 Outflow\n'.format(outflow.shape[0]))
    for pair in outflow:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    inflow = np.squeeze(np.array(inflow))
    f.write('{0} 2 Inflow\n'.format(inflow.shape[0]))
    for pair in inflow:
        f.write('%i %i\n' % (pair[0] + 1, pair[1] + 1))

    # Write out the elements
    f.write('{0} 1 TriLagrange\n'.format(T.shape[0]))
    for element in T:
        temp_element = geomGen.reorient_ccw(element[0], element[1], element[2], V)
        f.write('{0} {1} {2}\n'.format(temp_element[0] + 1, temp_element[1] + 1, temp_element[2] + 1))
    f.close()

    # Plot the actual mesh for visualization purposes
    fig = plt.figure(figsize=(12,12))
    # Plots all the triangles in the mesh in black
    plt.triplot(V[:,0], V[:,1], T, '-', color='black', linewidth=0.25)
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.savefig(fname + '.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()