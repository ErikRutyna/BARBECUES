import numpy as np
import plotmesh
import mesh_generation_helper as mgh
import matplotlib.pyplot as plt
import scipy.spatial as sp
import distmesh as dm

def main():
    # Two meshes to show off the abilities of the mesher - one to show meshing multiplie geometries, one to show off
    # making something useful for the fluid solver
    mesh1 = mgh.shapes(np.array([-1, -1, 1]), np.array([-1, 1, 1]), np.array([1, 0, 1, 4]))

    plate_outline = mgh.add_rectangle(0, 0, 2, 0.2, 10)
    # My DistMesh Vers
    mesh1_1 = dm.unit_circle_distmesh(2, 2, 0.2, np.array([-1, -1, 1]), np.array([-1, 1, 1]), np.array([1, 0, 1, 4]), mesh1)
    # mesh1_1 = dm.flat_plate_distmesh_shape(5, 2, 0.2, plate_outline, 2, 0.2)


    plt.triplot(mesh1_1[0][:, 0], mesh1_1[0][:, 1], mesh1_1[1])
    plt.plot(mesh1_1[0][:, 0], mesh1_1[0][:, 1], 'o')
    plt.show()
    plt.savefig('demo_circle.png')

if __name__ == "__main__":
    main()