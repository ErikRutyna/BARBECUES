import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plotMesh(V, E, BE, beNames, fname):
    """Plots the mesh and the different boundary conditions in their respective colors.

    :param V: Node positions
    :param E: Element-to-node mapping
    :param BE: Boundary edges
    :param beNames: Names of the boundary edges
    :param fname: filename for what the figure is saved under
    """
    f = plt.figure(figsize=(12,12))
    # Plots all the triangles in the mesh in black
    plt.triplot(V[:,0], V[:,1], E, '-', color='black')

    # Plots all the boundaries in their unique color scheme
    for i in range(BE.shape[0]):
        if beNames[BE[i, 3]] == 'Wall':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='magenta')
        elif beNames[BE[i, 3]] == 'Inflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='blue')
        elif beNames[BE[i, 3]] == 'Outflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='red')
        elif beNames[BE[i, 3]] == 'Exit':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='cyan')

    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(f)