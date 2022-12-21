import preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
import helper


def plot_mesh(Mesh, fname):
    """Plots the mesh and the different boundary conditions in their respective colors.

    :param Mesh: Mesh dictionary
    :param fname: filename for what the figure is saved under
    """
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    # Plots all the triangles in the mesh in black
    plt.triplot(V[:,0], V[:,1], E, '-', color='black')

    # Plots all the boundaries in their unique color scheme
    for i in range(BE.shape[0]):
        if Mesh['Bname'][BE[i, 3]] == 'Wall':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='magenta')
        elif Mesh['Bname'][BE[i, 3]] == 'Inflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='red')
        elif Mesh['Bname'][BE[i, 3]] == 'Outflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='blue')
        elif Mesh['Bname'][BE[i, 3]] == 'Exit':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='cyan')

    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(f)


def plot_mach(Mesh, state, fname):
    """Contour/tricolor plot of the Mach number at each cell in the mesh.

    :param Mesh: Mesh dictionary
    :param state: Nx4 state vectory array
    :param fname: filename for what the figure is saved under
    """
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    figure = plt.figure(figsize=(12, 12))

    # Calculate Mach numbers to plot
    mach = helper.calculate_mach(state)

    # Mach number contour plotting
    plt.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=mach, shading='flat', cmap='jet')
    plt.autoscale(enable=None, axis="x", tight=True)
    plt.autoscale(enable=None, axis="y", tight=True)
    plt.tick_params(axis='both', labelsize=12)
    plt.title('Mach number')
    # Forces the colorbar to show - I don't know how it works, I got it off of stack exchange
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    # Save and close configurations
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    figure.tight_layout()
    plt.savefig(fname)
    plt.close(figure)


def plot_stagnation_pressure(Mesh, state, fname):
    """Contour/tricolor plot of the stagnation pressure at each cell in the mesh.

    :param Mesh: Mesh dictionary
    :param state: Nx4 state vectory array
    :param fname: filename for what the figure is saved under
    """
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    figure = plt.figure(figsize=(12, 12))

    # Calculating the stagnation pressure
    mach = helper.calculate_mach(state)
    stagnation_pressure = helper.calculate_stagnation_pressure(state, mach)
    normalized_stagnation_pressure = np.divide(stagnation_pressure, max(stagnation_pressure))
    ATPR = helper.calculate_atpr(normalized_stagnation_pressure, Mesh)

    # Stagnation pressure contour plotting
    plt.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=normalized_stagnation_pressure, shading='flat', cmap='jet')
    plt.autoscale(enable=None, axis="x", tight=True)
    plt.autoscale(enable=None, axis="y", tight=True)
    plt.tick_params(axis='both', labelsize=12)
    plt.title('Stagnation Pressure - ATPR @ Exit = {0}'.format(ATPR))
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    # Save and close configurations
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    figure.tight_layout()
    plt.savefig(fname)
    plt.close(figure)


def plot_mesh_flagged(Mesh, flagged_cells, fname):
    """Plots the mesh and also highlights the cells flagged for refinement.

    :param Mesh: Mesh dictionary
    :param flagged_cells: Flagged cell indices
    :param fname: Filename for what the figure is saved under
    """
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    # Plotting the mesh
    plt.triplot(V[:,0], V[:,1], E, '-', color='black')
    # Emphasize the boundary edges of the domain
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='black')
    # Plotting the flagged cells
    plt.triplot(V[:, 0], V[:, 1], Mesh['E'][flagged_cells], '-', color='red')

    # Save and close configurations
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)


def plot_grid(nodes, fname):
    """Scatter plot of the nodes for a given mesh.

    :param nodes: [x, y] coordinate pairs of all node locations
    :param fname: Filename for what the figure is saved under
    """
    f = plt.figure(figsize=(12, 12))
    plt.scatter(x=nodes[:, 0], y=nodes[:, 1])
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)


def plot_moc(Mesh, moc, fname):
    """Plots the characteristic lines as they reflect off of the walls in the domain when a MOC initialization is
    performed.

    :param Mesh: Mesh dictionary
    :param moc: List of characteristic lines and their reflection points
    :param fname: Filename for what the figure is saved under
    """
    V = Mesh['V']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    # Plots all the boundaries in their unique color scheme
    for i in range(BE.shape[0]):
        if Mesh['Bname'][BE[i, 3]] == 'Wall':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='magenta')
        elif Mesh['Bname'][BE[i, 3]] == 'Inflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='red')
        elif Mesh['Bname'][BE[i, 3]] == 'Outflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='blue')
        elif Mesh['Bname'][BE[i, 3]] == 'Exit':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='cyan')

    for i in range(len(moc)):
        plt.plot(moc[i][:, 0], moc[i][:, 1], linewidth=0.5, color='green')

    # Save and close configurations
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)


def plot_config(mesh, state, fname, i):
    """Plotting driver that makes all the plots according to the data configuration config.

    :param mesh: Mesh dictionary
    :param state: Nx4 state vectory array
    :param fname: filename for what the figure is saved under
    :param i: Adaptive cycle number, is 0 for no adaptations
    """
    if pp.data_con['plot_mesh']: plot_mesh(mesh, fname + str(i) + '.png')
    if pp.data_con['plot_mach']: plot_mach(mesh, state, fname + '_M_' + str(i) + '.png')
    if pp.data_con['plot_stag_press']: plot_stagnation_pressure(mesh, state, fname + '_p0_' + str(i) + '.png')
    if pp.data_con['plot_residuals']: pass
