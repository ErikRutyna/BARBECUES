import preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
import helper
from matplotlib import ticker

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


def plot_mach(Mesh, state, fname, config):
    """Contour/tricolor plot of the Mach number at each cell in the mesh.

    :param Mesh: Mesh dictionary
    :param state: Nx4 state vectory array
    :param fname: filename for what the figure is saved under
    """
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    figure = plt.figure(figsize=(12, 12))

    # Calculate Mach numbers to plot
    mach = helper.calculate_mach(state, config['y'])

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
    cb = plt.colorbar(PCM, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    # Save and close configurations
    plt.axis('equal')
    figure.tight_layout()
    plt.savefig(fname)
    plt.close(figure)


def plot_stagnation_pressure(mesh, state, fname, config):
    """Contour/tricolor plot of the stagnation pressure at each cell in the mesh.

    :param mesh: Mesh dictionary
    :param state: Nx4 state vectory array
    :param fname: filename for what the figure is saved under
    """
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    figure = plt.figure(figsize=(12, 12))

    # Calculating the stagnation pressure
    mach = helper.calculate_mach(state, config['y'])
    stagnation_pressure = helper.calculate_stagnation_pressure(state, mach, config['y'])
    normalized_stagnation_pressure = np.divide(stagnation_pressure, max(stagnation_pressure))
    ATPR = helper.calculate_atpr(V, BE, normalized_stagnation_pressure)

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
    cb = plt.colorbar(PCM, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    # Save and close configurations
    plt.axis('equal')
    figure.tight_layout()
    plt.savefig(fname)
    plt.close(figure)


def plot_mesh_flagged(mesh, flagged_cells, fname):
    """Plots the mesh and also highlights the cells flagged for refinement.

    :param Mesh: Mesh dictionary
    :param flagged_cells: Flagged cell indices
    :param fname: Filename for what the figure is saved under
    """
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    f = plt.figure(figsize=(12,12))
    # Plotting the mesh
    plt.triplot(V[:,0], V[:,1], E, '-', color='black')
    # Emphasize the boundary edges of the domain
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='black')
    # Plotting the flagged cells
    plt.triplot(V[:, 0], V[:, 1], mesh['E'][flagged_cells], '-', color='red')

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


def plot_moc(mesh, moc, fname):
    """Plots the characteristic lines as they reflect off of the walls in the domain when a MOC initialization is
    performed.

    :param mesh: Mesh dictionary
    :param moc: List of characteristic lines and their reflection points
    :param fname: Filename for what the figure is saved under
    """
    V = mesh['V']; BE = mesh['BE']
    f = plt.figure(figsize=(12,12))
    # Plots all the boundaries in their unique color scheme
    for i in range(BE.shape[0]):
        if mesh['Bname'][BE[i, 3]] == 'Wall':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='magenta')
        elif mesh['Bname'][BE[i, 3]] == 'Inflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='red')
        elif mesh['Bname'][BE[i, 3]] == 'Outflow':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='blue')
        elif mesh['Bname'][BE[i, 3]] == 'Exit':
            plt.plot(V[BE[i, 0:2], 0], V[BE[i, 0:2], 1], '-', linewidth=2, color='cyan')

    for i in range(len(moc)):
        plt.plot(moc[i][:, 0], moc[i][:, 1], linewidth=0.5, color='green')

    # Save and close configurations
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)


def plot_residuals(residuals):
    """Plots all the residuals for the simulation.

    :param residuals: Nx4 residual array
    """
    f = plt.figure(figsize=(12, 12))

    plt.plot(np.array(range(residuals.shape[0])) + 1, residuals[:, 0], color='blue',    label='Continuity')
    plt.plot(np.array(range(residuals.shape[0])) + 1, residuals[:, 1], color='green',   label='X-Momentum')
    plt.plot(np.array(range(residuals.shape[0])) + 1, residuals[:, 2], color='purple',  label='Y-Momentum')
    plt.plot(np.array(range(residuals.shape[0])) + 1, residuals[:, 3], color='red',     label='Energy', marker="o")
    plt.plot(np.array(range(residuals.shape[0])) + 1, residuals[:, 4], color='black',   label='L1 - Norm', marker=".")

    plt.legend()
    plt.yscale('log')

    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('residuals.png')
    plt.close(f)


def plot_performance(coefficients):
    """Plots the performance coefficients of the simulation as a function of iteration number.

    :param coefficients: Nx4 array of coefficients [cd, cl, cmx, atpr]
    """
    f, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].plot(np.array(range(coefficients.shape[0])) + 1, coefficients[:, 0], 'tab:blue')
    axs[0, 0].set_title('Drag Coefficient ($C_D$)')

    axs[0, 1].plot(np.array(range(coefficients.shape[0])) + 1, coefficients[:, 1], 'tab:green')
    axs[0, 1].set_title('Lift Coefficient ($C_L$)')

    axs[1, 0].plot(np.array(range(coefficients.shape[0])) + 1, coefficients[:, 2], 'tab:purple')
    axs[1, 0].set_title('Pitching Moment Coefficient ($C_{mx}$)')

    axs[1, 1].plot(np.array(range(coefficients.shape[0])) + 1, coefficients[:, 3], 'tab:red')
    axs[1, 1].set_title('Average Total Pressure Recovery Factor ($ATPR$)')

    f.tight_layout()
    f.savefig('coefficients.png')
    plt.close(f)


def plot_config(mesh, state, residuals, coefficients, config, i):
    """Plotting driver that makes all the plots according to the data configuration config.

    :param mesh: Mesh dictionary {'E', 'V', 'IE', 'BE', 'Bname'}
    :param state: Nx4 state vector array
    :param residuals: Nx5 array of residuals [mass, x-momentum, y-momentum, energy, L1-Total]
    :param config: All simulation configuration options influding fluid information and filename
    :param i: Adaptive cycle number, is 0 for no adaptations
    """
    if config['plot_mesh']: plot_mesh(mesh, config['filename'] + str(i) + '.png')
    if config['plot_mach']: plot_mach(mesh, state, config['filename'] + '_M_' + str(i) + '.png', config)
    if config['plot_stag_press']: plot_stagnation_pressure(mesh, state, config['filename'] + '_p0_' + str(i) + '.png', config)
    if config['plot_residuals'] and residuals.shape[0] != 0: plot_residuals(residuals)
    if config['plot_performance'] and coefficients.shape[0] != 0: plot_performance(coefficients)
