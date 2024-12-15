import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import ticker
from BARBECUES.utilities.fluids import calculateMach

def plotMach(V, E, stateVectors, y, fname):
    """Contour/tricolor plot of the Mach number at each cell in the mesh.

    :param V: Nodes and their coordinates
    :param E: Element-to-node mapping
    :param stateVectors: [:, 4] Numpy array of state vectors, each row is 1 cell's state [rho, rho*u, rho*v, rho*E]
    :param y: Ratio of specific heats
    :param fname: filename for what the figure is saved under
    """
    figure = plt.figure(figsize=(12, 12))

    # Calculate Mach numbers to plot
    mach = calculateMach.calculateMach(stateVectors, y)

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