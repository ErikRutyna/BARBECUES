import matplotlib.pyplot as plt
from readgri import readgri
import numpy as np
import helper


def plotmesh(Mesh, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    # plt.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=value, shading='flat')
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    E2 = []
    for ie in Mesh['IE']:
        if not ie[2] in E2:
            E2.append(ie[2])
        if not ie[3] in E2:
            E2.append(ie[3])
    # plt.scatter(V[:, 0], V[:, 1], color='red')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='red')
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)
#-----------------------------------------------------------------------------------------------------------------------
def plotmesh_values(Mesh, state, fluid, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    figure = plt.figure(figsize=(12, 12))

    mach = helper.mach_calc(state, fluid)
    stag_pres = helper.calc_stag_pres(state, mach, fluid)
    # Stagnation Pressure Recovery Factor - ATPR
    stag_pres_recovered = helper.calc_atpr(stag_pres, Mesh)

    # Mach number
    mach_plot = plt.subplot(2, 1, 1)
    mach_plot.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=mach, shading='flat', cmap='jet')
    # mach_plot.triplot(V[:,0], V[:,1], E, 'k-')
    mach_plot.autoscale(enable=None, axis="x", tight=True)
    mach_plot.autoscale(enable=None, axis="y", tight=True)
    mach_plot.tick_params(axis='both', labelsize=12)
    mach_plot.set_title('Mach # Contours')
    # Some code off of stack exchange to get colorbars to show - I fucking hate plotting with matplotlib
    # and I'm not sure how it works, but it does, so I'll take it
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    # Stagnation Pressure
    stag_plot = plt.subplot(2, 1, 2)
    stag_plot.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=np.divide(stag_pres, max(stag_pres)), shading='flat', cmap='jet')
    stag_plot.triplot(V[:,0], V[:,1], E, 'k-')
    stag_plot.autoscale(enable=None, axis="x", tight=True)
    stag_plot.autoscale(enable=None, axis="y", tight=True)
    stag_plot.tick_params(axis='both', labelsize=12)
    stag_plot.set_title('Total Pressure Contours - ATPR: {0}'.format(stag_pres_recovered))
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    figure.tight_layout()
    plt.savefig(fname)
    plt.close(figure)


def plot_mesh_flagged(Mesh, Mesh2, flagged_cells):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    # plt.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=value, shading='flat')
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    plt.triplot(V[:, 0], V[:, 1], Mesh2['E'][flagged_cells], 'b-')
    # plt.triplot(V[:, 0], V[:, 1], Mesh['E'][np.array([411, 416, 436, 443, 450, 475, 488, 501, 509, 514, 522,
    #        539, 543, 579, 656, 658, 666, 1553, 1573, 1578, 1588, 1595,
    #        1600, 1607, 1611, 1620, 1627, 1633, 1639, 1648, 1665, 1677, 1699,
    #        1709, 1720, 1735], dtype=int)], 'g-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='black')
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig('flagged_cells.png')
    plt.close(f)


def plot_grid(nodes, fname):
    """Plots the 2D grid of nodes in a simple plot - nothing fancy.

    :param nodes: [x, y] coordinate pairs of all node locations
    :param fname: Filename to save the figure as
    :return: Nothing - saves a file in the current directory
    """
    f = plt.figure(figsize=(12, 12))
    plt.scatter(x=nodes[:, 0], y=nodes[:, 1])
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)
    return

def plot_moc(Mesh, moc, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='red')
    for i in range(len(moc)):
        plt.plot(moc[i][:, 0], moc[i][:, 1], linewidth=0.5, color='blue')
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout()
    plt.savefig(fname)
    plt.close(f)
    return

#-----------------------------------------------------------------------------------------------------------------------


def main():
    Mesh = readgri('mesh0.gri')
    plotmesh(Mesh, [])

if __name__ == "__main__":
    main()
