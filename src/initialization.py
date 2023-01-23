import cell_geometry_formulas as cfg
from numba import njit
import numpy as np
import math


@njit(cache=True)
def initialize_boundary(N, M, a, y):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param N: Number of state vectors to make
    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: initial_condition: np.array of the freestream initial condition of the mesh
    """
    initial_condition = np.zeros((N, 4))

    initial_condition[:, 0] = 1                                 # rho
    initial_condition[:, 1] = M * math.cos(a * math.pi / 180)   # rho*u
    initial_condition[:, 2] = M * math.sin(a * math.pi / 180)   # rho*v
    initial_condition[:, 3] = 1 / (y - 1) / y + M ** 2 / 2      # rho*E

    return initial_condition


@njit(cache=True)
def initialize_boundary_weak(N, M, a, y):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param N: Number of state vectors to make
    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: initial_condition: np.array of the weakened freestream initial condition of the mesh
    """
    m = M / 2
    initial_condition = np.zeros((N, 4))

    initial_condition[:, 0] = 1                                 # rho
    initial_condition[:, 1] = m * math.cos(a * math.pi / 180)   # rho*u
    initial_condition[:, 2] = m * math.sin(a * math.pi / 180)   # rho*v
    initial_condition[:, 3] = 1 / (y - 1) / y + m ** 2 / 2      # rho*E

    return initial_condition


@njit(cache=True)
def initialize_boundary_dist_linear(E, V, M, a, y):
    """Initializes the solution by setting the local state vector to be a function of the freestream via linear based
    scaling using the cell's centroid.

    :param E: Element-2-Node matrix
    :param V: Node coordinates
    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: initial_condition: np.array of the position-scaled freestream initial condition of the mesh
    """
    initial_condition = np.zeros((E.shape[0], 4))
    x_min = (V[:, 0]).min()
    x_max = (V[:, 0]).max()
    y_min = (V[:, 1]).min()
    y_max = (V[:, 1]).max()

    centroids = cfg.centroid(E, V)

    for i in range(E.shape[0]):
        if centroids[i, 0] <= 0:
            x_scale = centroids[i, 0] / x_min
        else:
            x_scale = centroids[i, 0] / x_max

        if centroids[i, 1] <= 0:
            y_scale = centroids[i, 1] / y_min
        else:
            y_scale = centroids[i, 1] / y_max

        avg_scale = (x_scale + y_scale) / 2

        m_init = M * avg_scale

        initial_condition[i, 0] = 1                                     # Rho
        initial_condition[i, 1] = m_init * math.cos(a * math.pi / 180)  # Rho*U
        initial_condition[i, 2] = m_init * math.sin(a * math.pi / 180)  # Rho*V
        initial_condition[i, 3] = 1 / (y - 1) / y + m_init ** 2 / 2     # Rho*E

    return initial_condition


@njit(cache=True)
def initialize_boundary_dist_exp(E, V, M, a, y):
    """Initializes the solution by setting the local state vector to be a function of the freestream via exponential
     based scaling using the cell's centroid.

    :param E: Element-2-Node matrix
    :param V: Node coordinates
    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: initial_condition: np.array of the position-scaled freestream initial condition of the mesh
    """
    initial_condition = np.zeros((E.shape[0], 4))
    x_min = (V[:, 0]).min()
    x_max = (V[:, 0]).max()
    y_min = (V[:, 1]).min()
    y_max = (V[:, 1]).max()

    centroids = cfg.centroid(E, V)

    for i in range(E.shape[0]):
        r = np.sqrt(centroids[i, 0] ** 2 + centroids[i, 1] ** 2)

        m_init = M / (np.exp(-r) + 1)

        initial_condition[i, 0] = 1                                     # Rho
        initial_condition[i, 1] = m_init * math.cos(a * math.pi / 180)  # Rho*U
        initial_condition[i, 2] = m_init * math.sin(a * math.pi / 180)  # Rho*V
        initial_condition[i, 3] = 1 / (y - 1) / y + m_init ** 2 / 2     # Rho*E

    return initial_condition


@njit(cache=True)
def init_state_mach(M, a, y):
    """Initializes the state vector for a local mach number in that state.

    :param M: Cell's local Mach number
    :param a: Freestream angle of attack
    :param y: Ratio of specific heats - gamma
    :return: State vector [rho, rho*u, rho*v, rho*E]
    """
    state = np.array([1,
                      M * math.cos(a * math.pi / 180),
                      M * math.sin(a * math.pi / 180),
                      1 / ((y - 1) * y) + M ** 2 / 2])
    return state


def init_state(mesh, config):
    """Initializes the simulation based on the option provided in the simulation config.json file. See documentation for
    information regarding different initialization options.

    :param mesh: Read in *.gri mesh
    :param config: Runtime configuration
    :return: Nx4 array of 1x4 state vectors [rho, rho*U, rho*V, rho*E]
    """
    if config['init'] == 'freestream': state = initialize_boundary(mesh['E'].shape[0],
                                                                   config['freestream_mach_numer'],
                                                                   config['angle_of_attack'],
                                                                   config['y'])
    if config['init'] == 'weak': state = initialize_boundary_weak(mesh['E'].shape[0],
                                                                  config['freestream_mach_numer'],
                                                                  config['angle_of_attack'],
                                                                  config['y'])
    if config['init'] == 'linear': state = initialize_boundary_dist_linear(mesh['E'], mesh['V'],
                                                                      config['freestream_mach_numer'],
                                                                      config['angle_of_attack'],
                                                                      config['y'])
    if config['init'] == 'exp': state = initialize_boundary_dist_exp(mesh['E'], mesh['V'],
                                                                      config['freestream_mach_numer'],
                                                                      config['angle_of_attack'],
                                                                      config['y'])
    if config['init'] == 'moc': pass
    return state