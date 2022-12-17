import numpy as np
import math


def initialize_boundary(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = config.M * math.cos(config.a * math.pi / 180) # Rho*U
    initial_condition[:, 2] = config.M * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (config.y - 1) / config.y + (config.M) ** 2 / 2 # Rho*E

    return initial_condition

def initialize_boundary_weak(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    m = config.M / 2
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = m * math.cos(config.a * math.pi / 180) # Rho*U
    initial_condition[:, 2] = m * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (config.y - 1) / config.y + (m) ** 2 / 2 # Rho*E

    return initial_condition


def initialize_boundary_dist(mesh, config):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :param config: Config file for simulation containing information regarding fluid and freestream information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((mesh['E'].shape[0], 4))
    x_min = (mesh['V'][:, 0]).min()
    x_max = (mesh['V'][:, 0]).max()
    y_min = (mesh['V'][:, 1]).min()
    y_max = (mesh['V'][:, 1]).max()

    for i in range(mesh['E'].shape[0]):
        centroid = (mesh['V'][mesh['E'][i, 0]] + mesh['V'][mesh['E'][i, 1]] + mesh['V'][mesh['E'][i, 2]]) / 3

        if centroid[0] <= 0:
            x_scale = centroid[0] / x_min
        else:
            x_scale = centroid[0] / x_max

        if centroid[1] <= 0:
            y_scale = centroid[1] / y_min
        else:
            y_scale = centroid[1] / y_max

        avg_scale = (x_scale + y_scale) / 2

        m_init = config.M * avg_scale

        initial_condition[i, 0] = 1  # Rho
        initial_condition[i, 1] = m_init * math.cos(config.a * math.pi / 180) # Rho*U
        initial_condition[i, 2] = m_init * math.sin(config.a * math.pi / 180) # Rho*V
        initial_condition[i, 3] = 1 / (config.y - 1) / config.y + (m_init) ** 2 / 2 # Rho*E

    return initial_condition


def init_state_mach(m, config):
    """Initializes the state vector for a local mach number in that state.

    :param m: Local cell's Mach number
    :return:
    """
    initial_condition = np.zeros((4))

    initial_condition[0] = 1  # Rho
    initial_condition[1] = m * math.cos(config.a * math.pi / 180)# Rho*U
    initial_condition[2] = m * math.sin(config.a * math.pi / 180) # Rho*V
    initial_condition[3] = 1 / (config.y - 1) / config.y + (m ** 2) / 2 # Rho*E

    return initial_condition