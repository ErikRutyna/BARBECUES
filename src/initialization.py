import cell_geometry_formulas as cfg
import preprocess as pp
import numpy as np
import math


def initialize_boundary(mesh):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = pp.flight_con['freestream_mach_numer'] * math.cos(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*U
    initial_condition[:, 2] = pp.flight_con['freestream_mach_numer'] * math.sin(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (pp.fluid_con['y'] - 1) / pp.fluid_con['y'] + pp.flight_con['freestream_mach_numer'] ** 2 / 2 # Rho*E

    return initial_condition

def initialize_boundary_weak(mesh):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    m = pp.flight_con['freestream_mach_numer'] / 2
    initial_condition = np.zeros((len(mesh['E']), 4))

    initial_condition[:, 0] = 1  # Rho
    initial_condition[:, 1] = m * math.cos(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*U
    initial_condition[:, 2] = m * math.sin(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*V
    initial_condition[:, 3] = 1 / (pp.fluid_con['y'] - 1) / pp.fluid_con['y'] + m ** 2 / 2 # Rho*E

    return initial_condition


def initialize_boundary_dist(mesh):
    """Initializes the solution by setting everything based on the freestream
    mach number.

    :param mesh: Dictionary that contains mesh information
    :return: initial_condition: np.array of the initial condition of the mesh
    """
    initial_condition = np.zeros((mesh['E'].shape[0], 4))
    x_min = (mesh['V'][:, 0]).min()
    x_max = (mesh['V'][:, 0]).max()
    y_min = (mesh['V'][:, 1]).min()
    y_max = (mesh['V'][:, 1]).max()

    centroids = cfg.centroid(mesh)

    for i in range(mesh['E'].shape[0]):
        if centroids[i, 0] <= 0:
            x_scale = centroids[i, 0] / x_min
        else:
            x_scale = centroids[i, 0] / x_max

        if centroids[i, 1] <= 0:
            y_scale = centroids[i, 1] / y_min
        else:
            y_scale = centroids[i, 1] / y_max

        avg_scale = (x_scale + y_scale) / 2

        m_init = pp.flight_con['freestream_mach_numer'] * avg_scale

        initial_condition[i, 0] = 1  # Rho
        initial_condition[i, 1] = m_init * math.cos(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*U
        initial_condition[i, 2] = m_init * math.sin(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*V
        initial_condition[i, 3] = 1 / (pp.fluid_con['y'] - 1) / pp.fluid_con['y'] + m_init ** 2 / 2 # Rho*E

    return initial_condition


def init_state_mach(m):
    """Initializes the state vector for a local mach number in that state.

    :param m: Local cell's Mach number
    :return:
    """
    initial_condition = np.zeros((4))

    initial_condition[0] = 1  # Rho
    initial_condition[1] = m * math.cos(pp.flight_con['angle_of_attack'] * math.pi / 180)# Rho*U
    initial_condition[2] = m * math.sin(pp.flight_con['angle_of_attack'] * math.pi / 180) # Rho*V
    initial_condition[3] = 1 / (pp.fluid_con['y'] - 1) / pp.fluid_con['y'] + (m ** 2) / 2 # Rho*E

    return initial_condition


def generate_freestream_state():
    """Generates a single state vector of the freestream configuration for the given flow conditions.

    :param M: Freestream Mach number
    :param a: Freestream angle of attack
    :return: state: 4 element array that has the freestream condition
    """
    state = np.array([1,
                      pp.flight_con['freestream_mach_numer'] * math.cos(pp.flight_con['angle_of_attack'] * math.pi / 180),
                      pp.flight_con['freestream_mach_numer'] * math.sin(pp.flight_con['angle_of_attack'] * math.pi / 180),
                      1 / ((pp.fluid_con['y'] - 1) * pp.fluid_con['y']) + pp.flight_con['freestream_mach_numer'] ** 2 / 2])
    return state