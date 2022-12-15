import numpy as np
import math


def calculate_atpr(stag_pressure, mesh):
    """Calculates the average total pressure recovered (ATPR) along the exit plane of the isolator.

    :param stag_pressure: Nx1 array of stagnation pressure
    :param mesh: Mesh to get boundary edges
    :return: Double that is the average total pressure recovery (ATPR)
    """
    # Boundary Edges at the isolator exit
    exit_edges = mesh['BE'][mesh['BE'][:, 3] == 1, :]

    # Boundary edge stagnation pressures
    boundary_stagnation = stag_pressure[exit_edges[:, 2]]

    # Freestream stagnation pressure - maximum of all values should always be freestream
    freestream_stagnation = max(stag_pressure)

    d = 0
    delta_y = np.zeros(len(exit_edges))
    for i in range(len(delta_y)):
        # delta_y = y_nodeB - y_nodeA
        delta_y[i] = abs(mesh['V'][exit_edges[i, 1]][1] - mesh['V'][exit_edges[i, 0]][1])
        # Total length of the exit plane
        d += delta_y[i]

    # Numerical integration via summation over exit plane edges
    atpr = 1 / d / freestream_stagnation * np.sum(np.multiply(boundary_stagnation, delta_y))

    return atpr


def calculate_stagnation_pressure(state, mach, fluid):
    """Calculates the stagnation pressure for each state vector

    :param state: Nx4 array of state variables - one per cell [rho, rho*u, rho*v, rho*E]
    :param fluid: Class for the current working fluid for the gamma value
    :param mach: Nx1 array of the local Mach number in each cell
    :return: Nx1 array of stagnation pressures - one per cell
    """
    # Constants to be used in the stagnation pressure formula
    c_1 = (fluid.y - 1) / 2
    c_2 = fluid.y / (fluid.y - 1)

    # Local static pressure per state vector
    pressure = calculate_static_pressure(state, fluid)
    # Stagnation pressure
    stagnation_pressure = np.multiply(np.power(1 + c_1 * np.power(mach, 2), c_2), pressure)

    return stagnation_pressure


def calculate_mach(state, fluid):
    """Calculates the Mach number for each unique state vector.

    :param state: Local state vector in a given cell
    :param fluid: Working fluid
    :return: Returns mach number
    """
    # Velocity magnitude
    q = np.sqrt(np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2))

    # Use speed of sound, c = sqrt(y*p/rho)
    p = calculate_static_pressure(state, fluid)
    c = math.sqrt(fluid.y) * np.sqrt(np.divide(p, state[:, 0]))

    mach = np.divide(q, c)

    return mach


def calculate_mach_single(state, fluid):
    """Calculates Mach number for a single state vector slice - only exists because I was too lazy to fix the original
    mach_calc function to account for single slices and whole mesh arrays.

    :param state: Local state vector in a given cell
    :param fluid: Working fluid
    :return: Returns mach number
    """
    # Velocity
    u = state[1] / state[0]
    v = state[2] / state[0]
    q = np.linalg.norm([u, v])

    # Enthalpy
    h = (state[3] + calculate_static_pressure(state, fluid)) / state[0]

    # Speed of sound
    c = math.sqrt((fluid.y - 1) * (h - q ** 2 / 2))

    mach = q / c

    return mach


def calculate_static_pressure(state, fluid):
    """Calculates local static pressure from the given state.

    :param state: Nx4 array of Euler state variables
    :param fluid: Class of working fluid
    :return: pressure: local static pressure for each state vector
    """
    # Leading constant term
    c = (fluid.y - 1)
    # Velocity term
    q = np.power(np.divide(state[:, 1], state[:, 0]), 2) + np.power(np.divide(state[:, 2], state[:, 0]), 2)
    # p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = c * (state[:, 3] - 0.5 * np.multiply(state[:, 0], q))

    return pressure


def calculate_static_pressure_single(state, fluid):
    """Calculates local static pressure from the given state.

    :param state: Nx4 array of Euler state variables
    :param fluid: Class of working fluid
    :return: pressure: local static pressure for each state vector
    """
    # Leading constant term
    c = (fluid.y - 1)
    # Velocity term
    q = np.power(np.divide(state[1], state[0]), 2) + np.power(np.divide(state[2], state[0]), 2)
    # p = (y - 1) * (rho*E - 0.5 * rho * q^2)
    pressure = c * (state[3] - 0.5 * np.multiply(state[0], q))

    return pressure


def generate_freestream_state(config):
    """Generates a single state vector of the freestream configuration for the given flow conditions.

    :param config: Class containing freestream and fluid information
    :return: state: 4 element array that has the freestream condition
    """
    state = np.array([1,
                      config.M * math.cos(config.a * math.pi / 180),
                      config.M * math.sin(config.a * math.pi / 180),
                      1 / ((config.y - 1) * config.y) + config.M ** 2 / 2])
    return state