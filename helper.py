import numpy as np
import math

import flux


def calc_stag_pres(state, mach, fluid):
    """Calculates the stagnation pressure in each cell

    :param state: Nx4 array of state variables - one per cell
    :param fluid: Class for the current working fluid
    :param mach: Nx1 array of the local Mach number in each cell
    :return: Nx1 array of stagnation pressures - one per cell
    """
    stagnation_pressure = np.zeros(len(state[:, 0]))
    for i in range(len(state[:, 0])):
        pressure = calc_pressure(state[i], fluid)
        stagnation_pressure[i] = pressure * np.power(1 + (fluid.y - 1) / 2 * np.power(mach[i], 2), fluid.y / (fluid.y - 1))

    return stagnation_pressure


def calc_atpr(stag_pressure, mesh):
    """Calculates the average total pressure recovered along the exit plane of the isolater.

    :param stag_pressure: Nx1 array of stagnation pressure
    :param mesh: Mesh
    :return: Double that is the average total pressure recovery (ATPR)
    """
    # Boundary Edges at the isolater exit
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

    # Do the integral equation - but it has to be done numerically via a sum
    atpr = 1 / d / freestream_stagnation * np.sum(np.multiply(boundary_stagnation, delta_y))

    return atpr


def mach_calc(state, fluid):
    """Calculates the Mach number from the given state.

    :param state: Local state vector in a given cell
    :param fluid: Working fluid
    :return: Returns mach number
    """
    mach = np.zeros(len(state[:, 0]))
    for i in range(len(state[:, 0])):
        # Velocity
        u = state[i, 1] / state[i, 0]
        v = state[i, 2] / state[i, 0]
        q = np.linalg.norm([u, v])

        # Enthalpy
        h = (state[i, 3] + calc_pressure(state[i], fluid)) / state[i, 0]

        # Speed of sound
        c = math.sqrt((fluid.y - 1) * (h - q ** 2 / 2))

        mach[i] = q / c

    return mach


def mach_calc_single(state, fluid):
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
    h = (state[3] + calc_pressure(state, fluid)) / state[0]

    # Speed of sound
    c = math.sqrt((fluid.y - 1) * (h - q ** 2 / 2))

    mach = q / c

    return mach


def calc_pressure(state, fluid):
    """Calculates local static pressure from the given state.

    :param state: 1x4 array of Euler state variables
    :param fluid: Class of working fluid
    :return: pressure: local static pressure
    """
    pressure = (fluid.y - 1) * (state[3] - 0.5 * state[0] * ((state[1] / state[0]) ** 2 + (state[2] / state[0]) ** 2))

    return pressure


def freestream_state(config):
    """Generates a single state vector of the freestream configuration for the given freestream configuration and
    working fluid classes

    :param config: Class containing freestream and fluid information
    :return: state: 4 element array that has the freestream condition
    """
    state = np.array([1,
                      config.M * math.cos(config.a * math.pi / 180),
                      config.M * math.sin(config.a * math.pi / 180),
                      1 / ((config.y - 1) * config.y) + config.M ** 2 / 2])

    return state